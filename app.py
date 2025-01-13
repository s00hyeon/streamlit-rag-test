import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time

from typing import Optional
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
openai_api_key = st.secrets["OPENAI_API_KEY"]

def get_openai_api_key() -> Optional[str]:
    '''
    OpenAI API 키를 가져오는 함수
    우선순위: 1) Streamlit Secrets 2) 환경변수 3) 사용자 입력
    '''
    # 1. Streamlit Secrets에서 확인
    if 'OPENAI_API_KEY' in st.secrets:
        return st.secrets['OPENAI_API_KEY']
    
    # 2. 세션 상태에서 확인 (이전에 입력한 값)
    if 'openai_api_key' in st.session_state:
        return st.session_state.openai_api_key
    
    # 3. 사용자 입력 받기
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Get one at https://platform.openai.com/account/api-keys"
    )
    
    if api_key:
        # 세션 상태에 저장
        st.session_state.openai_api_key = api_key
        return api_key
    
    return None


from langchain_openai import OpenAIEmbeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large",)


# 함수: 데이터변환(dataframe -> CSV -> Document)
def save_df_to_docs(df, fn):
    '''
    params:
        df: dataframe
        fn: 저장할 csv/faiss 파일명
        embedding_function: 임베딩 함수
    return:
        List of documents
    '''
    # dataframe -> CSV
    fn_csv = f'{fn}.csv'
    df.to_csv(fn_csv, index=False)
    loader = CSVLoader(fn_csv, encoding='utf-8-sig')
    all_documents = loader.load_and_split()
    st.sidebar.write(f"{len(all_documents)} ROWS")
    return all_documents


# 함수: 벡터변환 후 로컬 저장
def save_docs_to_faiss(all_documents, embedding_function):
    '''
    params:
        all_documents: document 객체 리스트
        embedding_function: 임베딩 함수
    return:
        vectorstore: 벡터스토어 객체
        vector_result: 벡터스토어 저장된 모든 벡터값
    '''
    # FAISS DB에 벡터화하여 저장하기
    vectorstore = FAISS.from_documents(documents=all_documents, embedding=embedding_function)
    # 임베딩 추출 (문서의 모든 벡터 가져오기), 임베딩 변환 결과(벡터) 저장
    # 벡터 저장소의 인덱스에 있는 모든 벡터(0부터 마지막 벡터까지)를 재구성하여 원래의 벡터 형태로 가져오는 작업을 수행
    vector_result = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
    
    st.sidebar.write(f"VECTORSTORE LOADED")
    st.sidebar.write(f"ㄴ{len(vector_result)} Documents, {len(vector_result[0])} Dimensions")

    return vectorstore, vector_result

# 함수: 기존 df + vector_result 추가
def array_to_df(df, vector_result):
    
    array_data = np.array(vector_result)
    column_names = [f'vec_{i:04d}' for i in range(len(array_data[0]))]
    df_vector = pd.DataFrame(array_data, columns=column_names)    
    
    df_add = pd.concat([df, df_vector], axis=1)
    return df_add

# 콜백 함수 정의
def process_query():
    st.session_state.processed_df_q = None
    st.session_state.processed_df_temp = st.session_state.processed_df
    
    query_text = st.session_state.query_text
    compression_retriever = st.session_state.compression_retriever
    
    if query_text:        
        compression_result = compression_retriever.invoke(query_text)
        
        # query와 score 컬럼이 이미 있는지 확인
        if 'query' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(1, 'query', None)
        if 'score' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(2, 'score', None)
        
        # 유사도 계산결과 추가
        for idx, val in enumerate(compression_result):
            meta = compression_result[idx].metadata
            row = meta['row']
            val = compression_result[idx].state['query_similarity_score']
            
            st.session_state.processed_df_temp.loc[row, 'query'] = query_text
            st.session_state.processed_df_temp.loc[row, 'score'] = val
        
        st.session_state.processed_df_q = st.session_state.processed_df_temp



def highlight_values(df):
    # pd.set_option("styler.render.max_elements", 8455596)
    """여러 조건에 따른 하이라이트"""
    # 스타일러 생성
    styler = df.style
    
    # 높은 점수 (0.8 이상) 빨간색으로 하이라이트
    styler = styler.highlight_between(
        subset=['score'],
        left=0.8,
        right=1.0,
        props='background-color: rgba(255, 0, 0, 0.2)'
    )
    
    # 중간 점수 (0.5-0.8) 노란색으로 하이라이트
    styler = styler.highlight_between(
        subset=['score'],
        left=0.5,
        right=0.8,
        props='background-color: rgba(255, 255, 0, 0.2)'
    )
    
    return styler




def main():
    st.set_page_config(layout="wide")

    if 'OPENAI_API_KEY' in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    else:
        st.warning('OPENAI API KEY를 저장하세요.')
    
    st.title("Embedding Experiments for RAG")
    
    # 세션 상태 초기화
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'compression_retriever' not in st.session_state:
        st.session_state.compression_retriever = None
    if 'query_text' not in st.session_state:
        st.session_state.query_text = None
    if 'processed_df_temp' not in st.session_state:
        st.session_state.processed_df_temp = None
    if 'processed_df_q' not in st.session_state:
        st.session_state.processed_df_q = None
    
    # 파일 업로더
    uploaded_file = st.sidebar.file_uploader("데이터 파일을 업로드하세요 (Excel 또는 CSV)", 
                                   type=['csv', 'xlsx', 'xls'])
    
    # 탭 생성
    tab1, tab2 = st.tabs(["임베딩변환", "유사도측정"])
    
    try:
        # 탭1: 임베딩 결과(벡터) 데이터 추가
        with tab1:
            if uploaded_file is None:
                st.warning("파일을 업로드해주세요.")
            else:
                # 파일 확장자 확인
                file_extension = uploaded_file.name.split('.')[-1]
                # 파일명 추출
                fn = uploaded_file.name
                
                # 파일 읽기
                if file_extension == 'csv':
                    st.session_state.original_df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.original_df = pd.read_excel(uploaded_file)
                
                # 원본 데이터 미리보기
                st.subheader("원본 데이터 미리보기")
                st.dataframe(st.session_state.original_df.head())
                
                # 데이터 기본 정보 표시
                with st.expander("데이터 기본 정보"):
                    st.write(f"행 수: {st.session_state.original_df.shape[0]}")
                    st.write(f"열 수: {st.session_state.original_df.shape[1]}")
                    st.write("컬럼 목록:", ", ".join(st.session_state.original_df.columns.tolist()))
                
                
                # 전처리 시작 버튼
                if st.sidebar.button("데이터 전처리 시작"):
                    
                    all_documents = save_df_to_docs(st.session_state.original_df, fn)
                    
                    with st.spinner('벡터 변환 중'):
                        st.session_state.vectorstore, vector_result = save_docs_to_faiss(all_documents, embedding_function)
                        st.session_state.processed_df = array_to_df(st.session_state.original_df, vector_result)
                        
                    st.sidebar.success('SUCCESS: 임베딩 변환 완료')
                    
                    with st.expander("전처리된 데이터 미리보기"):
                        st.dataframe(st.session_state.processed_df.head())
                    
                        # # 다운로드 버튼 생성
                        # csv_buffer = BytesIO()
                        # st.session_state.processed_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        # csv_buffer.seek(0)
                        # st.download_button(
                        #     label="CSV 파일로 다운로드",
                        #     data=csv_buffer,
                        #     file_name=f"{fn}-processed-data.csv",
                        #     mime="text/csv"
                        # )
                else:
                    st.warning("Sidebar에서 데이터 전처리 시작 버튼을 클릭하세요.")
        
        # 탭2: 질의-유사도 측정
        with tab2:
            if st.session_state.processed_df is None:
                st.warning("먼저 탭1에서 데이터 전처리를 진행해주세요.")
            else:
                with st.expander("전처리된 데이터 미리보기"):
                    st.dataframe(st.session_state.processed_df.head())
                    
                # Retriever load
                retriever = st.session_state.vectorstore.as_retriever()
                
                embeddings_filter = EmbeddingsFilter(
                    embeddings=embedding_function,
                    # similarity_threshold=0 # 0.55 
                )
                # 압축 검색기 생성
                st.session_state.compression_retriever = ContextualCompressionRetriever(
                    # embeddings_filter 설정
                    base_compressor=embeddings_filter,
                    # retriever 를 호출하여 검색쿼리와 유사한 텍스트를 찾음
                    # base_retriever=vectorstore.as_retriever()
                    base_retriever=retriever
                )
                st.sidebar.success('LOADED: retriever')
                
                # 사용자 질의 입력
                query_text = st.text_input(
                    "유사도 계산을 위한 질의를 입력하세요.",
                    key="query_text",
                    on_change=process_query
                )
                
                # if query_text is None:
                #     st.warning("질의를 입력하세요.")
                # else:
                #     compression_result = compression_retriever.invoke(query_text)
                    
                #     # 유사도 계산결과 추가
                #     for idx, val in enumerate(compression_result):
                #         meta = compression_result[idx].metadata
                #         row = meta['row']
                #         val = compression_result[idx].state['query_similarity_score']

                #         # 질의 및 score column 추가가
                #         st.session_state.processed_df.insert(1,'query', None)
                #         st.session_state.processed_df.loc[row, 'query'] = query_text
                #         st.session_state.processed_df.insert(2,'score', None)
                #         st.session_state.processed_df.loc[row, 'score'] = val
                      
                #     st.session_state.processed_df_q = st.session_state.processed_df 
                # 출력
                if st.session_state.processed_df_q is not None:
                    # 스타일 적용
                    sorted_df = st.session_state.processed_df_q.sort_values(by='score', ascending=False)
                    # styled_df = highlight_values(sorted_df)
                    st.dataframe(sorted_df)
        

    except Exception as e:
        st.error(f"파일을 처리하는 동안 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()