import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
from typing import Optional, List
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from sklearn.decomposition import PCA
import plotly.express as px

NAMES_EMBEDDING = ["ko-sroberta", "BAAI/bge-m3", "multilingual-e5", "openai (text-embedding-3-large)"]

# OpenAI API 키 관리 함수는 동일하게 유지
def get_openai_api_key() -> Optional[str]:
    if 'OPENAI_API_KEY' in st.secrets:
        return st.secrets['OPENAI_API_KEY']
    
    if 'openai_api_key' in st.session_state:
        return st.session_state.openai_api_key
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Get one at https://platform.openai.com/account/api-keys"
    )
    
    if api_key:
        st.session_state.openai_api_key = api_key
        return api_key
    
    return None

# 임베딩 모델 초기화 함수 추가
def initialize_embeddings(embedding_type: str):
    if embedding_type == "openai (text-embedding-3-large)":
        return OpenAIEmbeddings(model="text-embedding-3-large")
    elif embedding_type == "ko-sroberta":
        return HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif embedding_type == "BAAI/bge-m3":
            return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif embedding_type == "multilingual-e5":
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

# 텍스트 입력을 Document로 변환하는 함수 추가
def text_to_documents(text: str) -> List[Document]:
    lines = text.strip().split('\n')
    return [Document(page_content=line.strip(), metadata={'row': idx}) 
            for idx, line in enumerate(lines) if line.strip()]

# 데이터프레임을 Document로 변환하는 함수 수정
def save_df_to_docs(df, fn):
    if fn.endswith('.csv'):  # CSV 파일 업로드의 경우
        fn_csv = f'{fn}'
        df.to_csv(fn_csv, index=False)
        loader = CSVLoader(fn_csv, encoding='utf-8-sig')
        all_documents = loader.load_and_split()
    else:  # 텍스트 입력의 경우
        all_documents = [Document(page_content=str(row.to_dict()), metadata={'row': idx}) 
                        for idx, row in df.iterrows()]
    
    st.sidebar.write(f"{len(all_documents)} ROWS")
    return all_documents

# 문서->벡터스토어어
def save_docs_to_faiss(all_documents, embedding_function):
    vectorstore = FAISS.from_documents(documents=all_documents, embedding=embedding_function)
    vector_result = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
    
    st.sidebar.write(f"VECTORSTORE LOADED")
    st.sidebar.write(f"ㄴ{len(vector_result)} Documents, {len(vector_result[0])} Dimensions")

    return vectorstore, vector_result

def array_to_df(df, vector_result):
    array_data = np.array(vector_result)
    column_names = [f'vec_{i:04d}' for i in range(len(array_data[0]))]
    df_vector = pd.DataFrame(array_data, columns=column_names)    
    return pd.concat([df, df_vector], axis=1)

# 콜백 함수 정의 - 유사도 계산, 계산결과(cosine, L2, dot_product) 추가
def process_query():
    st.session_state.processed_df_q = None
    st.session_state.processed_df_temp = st.session_state.processed_df
    cols_only_vec = [x for x in list(st.session_state.processed_df_temp.columns) if 'vec_' in x]
    
    query_text = st.session_state.query_text
    compression_retriever = st.session_state.compression_retriever
    
    if query_text:
        n_rows = st.session_state.processed_df_temp.shape[0]
        # st.write(f"n rows = {n_rows}")
        # 유사도측정-L2
        res_l2 = st.session_state.vectorstore.similarity_search_with_score(query_text, k=n_rows)
        # st.write(res_l2)
        # 유사도측정-cosine
        compression_result = compression_retriever.invoke(query_text)
        
        
        # 사용자 질의의 임베딩 구하기
        q_document = [Document(page_content=query_text)]
        vectorstore_current = FAISS.from_documents(documents=q_document, embedding=st.session_state.embedding_func)
        vector_result_q = vectorstore_current.index.reconstruct_n(0, vectorstore_current.index.ntotal)
        # st.markdown(f"질의 **{query_text}**에 대한 벡터 변환 결과: {vector_result_q}")
        
        # 유사도측정-dot product
        vals_temp = st.session_state.processed_df_temp.loc[:, cols_only_vec].values
        res_dot_products = [np.dot(vector_result_q, x) for x in vals_temp]
        # st.write(res_dot_products)
        
        
        # query와 score 컬럼이 이미 있는지 확인
        if 'query' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(1, 'query', None)
        if 'score(cosine)' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(2, 'score(cosine)', None)
        if 'score(L2)' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(3, 'score(L2)', None)
        if 'dot_product' not in st.session_state.processed_df_temp.columns:
            st.session_state.processed_df_temp.insert(4, 'dot_product', None) 
            
        # 유사도 계산결과 추가
        for idx, val in enumerate(compression_result):
            meta = compression_result[idx].metadata
            row = meta['row']
            val = compression_result[idx].state['query_similarity_score']
            val_l2 = res_l2[idx][1]
            st.session_state.processed_df_temp.loc[row, 'query'] = query_text
            st.session_state.processed_df_temp.loc[row, 'score(cosine)'] = val
            st.session_state.processed_df_temp.loc[row, 'score(L2)'] = val_l2
            st.session_state.processed_df_temp.loc[row, 'dot_product'] = res_dot_products[idx]
        
        st.session_state.processed_df_q = st.session_state.processed_df_temp


def main():
    st.set_page_config(layout="wide")
    
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    else:
        st.warning('OPENAI API KEY를 저장하세요.')
    
    st.title("Embedding Experiments for RAG")
    
    # 세션 상태 초기화 (기존과 동일)
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'embedding_func' not in st.session_state:
        st.session_state.embedding_func = None
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
    
    # 데이터 입력 방식 선택
    use_file_upload = st.sidebar.toggle("파일 업로드 사용", value=False, 
                                      help="파일 업로드와 텍스트 입력 중 선택하세요")    
    if use_file_upload:
        uploaded_file = st.sidebar.file_uploader(
            "데이터 파일을 업로드하세요 (Excel 또는 CSV)", 
            type=['csv', 'xlsx', 'xls']
        )
    else:
        sample_text = '''
            꽁꽁 언 한강 위로 고양이가 돌아다닙니다.
            한강 작가가 작년에 노벨문학상을 받았어요.
            한강에서 먹는 라면이 제일 맛있죠.
            한강 소설 추천해주세요.
            한강 공원으로 자전거 타러 가자.
        '''
        text_input = st.sidebar.text_area(
            "데이터를 입력하세요 (각 줄을 새로운 항목으로 처리)",
            value=sample_text.replace('            ',''),
            height=200
        )
        
    # 임베딩 모델 선택
    embedding_type = st.sidebar.selectbox(
        "임베딩 모델 선택",
        NAMES_EMBEDDING,
        help="Choose the embedding model to use"
    )
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["임베딩변환", "유사도측정", "시각화"])
    
    try:
        # 탭1: 임베딩변환
        with tab1:
            if use_file_upload is True and uploaded_file is None:
                st.warning("파일을 업로드해주세요.")
            elif use_file_upload is False and not text_input:
                st.warning("텍스트를 입력해주세요.")
            else:
                if use_file_upload is True:
                    # 기존 파일 처리 로직
                    file_extension = uploaded_file.name.split('.')[-1]
                    fn = uploaded_file.name
                    if file_extension == 'csv':
                        st.session_state.original_df = pd.read_csv(uploaded_file)
                    else:
                        st.session_state.original_df = pd.read_excel(uploaded_file)
                else:
                    # 텍스트 입력 처리
                    lines = text_input.strip().split('\n')
                    st.session_state.original_df = pd.DataFrame({
                        'content': [line.strip() for line in lines if line.strip()]
                    })
                    fn = "text_input"
                
                # 원본 데이터 미리보기 및 처리 로직은 동일하게 유지
                st.subheader("원본 데이터 미리보기")
                st.dataframe(st.session_state.original_df.head())
                
                btn_start_embedding = st.sidebar.button("데이터 전처리 시작")
                
                if btn_start_embedding:
                    # 임베딩 모델 초기화
                    embedding_function = initialize_embeddings(embedding_type)
                    st.session_state.embedding_func = embedding_function
                    
                    all_documents = save_df_to_docs(st.session_state.original_df, fn)
                    
                    with st.spinner('벡터 변환 중'):
                        st.session_state.vectorstore, vector_result = save_docs_to_faiss(
                            all_documents, 
                            embedding_function
                        )
                        st.session_state.processed_df = array_to_df(
                            st.session_state.original_df, 
                            vector_result
                        )
                    
                    st.sidebar.success('SUCCESS: 임베딩 변환 완료')
                    
                    with st.expander("전처리된 데이터 미리보기"):
                        st.dataframe(st.session_state.processed_df.head())
                        
                else:
                    st.info('좌측 사이드바에서 데이터 전처리 시작 버튼을 클릭하세요.')
                
        # 탭2: 유사도측정
        with tab2:
            if st.session_state.processed_df is None:
                st.warning("먼저 탭1에서 데이터 전처리를 진행해주세요.")
            else:
                st.subheader(f"예상질의에 대한 유사도 측정하기")
                st.markdown(f" - embedding model: `{embedding_type}`")
                # 기존 탭2 로직 유지
                with st.expander("전처리된 데이터 미리보기"):
                    st.dataframe(st.session_state.processed_df.head())
                    
                # Retriever load
                row_num = st.session_state.processed_df.shape[0]
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": row_num})
                
                embeddings_filter = EmbeddingsFilter(
                    embeddings=st.session_state.embedding_func,
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
                    "유사도 계산을 위한 질의를 입력 후 엔터를 누르세요. (e.g. 한강이 어디에요?)",
                    key="query_text",
                    on_change=process_query
                )
                
                # 사용자 질의의 임베딩 구하기
                q_document = [Document(page_content=query_text)]
                vectorstore_current = FAISS.from_documents(documents=q_document, embedding=st.session_state.embedding_func)
                vector_result_q = vectorstore_current.index.reconstruct_n(0, vectorstore_current.index.ntotal)
                with st.expander(f"질의 **{query_text}**에 대한 벡터 변환 결과"):
                    st.write(f"{vector_result_q}")
                
                # 출력
                if st.session_state.processed_df_q is not None:
                    sorted_df = st.session_state.processed_df_q.sort_values(by='score(cosine)', ascending=False)
                    st.dataframe(sorted_df)

        # 탭3: 시각화
        with tab3:
            st.markdown(f" - embedding model: `{embedding_type}`")
            btn_pca = st.button('차원축소 및 시각화')
            if btn_pca and use_file_upload:
                st.warning('파일 업로드 시각화는 준비중입니다.')
            elif btn_pca and use_file_upload is False:
                with st.spinner('진행중'):
                    list_cols_embed = list(st.session_state.processed_df.columns)
                    list_cols_embed_only = [x for x in list_cols_embed if 'vec_' in x]
                    
                    embed_vals = st.session_state.processed_df.loc[:, list_cols_embed_only].values
                
                    
                    # PCA 모델 생성 및 적용
                    pca = PCA(n_components=3)
                    X_pca = pca.fit_transform(embed_vals)
                    
                    # 병합
                    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
                    process_df_pca = pd.concat([df_pca, st.session_state.processed_df], axis=1)
                    
                    with st.expander("차원 축소(PCA) 결과 확인하기"):
                        st.dataframe(process_df_pca)
                                    
                    # 3D 산점도 생성
                    toggle_vis_value = st.toggle("차트에 값 나타내기", value=True)
                    vis_title = f'3D PCA Visualization - {embedding_type}'
                    if toggle_vis_value:
                        fig = px.scatter_3d(process_df_pca
                                            ,x='PC1', y='PC2', z='PC3'
                                            ,title=vis_title
                                            ,text='content'
                                            ,symbol='content')
                    else:
                        fig = px.scatter_3d(process_df_pca
                                            ,x='PC1', y='PC2', z='PC3'
                                            ,title=vis_title
                                            ,symbol='content')
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.warning('위 버튼을 클릭하세요.')


    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
        print(e)

if __name__ == "__main__":
    main()