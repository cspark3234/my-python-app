import streamlit as st
import os
import tempfile
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Streamlit secrets에서 API 키를 가져옵니다.
openai_api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_and_process_docs(uploaded_file):
    """
    업로드된 문서를 로드하고 처리하여 벡터 저장소를 생성합니다.
    """
    # 임시 파일에 업로드된 파일 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 파일 확장자에 따라 로더 선택
    ext = os.path.splitext(uploaded_file.name)[1]
    if ext == '.pdf':
        loader = PyPDFLoader(tmp_path)
    elif ext == '.txt':
        loader = TextLoader(tmp_path)
    else:
        st.error("지원하지 않는 파일 형식입니다. .pdf 또는 .txt 파일을 업로드하세요.")
        return None, None

    all_docs = loader.load()

    # 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)

    # 임베딩 모델과 벡터 저장소 생성
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    os.remove(tmp_path)  # 임시 파일 삭제
    return vector_store, all_docs

# Streamlit 앱 UI 구성
st.title("RAG 기반 답변 검증 시스템")
st.markdown("자료를 업로드하면, 해당 자료를 기반으로 질문에 대한 답변을 생성하고 원본 출처를 제공합니다.")

uploaded_file = st.file_uploader("자료(PDF 또는 TXT)를 업로드하세요.", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("자료를 처리 중입니다... 잠시만 기다려주세요."):
        vector_store, all_docs = load_and_process_docs(uploaded_file)
    st.success("자료 처리가 완료되었습니다!")

    # 사용자 질문 입력
    user_query = st.text_input("자료에 대해 궁금한 점을 질문하세요:")

    if user_query and vector_store:
        with st.spinner("답변을 생성 중입니다..."):
            # 관련 문서 검색
            retrieved_docs = vector_store.similarity_search(user_query, k=3)
            context = " ".join([doc.page_content for doc in retrieved_docs])

            # LLM 체인 생성
            llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                당신은 유용한 챗봇입니다. 주어진 맥락(context)만을 활용하여 질문에 답하세요.
                만약 맥락에 정보가 없다면, "주어진 자료에서 답변을 찾을 수 없습니다."라고 답하세요.
                답변을 할 때 참고한 원본 자료의 페이지 번호를 반드시 포함해야 합니다.

                맥락:
                {context}

                질문:
                {question}

                답변:
                """
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            
            # RAG를 이용한 답변 생성
            response = llm_chain.run(context=context, question=user_query)

            st.subheader("답변")
            st.write(response)
            
            # 원본 출처 표시
            st.subheader("참고 출처")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', '알 수 없는 출처')
                page = doc.metadata.get('page', '알 수 없는 페이지')
                st.write(f"**출처 {i+1}:** {os.path.basename(source)}, 페이지: {page}")
