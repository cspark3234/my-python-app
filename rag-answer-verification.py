import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import tempfile

# .env 파일에서 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API 키를 .env 파일에 설정해 주세요.")
    st.stop()

@st.cache_resource
def load_and_process_docs(uploaded_file):
    """
    업로드된 PDF 파일을 임시로 저장하고, 텍스트를 분할한 뒤, 벡터 DB를 생성합니다.
    """
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # 임시 파일 삭제
        os.remove(tmp_path)
        
        return vector_store, docs
    return None, None

# Streamlit 앱 UI 설정
st.title("📚 RAG + 답변 평가 애플리케이션")
st.write("PDF 파일을 업로드하고, 답변의 신뢰도를 '상, 중, 하' 등급으로 확인하세요.")

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    # 문서 로딩 및 벡터 저장소 생성 (캐싱)
    vector_store, all_docs = load_and_process_docs(uploaded_file)

    if vector_store:
        # 1. RAG 체인 구성
        retriever = vector_store.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # 2. 답변 평가 프롬프트 및 LLM 정의
        verification_template = """
        제공된 '문서 내용'을 바탕으로 아래 '답변'의 사실성과 정확성을 평가하고 점수를 1점에서 5점 사이로 매겨주세요.
        1: 문서와 전혀 관련 없음.
        2: 문서에 일부 정보가 있지만 답변에 근거가 부족함.
        3: 답변의 절반 정도가 문서에 근거하며, 일부는 불분명함.
        4: 답변의 대부분이 문서에 정확히 근거함.
        5: 답변의 모든 내용이 문서에 완벽하고 정확하게 근거함.
        
        평가 점수만 숫자로 답하세요. (예: 5)
        
        ---
        문서 내용:
        {context}
        
        ---
        답변:
        {answer}
        """
        verification_prompt = PromptTemplate.from_template(verification_template)
        verification_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

        # 3. 사용자 질문 입력 및 처리
        question = st.text_input("질문을 입력하세요:", key="user_question")
        
        if question:
            with st.spinner("🔍 답변 및 평가 생성 중..."):
                # RAG를 통한 답변 생성
                rag_result = rag_chain.invoke({"query": question})
                answer = rag_result['result']
                source_documents = rag_result['source_documents']
                
                # 검색된 문서 내용을 하나의 문자열로 결합
                source_context = "\n\n".join([doc.page_content for doc in source_documents])

                # 답변 평가
                verification_result = verification_llm.invoke(verification_prompt.format(
                    context=source_context,
                    answer=answer
                ))
                
                # LLM의 답변에서 점수만 추출 (오류 처리 포함)
                try:
                    score = int(verification_result.content.strip())
                    if score >= 4:
                        grade = "상"
                        st.success(f"**답변:** {answer}")
                        st.info(f"✅ **평가 결과:** {grade} (정확하고 문서에 근거함)")
                    elif score == 3:
                        grade = "중"
                        st.warning(f"**답변:** {answer}")
                        st.info(f"🟡 **평가 결과:** {grade} (일부 근거가 부족할 수 있음)")
                    else:
                        grade = "하"
                        st.error(f"**답변:** {answer}")
                        st.info(f"❌ **평가 결과:** {grade} (문서와 관련성이 낮거나 환각 가능성)")
                except (ValueError, IndexError):
                    st.warning(f"**답변:** {answer}")
                    st.error("평가 결과를 판단할 수 없습니다.")
                
                st.markdown("---")
                st.markdown("### 📖 참고 문서")
                for i, doc in enumerate(source_documents):
                    st.markdown(f"**문서 {i+1}:** (페이지 {doc.metadata.get('page', 'N/A')})")
                    st.markdown(f"> {doc.page_content}")