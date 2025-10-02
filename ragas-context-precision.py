import streamlit as st
import os
import json
import re 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------- 환경 변수 --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.error(f"🚨 .env 파일에 OPENAI_API_KEY를 설정해주세요. (경로: {dotenv_path})")
    st.stop()

RAG_MODEL_NAME = "gpt-3.5-turbo"
EVAL_MODEL_NAME = "gpt-4o-mini"

# -------------------- LLM 초기화 --------------------
@st.cache_resource(show_spinner=False)
def initialize_llms(api_key):
    rag_llm = ChatOpenAI(model=RAG_MODEL_NAME, temperature=0.1, api_key=api_key)
    eval_llm = ChatOpenAI(model=EVAL_MODEL_NAME, temperature=0.0, api_key=api_key, request_timeout=60)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return rag_llm, eval_llm, embeddings

rag_llm, eval_llm, embeddings = initialize_llms(OPENAI_API_KEY)

# -------------------- JSON 안전 파싱 --------------------
def safe_json_parse(raw_content: str):
    try:
        return json.loads(raw_content) 
    except json.JSONDecodeError:
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL) or re.search(r'(\{.*\})', raw_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 파싱 실패: {str(e)[:50]}...")
        else:
            raise ValueError(f"JSON 객체를 찾을 수 없음: {raw_content[:100]}...")

# -------------------- 문서 처리 --------------------
def load_and_process_docs(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file_path = f"./temp_doc_{os.getpid()}_{os.path.basename(uploaded_file.name)}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension in [".txt", ".md"]:
        loader = TextLoader(temp_file_path, encoding='utf-8')
    else:
        os.remove(temp_file_path)
        return None, "지원하지 않는 파일 형식입니다. (PDF, TXT, MD만 지원)"

    try:
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        os.remove(temp_file_path)
        return None, f"문서 처리 중 오류 발생: {e}"

    os.remove(temp_file_path)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, f"문서 로드 및 인덱싱 완료! 총 {len(chunks)}개 청크."

# -------------------- RAG 답변 생성 --------------------
def retrieve_and_generate(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    retrieved_docs = retriever.invoke(query)
    
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    source_info = [{"content": doc.page_content, "source": doc.metadata.get('source', '업로드 문서'), "page": doc.metadata.get('page', 'N/A')} for doc in retrieved_docs]

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 제공된 '맥락'을 기반으로 질문에 답변하는 전문가입니다. 답변은 맥락의 내용을 정확히 인용해야 합니다. "
                   "만약 맥락에 답변할 충분한 정보가 없다면 '문서에 관련 정보가 충분하지 않습니다.'라고 답변하세요."), 
        ("user", "맥락:\n{context}\n\n질문: {question}")
    ])
    
    chain = rag_prompt | rag_llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": query})
    
    return response, context, source_info, retrieved_docs

# -------------------- Precision 평가 --------------------
class PrecisionEval(BaseModel):
    relevant: bool
    rationale: str

def evaluate_precision(query: str, retrieved_docs: list):
    evaluation_results = []
    total_chunks = len(retrieved_docs)
    relevant_chunks = 0

    prompt_template = PromptTemplate(
        input_variables=["query", "chunk"],
        template=(
            "질문: {query}\n"
            "청크 내용: {chunk}\n"
            "위 청크가 질문에 관련이 있는지 평가하고 JSON 형식으로 출력하세요.\n"
            "예시: {{\"relevant\": true, \"rationale\": \"관련 근거 내용\"}}"
        )
    )

    for i, doc in enumerate(retrieved_docs):
        try:
            input_text = prompt_template.format(query=query, chunk=doc.page_content)
            structured_llm = eval_llm.with_structured_output(PrecisionEval)
            eval_result: PrecisionEval = structured_llm.invoke(input_text)

            relevant = getattr(eval_result, "relevant", False)
            rationale = getattr(eval_result, "rationale", "이유 없음")
            if relevant:
                relevant_chunks += 1

            evaluation_results.append({
                "chunk_index": i + 1,
                "content": doc.page_content,
                "relevant": relevant,
                "score": 1.0 if relevant else 0.0,
                "rationale": rationale
            })
        except Exception as e:
            evaluation_results.append({
                "chunk_index": i + 1,
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "relevant": False,
                "score": 0.0,
                "rationale": f"LLM 평가 오류: {e}"
            })

    precision_score = relevant_chunks / total_chunks if total_chunks > 0 else 0.0
    return {
        "score": precision_score,
        "total_chunks": total_chunks,
        "relevant_chunks": relevant_chunks,
        "detailed_results": evaluation_results
    }

# -------------------- Faithfulness 평가 --------------------
class FactItem(BaseModel):
    fact: str
    supported: bool
    fact_rationale: str

class FaithfulnessEval(BaseModel):
    overall_rationale: str
    facts: List[FactItem]

def evaluate_faithfulness(answer: str, context: str):
    try:
        structured_llm = eval_llm.with_structured_output(FaithfulnessEval)
        prompt = f"맥락: {context}\n답변: {answer}\n위 답변의 사실 여부를 평가하고 JSON 형식으로 출력하세요."
        eval_result: FaithfulnessEval = structured_llm.invoke(prompt)
        facts = eval_result.facts
        total_facts = len(facts)
        supported_facts = sum(1 for f in facts if f.supported)
        score = supported_facts / total_facts if total_facts > 0 else 0.0
        return {
            "score": score,
            "overall_rationale": eval_result.overall_rationale,
            "verified_facts": [f.dict() for f in facts],
        }
    except Exception as e:
        return {"score": 0.0, "overall_rationale": f"평가 오류: {e}", "verified_facts": []}

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RAGAS 심층 분석 앱", layout="wide")
st.title("✨ RAGAS 검증 심층 분석")
st.markdown(f"### `{RAG_MODEL_NAME}`로 답변 생성, `{EVAL_MODEL_NAME}`로 평가 진행")
st.markdown("---")

# 파일 업로드
with st.sidebar:
    st.header("1. 문서 업로드")
    uploaded_file = st.file_uploader("PDF, TXT, MD 파일 업로드", type=["pdf","txt","md"])
    if uploaded_file:
        if "vectorstore" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with st.spinner("문서 처리 및 인덱싱 중..."):
                vectorstore, status_message = load_and_process_docs(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.status = status_message
                    st.session_state.uploaded_file_name = uploaded_file.name
                else:
                    st.error(status_message)
                    st.session_state.status = status_message
    if "status" in st.session_state:
        st.info(st.session_state.status)

# 질문 입력 및 실행
if "vectorstore" in st.session_state:
    st.header("2. 질문하기")
    question = st.text_input("질문 입력:", placeholder="세탁기 수평 맞추는 방법")
    if st.button("답변 생성 및 평가 실행"):
        if question:
            st.session_state.answer = None
            st.session_state.precision_evaluation = None
            st.session_state.faithfulness_evaluation = None
            with st.spinner("RAGAS 심층 검증 중..."):
                answer, context, source_info, retrieved_docs = retrieve_and_generate(question, st.session_state.vectorstore)
                precision_result = evaluate_precision(question, retrieved_docs)
                faithfulness_result = evaluate_faithfulness(answer, context)

                st.session_state.answer = answer
                st.session_state.context = context
                st.session_state.source_info = source_info
                st.session_state.precision_evaluation = precision_result
                st.session_state.faithfulness_evaluation = faithfulness_result
                st.success("✅ 심층 검증 완료!")

# 결과 표시
if st.session_state.get("answer"):
    st.markdown("---")
    precision_eval = st.session_state.precision_evaluation
    st.subheader("💡 리트리벌 평가지표 (Context Precision)")
    col_score, col_summary = st.columns([1,2])
    with col_score:
        st.metric("맥락 정확도 점수", f"{precision_eval['score']:.2f}")
    with col_summary:
        st.markdown(f"총 {precision_eval['total_chunks']}개 청크 중 {precision_eval['relevant_chunks']}개 관련 있음")

    with st.expander("청크별 상세 평가"):
        precision_data = []
        for item in precision_eval['detailed_results']:
            precision_data.append({
                "청크 번호": item['chunk_index'],
                "관련성": "✅ 관련 있음" if item['relevant'] else "❌ 관련 없음",
                "심사관 근거": item['rationale'],
                "청크 내용 (일부)": item['content'][:150]+"..." if len(item['content'])>150 else item['content']
            })
        st.dataframe(precision_data, hide_index=True, use_container_width=True)

    st.markdown("---")
    faithfulness_eval = st.session_state.faithfulness_evaluation
    answer = st.session_state.answer
    total_facts = len(faithfulness_eval['verified_facts'])
    supported_facts = sum(1 for f in faithfulness_eval['verified_facts'] if f.get('supported', False))
    st.subheader("💡 생성된 답변 (Faithfulness)")
    st.success(answer)
    col_score, col_rationale = st.columns([1,2])
    with col_score:
        st.metric(f"({supported_facts}/{total_facts}) Supported", f"{faithfulness_eval['score']:.2f}")
    with col_rationale:
        st.info(faithfulness_eval['overall_rationale'])