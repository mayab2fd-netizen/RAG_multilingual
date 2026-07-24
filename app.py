import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 페이지 설정
st.set_page_config(page_title="AI 글로벌 계약서 분석기", page_icon="⚖️", layout="wide")
st.title("⚖️ AI 글로벌 계약서 & 법률 문서 분석기")
st.markdown("영문/다국어 계약서를 업로드하고 질문해보세요. AI가 조항 분석, 위약금, 계약 기간 등 핵심 내용을 찾아 번역해 줍니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PDF 텍스트 추출 및 벡터 DB 생성 함수
def process_pdf(uploaded_file, api_key):
    # Streamlit UploadedFile을 PyMuPDF가 읽을 수 있도록 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. PDF 텍스트 추출
        try:
            pdf_doc = fitz.open(tmp_path)
            text = ""
            for page in pdf_doc:
                text += page.get_text() + "\n\n"
            pdf_doc.close()
        except Exception:
            st.error("파일을 읽을 수 없거나 암호가 걸려 있습니다. 정상적인 PDF 파일인지 확인해주세요.")
            return None

        # 텍스트 추출 결과가 비어있는 경우 (이미지 스캔본 PDF 등)
        if not text.strip():
            st.error("PDF에서 텍스트를 추출할 수 없습니다. 이미지 스캔본이 아닌 텍스트 기반 PDF를 업로드해주세요.")
            return None

        # 2. 텍스트 청크 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)

        # 3. FAISS 벡터스토어 생성
        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        return vectorstore
    finally:
        # 임시 파일 삭제
        os.remove(tmp_path)

# ----------------- 사이드바 (설정 영역) -----------------
with st.sidebar:
    st.header("⚙️ 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    target_language = st.selectbox(
        "출력 언어 선택 (Target Language)", 
        ["한국어", "English", "日本語", "中文", "Español", "Français"]
    )

    uploaded_file = st.file_uploader("PDF 문서를 업로드하세요", type="pdf")

    if st.button("문서 분석하기"):
        if not openai_api_key:
            st.error("OpenAI API Key를 입력해주세요.")
        elif not uploaded_file:
            st.error("PDF 파일을 업로드해주세요.")
        else:
            with st.spinner("문서를 분석하고 벡터 DB를 구축하는 중입니다..."):
                result_vectorstore = process_pdf(uploaded_file, openai_api_key)
                if result_vectorstore is not None:
                    st.session_state.vectorstore = result_vectorstore
                    st.success("문서 처리가 완료되었습니다! 우측에서 질문해주세요.")

# ----------------- 메인 화면 (채팅 영역) -----------------
# 이전 채팅 기록 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 질문 입력란
if query := st.chat_input("계약서 내용에 대해 질문해주세요. (예: 지각 시 위약금 조항이 어떻게 돼? 계약 해지 조건은 뭐야?)"):
    # 사용자 질문 화면에 표시
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if st.session_state.vectorstore is None:
        st.warning("👈 먼저 좌측 사이드바에서 PDF 파일을 업로드하고 '문서 분석하기' 버튼을 눌러주세요.")
    elif not openai_api_key:
        st.warning("🔑 질문을 입력하기 전, 좌측 사이드바에 OpenAI API Key를 입력해주세요.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("문서에서 정답을 찾아 번역하는 중입니다..."):
                try:
                    # 1. 유사도 검색 (k=5로 설정하여 컨텍스트 길이 최적화)
                    vectorstore = st.session_state.vectorstore
                    docs = vectorstore.similarity_search_with_score(query, k=5)

                    context = ""
                    for doc, score in docs:
                        context += doc.page_content + "\n\n"

                    # 2. LLM 및 프롬프트 체인 구성
                    # 답변을 사용자가 선택한 언어로 강제하는 프롬프트 사용
                    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)

                    template = """당신은 글로벌 계약서 및 법률 문서를 전문적으로 분석하는 AI 법무 검토관입니다.
                    다음 제공된 계약서/법률 문서의 내용을 바탕으로 사용자 질문에 정확하고 객관적으로 답변해주세요.
                    만약 문서에 질문에 대한 답이 명확히 없다면, 추측하지 말고 '문서에서 해당 조항을 찾을 수 없습니다'라고 답변하세요.

                    중요: 답변은 반드시 **{language}**로 작성하고 번역해야 합니다. 중요한 법률/계약 용어는 원문을 괄호 안에 병기하고 이해하기 쉽게 설명해주세요.

                    [계약서/문서 내용]
                    {context}

                    [사용자 질문]
                    {question}"""

                    prompt = ChatPromptTemplate.from_template(template)
                    chain = prompt | llm | StrOutputParser()

                    # 3. 답변 생성 및 세션 저장
                    response = chain.invoke({
                        "context": context,
                        "question": query,
                        "language": target_language
                    })

                    st.write(response)
                    # 어시스턴트 답변 세션에 저장 (response가 정상적으로 생성된 경우에만 저장)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
