import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ---------------- UI ----------------
st.set_page_config(page_title="AI PDF Chat Assistant", layout="wide")
st.title("💬 AI PDF Chat Assistant (FREE & SMART)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# ---------------- SESSION ----------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PROCESS PDF ----------------
if uploaded_file is not None and st.session_state.qa_chain is None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ File uploaded!")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    st.write(f"📄 Pages: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    st.write(f"🧩 Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(chunks, embeddings)
    
    llm = Ollama(model="phi")

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    st.session_state.qa_chain = qa_chain

    st.success("✅ Ready to chat!")

# ---------------- ASK QUESTION ONLY AFTER READY ----------------
if st.session_state.qa_chain:

    query = st.text_input("💬 Ask your question")

    if query:
        result = st.session_state.qa_chain.run(query)

        st.session_state.chat_history.append((query, result))

# ---------------- CHAT HISTORY ----------------
if st.session_state.chat_history:
    st.subheader("📁 Chat History")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"🧑 **You:** {q}")
        st.markdown(f"🤖 **Answer:** {a}")
        st.markdown("---")
