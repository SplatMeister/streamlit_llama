import streamlit as st
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import notebook_login
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import PromptTemplate

st.set_page_config(page_title="Financial Analysis Chatbot", page_icon="💼")

# CSS styles
st.markdown("""
    <style>
    /* Your CSS styles */
    </style>
""", unsafe_allow_html=True)

# Document class
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Read PDF function
def read_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Initialize session state variables
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for Hugging Face login and PDF upload
with st.sidebar:
    st.subheader("Hugging Face Login")
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    submit_button = st.button("Login")

    if submit_button:
        try:
            notebook_login(hf_token)
            st.success("Connected successfully to Hugging Face Hub.")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.subheader("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
    process_button = st.button("Process PDFs")

# Main interface
st.header("Financial Analysis Chatbot 💼")

# Process and setup chain
if process_button and uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        text = read_pdf(io.BytesIO(bytes_data))
        documents.append(Document(text))

    st.session_state.documents_processed = True

    combined_text = " ".join([doc.page_content for doc in documents])
    DEVICE = "cpu"
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_text = text_splitter.split_documents(documents)

    db = Chroma.from_documents(split_text, embeddings, persist_directory="db")

    # Using the specified model
    model = "meta-llama/Llama-2-7b-chat-hf"
    try:
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=hf_token, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=hf_token)
        else:
            st.error("Please enter a valid Hugging Face token.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline, model_kwargs={"temperature": 0})
    prompt = PromptTemplate(template=generate_prompt, input_variables=["context", "question"])
    st.session_state.chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# Chat interface
if st.session_state.documents_processed:
    user_query = st.text_input("Ask a question about your documents:", key="user_query")
    submit_query = st.button("Submit Query")

    if submit_query and user_query:
        financial_analyst_query = f"As a financial analyst, {user_query}"
        result = st.session_state.chain({'question': financial_analyst_query, 'chat_history': st.session_state.chat_history})
        st.session_state.chat_history.append((financial_analyst_query, result['answer']))

        st.subheader("Chat History")
        for query, response in reversed(st.session_state.chat_history):
            st.text(f"You: {query}")
            st.text(f"Financial Analyst: {response}\n")
else:
    st.write("Please upload and process PDFs to enable the chat feature.")