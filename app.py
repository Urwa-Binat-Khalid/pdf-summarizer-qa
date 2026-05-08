import streamlit as st
import tempfile
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="PDF Summarizer and Q&A", page_icon="")

st.title("PDF Document Summarizer with Q&A")
st.write("Upload a PDF file. Get a summary. Then ask questions about the document.")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "processed" not in st.session_state:
    st.session_state.processed = False

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_groq_client():
    api_key = st.secrets["GROQ_API_KEY"]
    return Groq(api_key=api_key)

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    os.unlink(tmp_path)
    return text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def create_vector_store(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_relevant_chunks(query, index, chunks, embeddings, embedding_model, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def get_summary_from_llm(client, full_text):
    context = full_text[:8000]
    
    prompt = f"""Summarize the following document. Include the main topics and key points. Be concise.

Document:
{context}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a document analysis assistant. Provide accurate summaries based only on the given text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

def answer_question(client, question, context_chunks):
    context = "\n\n".join(context_chunks)
    
    if len(context) > 6000:
        context = context[:6000]
    
    prompt = f"""Answer the question based ONLY on the provided document context. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You answer questions based strictly on the provided document text. Do not make up information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    return response.choices[0].message.content

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if not st.session_state.processed:
        with st.spinner("Processing PDF..."):
            full_text = extract_text_from_pdf(uploaded_file)
            
            if len(full_text) < 100:
                st.error("Could not extract text from this PDF. The file may be scanned or image-based.")
                st.stop()
            
            chunks = split_text_into_chunks(full_text)
            
            embedding_model = load_embedding_model()
            index, embeddings = create_vector_store(chunks, embedding_model)
            
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.session_state.full_text = full_text
            st.session_state.processed = True
            st.session_state.embedding_model = embedding_model
            
            groq_client = load_groq_client()
            summary = get_summary_from_llm(groq_client, full_text)
            st.session_state.summary = summary
        
        st.success("PDF processed successfully")
    
    st.subheader("Summary")
    st.write(st.session_state.summary)
    
    st.divider()
    
    st.subheader("Ask Questions About This Document")
    
    user_question = st.text_input("Enter your question:", placeholder="Example: What is the termination clause?")
    
    if user_question:
        with st.spinner("Finding answer..."):
            groq_client = load_groq_client()
            relevant_chunks = retrieve_relevant_chunks(
                user_question, 
                st.session_state.index, 
                st.session_state.chunks, 
                st.session_state.embeddings, 
                st.session_state.embedding_model
            )
            answer = answer_question(groq_client, user_question, relevant_chunks)
        
        st.subheader("Answer")
        st.write(answer)
        
        with st.expander("Show relevant document sections"):
            for i, chunk in enumerate(relevant_chunks):
                st.text(f"Section {i+1}:")
                st.write(chunk)
                st.divider()

else:
    st.info("Please upload a PDF file to begin.")
    