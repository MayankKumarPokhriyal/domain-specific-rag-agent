import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Domain RAG Agent", layout="centered")
st.title("📄 Domain-Specific RAG Agent")

# ---- PDF Upload ----
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Ingest Documents"):
    if uploaded_files:
        for file in uploaded_files:
            files = {"file": (file.name, file, "application/pdf")}
            requests.post(f"{API_URL}/ingest", files=files)
        st.success("Documents ingested successfully")
    else:
        st.warning("Please upload at least one PDF")

# ---- Query Section ----
st.header("Ask a Question")
query = st.text_input("Enter your question")

if st.button("Submit Query"):
    if query:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query}
        )
        st.subheader("Answer")
        st.write(response.json().get("answer", "No response"))
    else:
        st.warning("Please enter a question")