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
            response = requests.post(
                f"{API_URL}/ingest/upload",
                files=files
            )
            if response.status_code != 200:
                st.error(response.text)
                break
        else:
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
        if response.status_code == 200:
            st.subheader("Answer")
            st.write(response.json().get("answer", "No response"))
        else:
            st.error(response.text)
    else:
        st.warning("Please enter a question")