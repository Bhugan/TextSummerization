import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

# Create a summarization pipeline using a pre-trained model
summarizer = pipeline("summarization")

@st.cache_resource
def text_summary(text, maxlength=None):
    result = summarizer(text, max_length=150)  # Adjust max_length as needed
    return result[0]['summary_text']

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

# Title and choice
st.title("Text and Document Analyzer")

# User choice
choice = st.radio("Select your choice", ["Analyze Single Text", "Analyze Document"])

if choice == "Analyze Single Text":
    st.subheader("Analyze Single Text using Hugging Face Transformers")
    input_text = st.text_area("Enter your text here")
    if input_text is not None and st.button("Analyze"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Your Input Text**")
            st.info(input_text)
        with col2:
            st.markdown("**Analysis Result**")
            result = text_summary(input_text)
            st.success(result)

elif choice == "Analyze Document":
    st.subheader("Analyze Document using Hugging Face Transformers")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None and st.button("Analyze"):
        with open("doc_file.pdf", "wb") as f:
            f.write(input_file.getbuffer())
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("File uploaded successfully")
            extracted_text = extract_text_from_pdf("doc_file.pdf")
            st.markdown("**Extracted Text is Below:**")
            st.info(extracted_text)
        with col2:
            st.markdown("**Analysis Result**")
            doc_summary = text_summary(extracted_text)
            st.success(doc_summary)
