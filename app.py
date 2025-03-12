import streamlit as st
import pdfplumber
import docx
import spacy
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to process text with NLP
def process_text(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return set(skills)

# Function to compare resume with job description
def calculate_similarity(resume_text, job_desc):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_desc])
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix[0, 1] * 100  # Convert to percentage

# Streamlit UI
st.title("📄 AI-Powered Resume Screener")
st.sidebar.header("Upload Resume")
uploaded_resume = st.sidebar.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description", "")

if uploaded_resume and job_description:
    # Extract text from uploaded file
    file_extension = uploaded_resume.name.split(".")[-1]
    if file_extension == "pdf":
        resume_text = extract_text_from_pdf(uploaded_resume)
    elif file_extension == "docx":
        resume_text = extract_text_from_docx(uploaded_resume)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Process text
    resume_skills = process_text(resume_text)
    similarity = calculate_similarity(resume_text, job_description)

    # Display results
    st.subheader("🔍 Resume Analysis")
    st.write(f"**Extracted Skills:** {', '.join(resume_skills)}")
    st.write(f"**Match Score:** {similarity:.2f}%")

    if similarity > 70:
        st.success("✅ Strong Match! This resume fits well with the job description.")
    elif similarity > 40:
        st.warning("⚠️ Moderate Match! Some skills match, but improvements are needed.")
    else:
        st.error("❌ Weak Match! This resume needs significant improvement.")

st.sidebar.info("💡 Upload a resume and paste a job description to get started!")
