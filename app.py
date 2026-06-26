import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def clean_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower()

def calculate_similarity(resume_text, jd_text):
    # Create TF-IDF Vectorizer with english stop words and disable IDF
    # Disabling IDF prevents the algorithm from mathematically penalizing words that appear in both documents
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    
    # Fit the vectorizer ONLY on the Job Description to create a vocabulary of "required keywords"
    vectorizer.fit([jd_text])
    
    # Transform both the Resume and the JD into this JD-specific vector space
    jd_vector = vectorizer.transform([jd_text])
    resume_vector = vectorizer.transform([resume_text])
    
    # Calculate raw Cosine Similarity
    raw_score = cosine_similarity(resume_vector, jd_vector)[0][0]
    
    # Scale the raw score. Raw cosine similarity of sparse text rarely exceeds 0.3-0.4 for a great match.
    # We multiply by 3.5 to map the realistic range [0, ~0.28] closer to [0, 1.0].
    scaled_score = min(raw_score * 3.5, 1.0)
    
    return scaled_score, vectorizer

def main():
    st.set_page_config(page_title="ResumeIQ Analyzer", page_icon="📄", layout="wide")
    
    st.title("ResumeIQ: NLP-Powered Resume Analyzer")
    st.markdown("""
    Upload your resume (PDF) and paste a Job Description (JD) to see how well they match.
    This tool uses **TF-IDF vectorization** and **cosine similarity** to calculate a keyword match score.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Resume")
        uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
        resume_text = ""
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            st.success("Resume uploaded and text extracted!")
            with st.expander("View Extracted Resume Text"):
                st.text(resume_text)
                
    with col2:
        st.subheader("2. Job Description")
        jd_text = st.text_area("Paste the Job Description here", height=250)
        
    st.markdown("---")
    
    if st.button("Analyze Resume", type="primary"):
        if not resume_text:
            st.error("Please upload a resume.")
        elif not jd_text:
            st.error("Please paste a job description.")
        else:
            with st.spinner("Analyzing with TF-IDF..."):
                clean_resume = clean_text(resume_text)
                clean_jd = clean_text(jd_text)
                
                score, vectorizer = calculate_similarity(clean_resume, clean_jd)
                
                st.subheader("Analysis Results")
                
                # Display Score
                score_percentage = round(score * 100, 2)
                st.metric(label="Resume Match Score", value=f"{score_percentage}%")
                
                if score_percentage >= 75:
                    st.success("Excellent match! Your resume strongly aligns with this JD.")
                elif score_percentage >= 50:
                    st.warning("Good match, but consider adding more keywords from the JD to your resume.")
                else:
                    st.error("Low match. You may need to significantly tailor your resume for this role.")
                    
                # Show top keywords from JD
                try:
                    feature_names = vectorizer.get_feature_names_out()
                    jd_vector = vectorizer.transform([clean_jd])
                    
                    # Get top keywords by TF-IDF score
                    sorted_items = np.argsort(jd_vector.toarray()[0])[::-1]
                    top_keywords = [feature_names[i] for i in sorted_items[:15] if jd_vector.toarray()[0][i] > 0]
                    
                    st.write("**Top Keywords in Job Description (by TF-IDF):**")
                    st.write(", ".join(top_keywords))
                    
                    st.info("Tip: Ensure your resume includes these keywords if you have the relevant experience.")
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
