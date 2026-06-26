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

def parse_jd_projects(jd_text):
    """
    Parses the JD text to extract individual projects based on the pattern 'Project X Title:'.
    If no projects are found, returns the entire text as a single generic project.
    """
    projects = []
    # Find all project titles including the 'Project X' part
    pattern = r'(Project \d+ Title:\s*.*)'
    titles = re.findall(pattern, jd_text)
    
    # If no specific projects are found, treat the whole JD as one project
    if not titles:
        return [{'title': 'General Role Description', 'text': jd_text}]
        
    # Split text by the project title pattern (accounting for possible newlines)
    parts = re.split(r'Project \d+ Title:\s*.*\n?', jd_text)
    
    for i, title in enumerate(titles):
        body = parts[i+1]
        # Clean up any trailing preamble from the next section (like 'Role: Internship')
        if 'Role: ' in body:
            body = body.split('Role: ')[0]
            
        projects.append({
            'title': title.strip(),
            'text': title.strip() + "\n" + body.strip()
        })
        
    return projects

def clean_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower()

def calculate_similarity(resume_text, jd_text):
    # Create TF-IDF Vectorizer with english stop words and disable IDF
    # Disabling IDF prevents the algorithm from mathematically penalizing words that appear in both documents
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    
    # Fit the vectorizer ONLY on the Job Description to create a vocabulary of "required keywords"
    try:
        vectorizer.fit([jd_text])
    except ValueError:
        # Happens if jd_text has no valid english words
        return 0.0, None
    
    # Transform both the Resume and the JD into this JD-specific vector space
    jd_vector = vectorizer.transform([jd_text])
    resume_vector = vectorizer.transform([resume_text])
    
    # Calculate raw Cosine Similarity
    raw_score = cosine_similarity(resume_vector, jd_vector)[0][0]
    
    # Scale the raw score. Raw cosine similarity of sparse text rarely exceeds 0.3-0.4 for a great match.
    # We multiply by 2.8 to map the realistic range [0, ~0.35] closer to [0, 1.0]. This prevents it from maxing out too easily.
    scaled_score = min(raw_score * 2.8, 1.0)
    
    return scaled_score, vectorizer

def main():
    st.set_page_config(page_title="ResumeIQ Analyzer", page_icon="📄", layout="wide")
    
    st.title("ResumeIQ: NLP-Powered Resume Analyzer")
    st.markdown("""
    Upload your resume (PDF) and a Job Description (PDF) to see how well they match.
    The analyzer will automatically detect multiple projects within the JD and score your relevance for each!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Resume")
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume")
        resume_text = ""
        if resume_file is not None:
            with st.spinner("Extracting text from Resume..."):
                resume_text = extract_text_from_pdf(resume_file)
            st.success("Resume uploaded successfully!")
                
    with col2:
        st.subheader("2. Upload Job Description")
        jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd")
        jd_text = ""
        if jd_file is not None:
            with st.spinner("Extracting text from JD..."):
                jd_text = extract_text_from_pdf(jd_file)
            st.success("JD uploaded successfully!")
        
    st.markdown("---")
    
    if st.button("Analyze Resume Relevance", type="primary"):
        if not resume_text:
            st.error("Please upload a resume.")
        elif not jd_text:
            st.error("Please upload a job description.")
        else:
            with st.spinner("Parsing projects and analyzing with TF-IDF..."):
                projects = parse_jd_projects(jd_text)
                clean_resume = clean_text(resume_text)
                
                st.subheader(f"Analysis Results ({len(projects)} Projects Found)")
                
                # Create a grid layout for project cards
                cols = st.columns(min(len(projects), 2))
                
                for idx, project in enumerate(projects):
                    col = cols[idx % 2]
                    
                    with col:
                        st.markdown(f"### 📌 {project['title']}")
                        clean_jd = clean_text(project['text'])
                        
                        score, vectorizer = calculate_similarity(clean_resume, clean_jd)
                        
                        if vectorizer is None:
                            st.error("Not enough text in this project to analyze.")
                            continue
                            
                        # Display Score
                        score_percentage = round(score * 100, 2)
                        st.metric(label="Relevance Score", value=f"{score_percentage}%")
                        
                        if score_percentage >= 75:
                            st.success("Excellent match!")
                        elif score_percentage >= 50:
                            st.warning("Good match, but consider adding keywords.")
                        else:
                            st.error("Low match.")
                            
                        # Show top keywords from JD
                        try:
                            feature_names = vectorizer.get_feature_names_out()
                            jd_vector = vectorizer.transform([clean_jd])
                            
                            # Get top keywords by TF-IDF score
                            sorted_items = np.argsort(jd_vector.toarray()[0])[::-1]
                            top_keywords = [feature_names[i] for i in sorted_items[:10] if jd_vector.toarray()[0][i] > 0]
                            
                            st.write("**Key Project Requirements:**")
                            st.write(", ".join(top_keywords))
                        except Exception as e:
                            pass
                        st.markdown("---")

if __name__ == "__main__":
    main()
