# ResumeIQ: NLP-Powered Resume Analyzer

This project is a direct implementation of the "ResumeIQ" project mentioned on your resume. It's a Streamlit application that uses Natural Language Processing (NLP) to compare a resume to a job description.

## How It Works

1. **Upload & Extract**: The user uploads their resume as a PDF file. The app uses the `pdfplumber` library to read the text from the PDF file.
2. **Pre-processing**: The extracted resume text and the provided Job Description (JD) text are cleaned (converted to lowercase, special characters removed).
3. **TF-IDF Vectorization**: We use `scikit-learn`'s `TfidfVectorizer` to convert the text into numerical vectors. TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates how relevant a word is to a document in a collection of documents. This means it gives higher weight to keywords that are frequent in the JD but not just common English words (like "the", "and").
4. **Cosine Similarity**: We then calculate the cosine similarity between the resume vector and the JD vector. Cosine similarity measures the angle between two vectors, resulting in a score between 0 (completely different) and 1 (identical). We convert this to a percentage score.
5. **UI**: The `streamlit` library provides a simple, clean, and reactive web interface.

## How this matches your resume bullet points:

> **"Built an NLP pipeline using TF-IDF vectorization and cosine similarity to score resume-JD keyword match..."**
*   **Implementation:** Look at the `calculate_similarity` function in `app.py`. It uses `TfidfVectorizer` and `cosine_similarity` from `sklearn` to calculate the match score.

> **"...implemented PDF text extraction via pdfplumber across varied document formats."**
*   **Implementation:** Look at the `extract_text_from_pdf` function in `app.py`. It explicitly uses the `pdfplumber` library to iterate through pages and extract text.

> **"Deployed on Streamlit Cloud with real-time analysis interface..."**
*   **Implementation:** The entire app is built with Streamlit (`app.py`), which provides the "real-time analysis interface". To fulfill the "Deployed on Streamlit Cloud" part, you would simply push this code to a public GitHub repository and connect it to Streamlit Community Cloud (which is free and takes about 2 minutes).

## How to run the project locally

1. Open your terminal in this directory (`d:\cursor projects\NLP Resume`).
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. A new browser window should open automatically with your app running at `http://localhost:8501`.

## Interview Tips

If you are asked about this project in an interview:
- Be ready to explain what **TF-IDF** is conceptually: It's a statistical measure used to evaluate how important a word is to a document in a collection or corpus.
- Be ready to explain what **Cosine Similarity** is: It measures the similarity between two non-zero vectors of an inner product space. It measures the cosine of the angle between them.
- Mention that you chose `pdfplumber` because it's generally more robust for extracting text from complex PDFs compared to older libraries like `PyPDF2`.
- Mention that `Streamlit` allowed you to rapidly prototype the UI without needing to write complex frontend code (HTML/CSS/JS) or backend APIs (Flask/FastAPI).
