# ResumeIQ Architecture

## High-Level Architecture
ResumeIQ is built as a monolithic, single-page web application using **Streamlit** for both the frontend and backend logic. The application processes data entirely in-memory during the user's session without relying on an external database.

## Technology Stack
- **Frontend & UI**: Streamlit (Python)
- **Text Extraction**: `pdfplumber`
- **Text Preprocessing**: Python `re` (Regular Expressions)
- **NLP & Machine Learning**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- **Numerical Operations**: `numpy`

## System Flow & Pipeline

1. **Input Layer**: 
   - Receives PDF file from the user via Streamlit's `file_uploader`.
   - Receives raw Job Description text via Streamlit's `text_area`.

2. **Extraction Layer**:
   - `extract_text_from_pdf(pdf_file)`: Iterates through the pages of the uploaded PDF using `pdfplumber` and aggregates the text into a single string.

3. **Preprocessing Layer**:
   - `clean_text(text)`: Applies regular expressions to remove non-alphanumeric characters and converts all text to lowercase to ensure uniformity before vectorization.

4. **NLP Processing Pipeline**:
   - **Vectorization**: Initializes a `TfidfVectorizer` (with English stop words removed) to transform both the cleaned resume text and the cleaned JD text into TF-IDF numerical vectors.
   - **Scoring**: Calculates the `cosine_similarity` between the two resulting vectors to determine the angle/closeness of the documents.
   - **Feature Extraction**: Extracts the top keywords from the JD vector by sorting the TF-IDF scores in descending order.

5. **Output/Presentation Layer**:
   - Displays the calculated similarity score as a percentage.
   - Renders feedback messages based on the score threshold (e.g., >75% Excellent, >50% Good, <50% Low).
   - Lists the top keywords identified in the JD for user improvement.

## Deployment Strategy
The application is designed to be easily deployable on **Streamlit Community Cloud**. It relies on `requirements.txt` to define its environment dependencies and `app.py` as its entry point.
