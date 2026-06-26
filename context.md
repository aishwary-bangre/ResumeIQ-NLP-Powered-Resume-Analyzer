# ResumeIQ Context

## Project Overview
ResumeIQ is a Natural Language Processing (NLP) powered resume analyzer designed to compare a candidate's resume against a specific Job Description (JD). It provides an objective similarity score to help users understand how well their resume aligns with the target role.

## Background & Motivation
The project was built to address the challenge of keyword matching in modern Applicant Tracking Systems (ATS). By simulating how an ATS evaluates a resume based on a job description, candidates can better tailor their resumes to increase their chances of passing automated screenings.

## Key Features
- **PDF Resume Upload**: Users can seamlessly upload their resumes in PDF format.
- **Job Description Input**: Users can paste any job description into a text area.
- **Automated Text Extraction**: Extracts raw text from complex PDF layouts using `pdfplumber`.
- **Intelligent Keyword Matching**: Uses TF-IDF vectorization to identify important keywords in the JD and scores the resume against them using cosine similarity.
- **Top Keyword Recommendations**: Highlights the most significant keywords from the JD that the user should consider including in their resume.

## Value Proposition
- Rapidly prototype and test resume variations.
- Get a quantitative score on resume-JD alignment.
- Identify missing keywords before applying to a job.
