# NLP Algorithm Design

This document explains the mathematical and architectural decisions behind the NLP scoring pipeline in ResumeIQ.

## Core Problem in Resume-JD Matching

A standard TF-IDF (Term Frequency - Inverse Document Frequency) and Cosine Similarity approach on just two documents (one Resume, one Job Description) often yields artificially low match scores (e.g., < 10%) even for highly qualified candidates. This happens due to two main reasons:

1. **The IDF Penalty on Shared Words**: Standard TF-IDF penalizes words that appear frequently across documents. When comparing only 2 documents, any keyword that successfully appears in *both* the JD and the Resume gets a lower mathematical weight than unique noise words that appear in only one document. This essentially punishes the algorithm for finding a match.
2. **Noise and Dimensionality**: Job Descriptions often contain hundreds of unique words (boilerplate text, equal opportunity clauses), and Resumes contain hundreds of unique words (hobbies, past unrelated jobs). In a raw Cosine Similarity calculation, these unique words act as massive noise, pushing the documents far apart in vector space and diluting the importance of the actual matched keywords.

## Optimized Solution

To ensure realistic, human-readable scores that accurately reflect candidate alignment, ResumeIQ implements a customized NLP pipeline:

### 1. Disabling Inverse Document Frequency (IDF)
We utilize `scikit-learn`'s `TfidfVectorizer` but explicitly disable the IDF component (`use_idf=False`). This ensures that the algorithm strictly measures Term Frequency and L2 normalization, completely removing the mathematical penalty for shared keywords. 

### 2. Vocabulary Restriction (JD as Source of Truth)
Instead of comparing all the noise in the Resume against all the noise in the JD, we restrict the "universe of words" strictly to the words found in the JD:
- The vectorizer is explicitly fitted *only* on the Job Description text.
- The Resume text is then transformed into that specific JD vector space.
This design guarantees that the algorithm completely ignores irrelevant resume fluff (e.g., unrelated past jobs) and only rewards the presence of required JD keywords.

### 3. Score Scaling
Because raw cosine similarity between documents of drastically different lengths naturally yields very low decimals, we apply a mathematical scaling multiplier to map the raw dot product into a realistic 0-100% human-readable percentage. This ensures users get intuitive feedback on their matching score.
