import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- UI Setup ---
st.set_page_config(page_title="Bulk Resume Ranker", layout="wide")
st.title("📊 Bulk Candidate Screening System")
st.markdown("Upload a dataset of resumes to rank candidates against a job role.")

# --- Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

# --- Layout ---
jd_input = st.text_area("Step 1: Paste Job Description Here", height=150)
uploaded_csv = st.file_uploader("Step 2: Upload Resume CSV Dataset", type="csv")

if uploaded_csv and jd_input:
    # Read the dataset
    df = pd.read_csv(uploaded_csv)
    
    st.write("### Dataset Preview")
    st.dataframe(df.head(3))
    
    # Let the user pick which column contains the resume text
    column_names = df.columns.tolist()
    text_col = st.selectbox("Select the column containing Resume Text:", column_names)
    
    if st.button("Rank All Candidates"):
        with st.spinner("Processing large dataset..."):
            # 1. Clean data
            clean_jd = clean_text(jd_input)
            df['cleaned_resume'] = df[text_col].apply(clean_text)
            
            # 2. Vectorization & Scoring
            tfidf = TfidfVectorizer(stop_words='english')
            # Combine JD with all resumes for the TF-IDF matrix
            all_texts = [clean_jd] + df['cleaned_resume'].tolist()
            tfidf_matrix = tfidf.fit_transform(all_texts)
            
            # Compare JD (index 0) against all resumes (index 1 to end)
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # 3. Add scores to DataFrame
            df['Match_Score %'] = (scores[0] * 100).round(2)
            
            # 4. Sort and Filter
            results_df = df.sort_values(by='Match_Score %', ascending=False)
            
            st.divider()
            st.success(f"Ranked {len(df)} candidates successfully!")
            
            # Display Top Candidates
            st.subheader("Top Ranked Candidates")
            st.dataframe(results_df.drop(columns=['cleaned_resume']).head(10))
            
            # 5. Download Ranked CSV
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Ranked Results CSV",
                data=csv_data,
                file_name="ranked_candidates.csv",
                mime="text/csv"
            )
else:
    st.info("Please paste a Job Description and upload your CSV file to begin.")