import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit app title
st.title("Sentiment Analysis App")

# Choose analysis type: Single Text or CSV File
analysis_type = st.radio("Select Analysis Type:", ["Single Text", "CSV File"])

if analysis_type == "Single Text":
    # Input for single text
    single_text = st.text_area("Enter a single text for analysis:")

    if st.button("Analyze"):
        if single_text:
            result = sentiment_analyzer(single_text)
            st.write(f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']:.4f}")

elif analysis_type == "CSV File":
    # Input for CSV file
    csv_file = st.file_uploader("Upload a CSV file for batch analysis:", type=["csv"])

    if csv_file:
        # Read CSV file into DataFrame
        df = pd.read_csv(csv_file)

        # Check if the required column 'text' is present in the DataFrame
        if 'text' not in df.columns:
            st.error("CSV file must contain a column named 'text'")
        else:
            # Analyze sentiments for each text in the CSV
            sentiments = sentiment_analyzer(df["text"].tolist())

            # Add new columns for sentiment and confidence in the DataFrame
            df["sentiment"] = [res["label"] for res in sentiments]
            df["confidence"] = [res["score"] for res in sentiments]

            # Display the results
            st.dataframe(df)

            # EDA Section
            st.subheader("Exploratory Data Analysis (EDA)")

            # Display distribution of sentiments
            st.write("Sentiment Distribution:")
            sns.countplot(x="sentiment", data=df)
            st.pyplot()

            # Display confidence distribution
            st.write("Confidence Distribution:")
            plt.hist(df["confidence"], bins=20, color='skyblue', edgecolor='black')
            st.pyplot()

# Note: This is a basic EDA section. You can customize it based on your specific analysis needs.
