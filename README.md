# 📚 StudyMate Query Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studymate-classifier.streamlit.app)

An NLP & Machine Learning web application built with **Streamlit** and **Scikit-Learn** that classifies student prompts into **Education** or **Non-Education** categories in real time.

## 🌐 Live Demo
🔗 **[Open StudyMate Query Classifier](https://studymate-classifier.streamlit.app)** *(replace with your exact deployed Streamlit or Render URL)*

## 🚀 Features
- **1,000-Sample Balanced Dataset:** Trained on 500 academic prompts and 500 everyday non-academic prompts.
- **NLP Feature Extraction:** Uses `TfidfVectorizer` (unigrams & bigrams) paired with `LogisticRegression` for fast, lightweight inference without external API keys.
- **Interactive Streamlit UI:** Includes a live dashboard, one-click sample test prompts, class probability breakdown tables, and chat prediction history.

## 🧠 ML Pipeline
```text
Student Prompt ─▶ TF-IDF Vectorizer ─▶ Logistic Regression ─▶ Education / Non-Education + Confidence %
