import streamlit as st
import pickle
import html
import re
import string
import pandas as pd
import glob
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import streamlit.components.v1 as components

# Load saved model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("sentiment_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Stopwords for cleaning
stop_words = set(stopwords.words('english'))

# Define text cleaning function
def clean_review(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\s*(<br\s*/?>)+\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered).strip()

# Session state setup
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False
if "just_ran" not in st.session_state:
    st.session_state.just_ran = False
elif st.session_state["just_ran"]:
    st.session_state.just_ran = False
    st.rerun()

# --- UI Section ---
st.title("🎬 IMDB Review Sentiment Predictor")

# Sidebar toggle
show_history = st.sidebar.checkbox("📁 Show Review History", value=False)
if show_history:
    review_files = sorted(glob.glob("analyzed_reviews/review_*.txt"), reverse=True)[:5]
    st.sidebar.markdown("### 📝 Recent Reviews")
    if review_files:
        for file in review_files:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            st.sidebar.text(content)
    else:
        st.sidebar.info("No reviews found yet.")

# Input box
if st.session_state.clear_input:
    user_input = st.text_area("Paste a movie review below:", value="", key="input_box")
    st.session_state.clear_input = False
else:
    user_input = st.text_area("Paste a movie review below:", value=st.session_state["user_input"], key="input_box")
    st.session_state["user_input"] = user_input

# Analyze button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_review(user_input)
        vect_input = vectorizer.transform([cleaned])
        prediction = model.predict(vect_input)[0]
        label = "Positive 😊" if prediction == 1 else "Negative 😞"

        st.subheader("Prediction:")
        st.success(label)

        # Optional: Play result aloud
        if st.checkbox("🔊 Read Prediction Aloud"):
            speak_script = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance();
                    msg.text = "The sentiment is {label}";
                    window.speechSynthesis.speak(msg);
                </script>
            """
            components.html(speak_script, height=0, width=0)

        if prediction == 1:
            st.balloons()

        # Save the review and prediction
        os.makedirs("analyzed_reviews", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyzed_reviews/review_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Review:\n{user_input.strip()}\n\nPrediction: {label}\n")

        # Clear input and rerun
        st.session_state.clear_input = True
        st.session_state.just_ran = True
        st.stop()

# --- Post-prediction Display Section ---

# Show latest review
files = sorted(glob.glob("analyzed_reviews/review_*.txt"), reverse=True)
if files:
    with open(files[0], "r", encoding="utf-8") as f:
        last_review = f.read()
    st.markdown("---")
    st.markdown("### 📁 Last Logged Review")
    st.text(last_review)

# Summary analytics
pos_count, neg_count = 0, 0
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        if "Positive" in content:
            pos_count += 1
        elif "Negative" in content:
            neg_count += 1

total = pos_count + neg_count
if total > 0:
    st.markdown("### 📊 Sentiment Summary")
    st.write(f"✅ Positive: {pos_count}")
    st.write(f"❌ Negative: {neg_count}")
    st.progress(pos_count / total)

# Show full review log and enable CSV download
data = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        review = lines[1].strip() if len(lines) > 1 else ""
        prediction = "Positive" if "Positive" in lines[-1] else "Negative"
        data.append({"Review": review, "Prediction": prediction})

df = pd.DataFrame(data)
if not df.empty:
    st.markdown("### 📄 Full Review Log")
    st.dataframe(df)
    st.download_button(
        label="Download All Reviews as CSV",
        data=df.to_csv(index=False),
        file_name="review_log.csv",
        mime="text/csv"
    )
