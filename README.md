# 🎬 IMDB Sentiment Predictor

This interactive web app analyzes movie reviews and predicts their sentiment—**Positive 😊** or **Negative 😞**—using a trained machine learning model and natural language processing. Built with [Streamlit](https://streamlit.io/), it includes real-time predictions, data visualization, and optional voice playback.

---

## 🔍 Features

- ✅ Clean NLP preprocessing with BeautifulSoup, regex, and stopword removal
- 🎓 Machine Learning using a scikit-learn classifier and TF-IDF vectorizer
- 📊 Summary stats with visual feedback (positive/negative proportions)
- 💬 Text-to-speech playback of predictions (browser-based voice)
- 💾 Save and display recent reviews and predictions
- 📁 Export all analyzed reviews to CSV

---

## 🚀 Try it Online

🖥️ Deploy your own version or use this repo with [Streamlit Cloud](https://streamlit.io/cloud).  

Just make sure your main app script is correctly set in the app configuration (e.g., `sentiment_webapp.py` or `app.py`).

---

## ⚙️ Requirements

Install dependencies with:
streamlit
scikit-learn
pandas
beautifulsoup4
nltk
pickle-mixin
html5lib
numpy
requests
streamlit==1.33.0
scikit-learn==1.4.1.post1
pandas==2.2.2
nltk==3.8.1
beautifulsoup4==4.12.3
dir


```bash
pip install -r requirements.txt
# imdb-sentiment-app
