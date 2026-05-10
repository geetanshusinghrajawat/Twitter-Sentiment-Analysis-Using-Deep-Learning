# 🐦 Twitter Sentiment Analysis

A deep learning model that classifies tweets as **Positive**, **Neutral**, or **Negative** — built with an RNN + Embedding architecture and deployed as a real-time Streamlit web app.

## 🚀 Live Demo
[**Try it here →**](https://twitter-sentiment-analysis-using-deep-learning-by-geetanshu.streamlit.app/)

## 📌 Overview

Enter any tweet and the app instantly predicts its sentiment. The model was trained on a large Twitter dataset and uses a Sequential RNN with an Embedding layer to understand the contextual meaning of words.

## 🛠️ How It Works

1. **Data** — Twitter training dataset (`twitter_training.csv`); filtered to Positive, Negative, and Neutral labels only (irrelevant tweets removed)
2. **Cleaning** — Null removal, deduplication, column renaming, label encoding (Negative=0, Neutral=1, Positive=2)
3. **Tokenization** — Keras Tokenizer fitted on the full tweet corpus; sequences converted and padded to uniform length (166 tokens)
4. **Model Iterations**:
   - `model` — SimpleRNN (32 units) baseline
   - `model_2` — Embedding + Dense (experimental)
   - `model_final` — **Embedding (vocab × 128) → SimpleRNN (32) → Dense (3, Softmax)** ← deployed
5. **Training** — 10 epochs, Adam optimizer, sparse categorical crossentropy, 80/20 train-test split, batch size 64
6. **Deployment** — Streamlit app for real-time tweet sentiment prediction

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | Embedding + SimpleRNN + Dense |
| Preprocessing | Keras Tokenizer, pad_sequences, LabelEncoder |
| Serialization | Pickle |
| Deployment | Streamlit |

## 📁 Project Structure

```
├── project.ipynb      # Data preprocessing, model training & evaluation
├── main.py            # Streamlit web app
├── model.h5           # Saved final Keras model
└── tokenizer.pkl      # Serialized tokenizer
```

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/geetanshusinghrajawat/your-repo-name
cd your-repo-name

# Install dependencies
pip install streamlit tensorflow scikit-learn numpy pandas

# Run the app
streamlit run main.py
```

## 📊 Dataset

Twitter sentiment dataset containing tweets labeled across Positive, Negative, Neutral, and Irrelevant categories across multiple gaming and tech topics.

## ⚠️ Known Issue
The training notebook uses `padding='pre'` with `maxlen=166`. Ensure `main.py` matches these settings for accurate predictions.

## 👤 Author

**Geetanshu Singh Rajawat**  
[LinkedIn](https://www.linkedin.com/in/geetanshu-singh-rajawat/) | [GitHub](https://github.com/geetanshusinghrajawat)
