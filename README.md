# 📰 News Sentiment Analysis using NLP 🚀

This project is a **News Sentiment Analysis Web App** that fetches the latest news articles about a company, analyzes sentiment, and generates an audio summary in Hindi. It uses **FastAPI for the backend** and **Streamlit for the frontend**.

## **✨ Features**
- ✅ Fetches **real-time news** about a company  
- ✅ **Scrapes news articles** (title, summary, full content)  
- ✅ **Performs sentiment analysis** (Positive, Neutral, Negative)  
- ✅ **Extracts key topics** using NLP  
- ✅ **Generates Hindi audio summary** using Google Text-to-Speech (gTTS)  
- ✅ **FastAPI Backend & Streamlit Frontend**  

---

## **🛠️ Technologies Used**
- **FastAPI** (Backend API)  
- **Streamlit** (Frontend UI)  
- **Newspaper3k** (Scraping news content)  
- **BeautifulSoup** (Web scraping)  
- **NLTK & spaCy** (Text preprocessing & sentiment analysis)  
- **YAKE & TF-IDF** (Keyword extraction)  
- **Google Text-to-Speech (gTTS)** (Hindi audio generation)  
- **Hugging Face Spaces** (Deployment)  

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
# 📰 News Sentiment Analysis using NLP 🚀

This project is a **News Sentiment Analysis Web App** that fetches the latest news articles about a company, analyzes sentiment, and generates an audio summary in Hindi. It uses **FastAPI for the backend** and **Streamlit for the frontend**.

---

## **🛠️ Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/SKC9372/News-Analysis-using-NLP.git
cd News-Analysis-using-NLP
```

### **2️⃣ Create a Virtual Environment & Install Dependencies**
#### **🔹 Using Conda**
```bash
conda create -n news_analysis python=3.10 -y
conda activate news_analysis
pip install -r requirements.txt
```

#### **🔹 Using Virtualenv**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

---

## **🛠️ Running the Application**
### **1️⃣ Start the FastAPI Backend**
Run the FastAPI server (it will run on `http://127.0.0.1:8000`)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### **2️⃣ Start the Streamlit Frontend**
Run the Streamlit app (it will run on `http://localhost:8501`)
```bash
streamlit run app.py
```

---

## **🌍 Deployment**
The project is **deployed on Hugging Face Spaces**.  
To deploy it yourself:
1. Push your repository to GitHub  
2. Add a `Dockerfile`  
3. Connect your GitHub repo to Hugging Face Spaces  

---

## **📜 License**
This project is licensed under the **MIT License**.

---

## **🤝 Contributing**
Contributions are welcome!  
If you have ideas or improvements, feel free to create a **Pull Request**.

---

## **📩 Contact**
For any questions, contact me at:  
📧 **suryakantchaubey2001@gmail.com**  

---

