# My ChatGroq Project 🤖

## What is this?
This is my project that builds a smart chatbot using AI! It can answer questions about machine learning by reading HuggingFace documentation.

## What does it do?
- Loads a website (HuggingFace docs) automatically
- Breaks the content into small pieces
- When you ask a question, it finds the most relevant pieces
- Uses AI (Groq) to give you a smart answer
- Shows you which parts of the docs it used

## Stuff I need to install:
```bash
pip install streamlit langchain langchain-community langchain-groq faiss-cpu python-dotenv
```

## Setup:
1. Get a free Groq API key from groq.com
2. Make a `.env` file and put: `GROQ_API_KEY=your_key_here`
3. Make sure Ollama is running on your computer

## How to run:
```bash
streamlit run app.py
```
Then go to http://localhost:8501

## How I built this:
1. **WebBaseLoader** - grabs content from websites
2. **Text Splitter** - chops up long docs into chunks
3. **Ollama Embeddings** - turns text into numbers the computer understands
4. **FAISS** - super fast search through all the chunks
5. **Groq** - the AI brain that writes the answers
6. **Streamlit** - makes it look nice on the web

## Cool features:
- Only loads the docs once (saves time!)
- Shows how long each answer takes
- You can see exactly which parts of the docs were used
- Clean web interface

---
*This project demonstrates key concepts in AI, NLP, and web development!*
