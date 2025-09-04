# 🖼️ Handwritten Digit Generation Web App

A simple **Streamlit web application** powered by a **Conditional Generative Adversarial Network (cGAN)** trained on the MNIST dataset.  
This app generates **handwritten digits (0–9)** based on user input and displays multiple unique variations.

---

## 📌 Features
- Generate handwritten digits (0–9) interactively.
- Produces **5 unique images** per selected digit.
- Lightweight Streamlit interface for quick testing.
- Built with **PyTorch** and **Streamlit**.
- Works fully on **CPU** (no GPU required for inference).

---

## ⚙️ Requirements
Python 3.8+ is recommended.  

Install all dependencies with:
```bash
pip install -r requirements.txt
▶️ Usage

Clone the repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Download/Prepare the trained Generator model

Ensure you have the trained generator model file:

Default expected filename: generator_epoch_50.pth

Place it in the same directory as app.py.

Run the Streamlit app

streamlit run app.py


Open in your browser
Streamlit will show a local URL, e.g. http://localhost:8501.
Open it to interact with the app.
