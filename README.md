

# 🧠 Stroke Risk Analysis: Deep Learning Edition
### *Bridging Clinical Expertise and Artificial Intelligence*

(https://stroke-risk-analysis-nn-neygflfkrg7qqdfumtevcb.streamlit.app/)

## 📋 Project Overview
As a **Medical Laboratory Science** graduate transitioning into **AI & Data Science**, I developed this application to transform clinical data into actionable insights. Unlike standard binary classifiers, this tool uses a **Neural Network (Multi-Layer Perceptron)** to calculate stroke probability and provides **Clinical Reasoning** for its predictions.

## 🔬 Clinical Logic & AI Integration
The model analyzes 11 patient features, including hypertension, heart disease, and metabolic markers. 

### Key Features:
* **Deep Learning Engine:** Built using **TensorFlow/Keras**, trained on the Stroke Prediction Dataset to identify non-linear relationships between risk factors.
* **Explainable AI (XAI):** Features a "Patient Feature Intensity" chart that visualizes how the model weights specific patient markers (e.g., how much Age contributed vs. Smoking Status).
* **Dynamic Clinical Summary:** Generates a human-readable explanation for the risk assessment, mirroring the diagnostic process of a clinician.

## 🛠️ Technical Stack
* **Backend:** Python 3.11+, TensorFlow (Neural Networks), Scikit-Learn (Data Scaling)
* **Frontend:** Streamlit
* **Deployment:** GitHub & Streamlit Cloud
* **Data Processing:** Pandas, NumPy, Joblib (Model Serialization)



## 🚀 Installation & Local Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/classicwebdev/stroke-risk-nn.git
   cd stroke-risk-nn
   ```
2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## 📈 Model Performance
The model was evaluated using a test split to ensure sensitivity to high-risk markers while maintaining a low false-positive rate, crucial for medical triage tools.

---
**Author:** Mustapha  
*Medical Laboratory Science | AI & Data Science Aspirant*

