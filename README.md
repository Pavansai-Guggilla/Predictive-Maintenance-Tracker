# Predictive-Maintenance-Tracker

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-4285F4?style=flat&logo=googlecolab&logoColor=white) 
![GitHub](https://img.shields.io/badge/license-MIT-blue) ![Streamlit](https://img.shields.io/badge/Deployed_on-Streamlit-FF4B4B)  

This application predicts machine failures using sensor and operational data. Built with a **Random Forest Classifier** (98.25% accuracy), it enables proactive maintenance to reduce downtime.  

**[🚀 Live Demo](https://predictive-maintenance-using-machine-learning.streamlit.app/)** | **[📚 Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)** | 

---

## Table of Contents  
- [Key Features](#key-features)  
- [Dataset Overview](#dataset-overview)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Model Training & Results](#model-training--results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Key Features  
- **Failure Prediction**: Classify failures into 5 types (Heat Dissipation, Power, Overstrain, Tool Wear, Random).  
- **Web Interface**: Streamlit-based UI for real-time predictions.  
- **Model Comparison**: Tested 4 models (Random Forest, LSTM, Transformer, MLP).  
- **Scalability**: Designed for easy cloud deployment.  

---

## Dataset Overview  
The synthetic dataset contains **10,000 entries** with 10 features.  

### Feature Description  
| Feature | Type | Description | Notes |  
|---------|------|-------------|-------|  
| UID | `int` | Unique identifier (1-10000) | Not used in training |  
| Product ID | `object` | Product variant (L/M/H) + serial | Encoded as categorical |  
| Air Temp [K] | `float` | Ambient temperature | Normalized around 300K |  
| Process Temp [K] | `float` | Operational temperature | Air Temp + 10K ± noise |  
| Rotational Speed [rpm] | `float` | Motor speed | Derived from power (2860W) |  
| Torque [Nm] | `float` | Rotational force | Normal distribution (μ=40) |  
| Tool Wear [min] | `int` | Wear duration | Quality-dependent increments |  
| Failure Type | `object` | Failure category | Target variable |  

**⚠️ Warning**: Avoid using `Machine Failure` as a feature to prevent data leakage.  

---

## Installation  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/Pavansai-Guggilla/Predictive-Maintenance-Tracker.git  
   cd Predictive-Maintenance-Tracker  
   ```  

2. **Set up a virtual environment** (recommended):  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # Linux/Mac  
   venv\Scripts\activate     # Windows  
   ```  

3. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  # Create this file with listed packages  
   ```  

---

## Usage  
1. **Run the Streamlit app**:  
   ```bash  
   streamlit run app.py  
   ```  
2. **Input parameters** via the web form:  
   - Product type (L/M/H)  
   - Sensor data (temperature, speed, torque, tool wear)  
3. **View predictions**: The app returns failure probabilities and classifications.  

![Application Interface](Capture.PNG)  

---

## Exploratory Data Analysis  

### 1. Class Distribution  
- **Severe Class Imbalance**: 96.7% of entries are "No Failure."  
 
  ![Figure_1](https://github.com/user-attachments/assets/e80ee2f6-363f-414b-9ad3-6e0ff88a5b64)

### 2. Feature Distributions  
- **Rotational Speed vs. Torque**: Strong negative correlation (-0.88).  
- **Tool Wear**: Bimodal distribution (peaks at 50min and 200min).  
  ![Figure_2](https://github.com/user-attachments/assets/e41359c2-2b85-44c0-9169-1efd3aab1719)
  

### 3. Correlation Heatmap  
- Air and Process Temperatures are highly correlated (0.88).  
- Rotational Speed and Torque show inverse relationships.  
  ![Figure_4](https://github.com/user-attachments/assets/56a1f411-1ed9-40f0-871e-25700a3fa7e9)
 

---

## Model Training & Results  

### Model Comparison  
| Model | Accuracy | Precision | Recall |  
|-------|----------|-----------|--------|  
| Random Forest | **98.25%** | 0.97 | 1.00 |  
| LSTM | 96.75% | 0.16 | 0.17 |  
| Transformer | 96.75% | 0.16 | 0.17 |  
| MLP | 96.75% | 0.16 | 0.17 |  

**Key Insight**: Random Forest outperforms neural networks due to class imbalance and tabular data structure.  

### Classification Report  
- **High False Negatives**: Minority classes (e.g., Heat Dissipation) show 0% recall.  
- **Recommendation**: Use SMOTE or class weights in future iterations.  

Performance Summary

![Figure_5](https://github.com/user-attachments/assets/77fe35ee-1270-4648-8de4-24b479fe09c0)
![Figure_6](https://github.com/user-attachments/assets/6ae38394-f099-4843-bdbd-99bf32d574ac)
![Figure_7](https://github.com/user-attachments/assets/b0c24f24-c495-4f14-8948-bd8d7b027091)
![final](https://github.com/user-attachments/assets/2487ee01-bf3a-43a9-b25e-4d798b1a5f09)



---

## Future Improvements  
- Handle class imbalance with **SMOTE** or **ADASYN**.  
- Optimize hyperparameters via **Optuna** or **Bayesian Optimization**.  
- Deploy on **AWS EC2** for scalability.  
- Add **SHAP** explanations for model interpretability.  

---

## Contributing  
Contributions are welcome!  
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/new-feature`.  
3. Commit changes: `git commit -m "Add new feature"`.  
4. Push to the branch: `git push origin feature/new-feature`.  
5. Open a Pull Request.  

---

## License  
Distributed under the MIT License. See `LICENSE` for details.  

---

**Disclaimer**: This project uses a synthetic dataset for educational purposes. Real-world performance may vary.
