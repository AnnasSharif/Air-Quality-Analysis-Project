# 🌬️ Air Quality Analysis & Prediction

**Course:** IDS F24  
**Instructor:** Dr M Nadeem Majeed

## 📌 Project Overview

This project performs comprehensive analysis on Air Quality data including:
- Exploratory Data Analysis (15 analyses)
- Data Preprocessing
- Machine Learning Model Training
- Interactive Predictions via Streamlit

## 📂 Project Structure

```
IDS_Project/
├── data/
│   └── Air_Quality. csv
├── src/
│   ├── eda.py              # EDA functions
│   ├── preprocessing. py    # Data preprocessing
│   └── model.py            # ML model training
├── app. py                  # Main Streamlit app
├── requirements.txt
├── README.md
└── . gitignore
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/IDS_Project.git
cd IDS_Project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 📊 Features

### EDA (15 Analyses)
1. Summary Statistics
2. Missing Value Analysis
3. Data Types & Unique Values
4. Correlation Matrix
5. Top Correlations
6. Outlier Detection
7. Distribution Analysis
8. Categorical Analysis
9. Grouped Aggregation
10. Value Ranges
11. Zero/Negative Analysis
12. Duplicate Analysis
13. Quartile Analysis
14. Variability Analysis
15. Visualizations

### Preprocessing
- Missing value imputation
- Categorical encoding (Label/One-Hot)
- Feature scaling (Standard/MinMax)
- Outlier handling (Clip/Remove)

### Machine Learning
- Linear Regression
- Random Forest Regressor
- Model evaluation metrics
- Feature importance
- **Runtime predictions**

## 🌐 Deployment

Deploy on Streamlit Cloud: 
1. Push code to GitHub
2. Go to [share. streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy! 

## 📝 License

This project is for educational purposes - IDS F24 Course