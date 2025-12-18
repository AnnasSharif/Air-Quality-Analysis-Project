"""
Air Quality Analysis - Streamlit Application
Course: IDS F24 | Instructor: Dr M Nadeem Majeed
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src. eda import EDAAnalyzer
from src.preprocessing import DataPreprocessor
from src.model import MLModel

# Page Config
st.set_page_config(page_title="Air Quality Analysis", page_icon="🌬️", layout="wide")

# Load Data
@st.cache_data
def load_data():
    paths = ['data/Air_Quality. csv', 'data/air_quality. csv', 'Air_Quality.csv']
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except: 
                continue
    st.error("❌ Could not find Air_Quality.csv")
    return None

df = load_data()

# Sidebar Navigation
st.sidebar.title("🌬️ Air Quality Analysis")
page = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Preprocessing", "ML Model", "Conclusion"])

# Helper Functions
def display_metrics(data_dict):
    """Display metrics in columns"""
    cols = st.columns(len(data_dict))
    for i, (key, val) in enumerate(data_dict. items()):
        cols[i].metric(key, val)

def find_city_column(df):
    """Find the city/location column"""
    for col in df.select_dtypes(include=['object']).columns:
        if 2 <= df[col].nunique() <= 50:
            return col
    return None

def plot_figure(plot_func, figsize=(10, 5), **kwargs):
    """Wrapper for matplotlib plots"""
    fig, ax = plt.subplots(figsize=figsize)
    plot_func(ax, **kwargs)
    plt.tight_layout()
    st.pyplot(fig)

# ===================== INTRODUCTION =====================
if page == "Introduction":
    st.title("🌬️ Air Quality Analysis & Prediction")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Project Overview
    Analyze air quality data, understand pollution patterns, and build ML models. 
    
    ### 🎯 Objectives
    - **15 EDA analyses** | Preprocessing pipeline | ML models | Runtime predictions
    """)
    
    if df is not None:
        st. markdown("### 📊 Dataset Overview")
        display_metrics({
            "Rows": len(df),
            "Columns": len(df.columns),
            "Numeric": len(df.select_dtypes(include=[np.number]).columns),
            "Categorical": len(df. select_dtypes(include=['object']).columns)
        })
        
        city_col = find_city_column(df)
        
        if city_col:
            st.markdown(f"### 🌍 Data by {city_col}")
            cities = df[city_col].unique()
            
            col1, col2 = st. columns([2, 1])
            with col1:
                selected = st.multiselect(f"Select {city_col}(s)", cities.tolist(), 
                                         default=cities[: 3]. tolist() if len(cities) >= 3 else cities. tolist())
            with col2:
                rows_option = st.selectbox("Rows", ["Sample (10/city)", "50", "100", "All"])
            
            if selected:
                filtered_df = df[df[city_col].isin(selected)]
                st.write(f"**{len(filtered_df):,} records from {len(selected)} location(s)**")
                
                # Display logic
                row_map = {"Sample (10/city)": pd.concat([filtered_df[filtered_df[city_col]==c].head(10) for c in selected]),
                          "50": filtered_df. head(50), "100": filtered_df.head(100), "All": filtered_df}
                st.dataframe(row_map[rows_option], height=400)
                
                st.markdown("#### 📈 Records per Location")
                st.bar_chart(filtered_df[city_col].value_counts())
        else:
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(20))

# ===================== EDA =====================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    if df is not None:
        eda = EDAAnalyzer(df)
        
        analysis = st.selectbox("Select Analysis", [
            f"{i+1}. {name}" for i, name in enumerate([
                "Summary Statistics", "Missing Values", "Data Types & Unique Values",
                "Correlation Matrix", "Top Correlations", "Outlier Detection",
                "Distribution Analysis", "Categorical Analysis", "Grouped Aggregation",
                "Value Ranges", "Zero/Negative Analysis", "Duplicate Analysis",
                "Quartile Analysis", "Variability Analysis", "Visualizations"
            ])
        ])
        
        st.markdown("---")
        
        # Analysis mapping
        analysis_map = {
            "1":  lambda:  st.dataframe(eda.summary_statistics()),
            "2": lambda: display_missing_values(eda),
            "3": lambda: st.dataframe(eda.data_types_info()),
            "4": lambda:  plot_correlation(eda),
            "5": lambda: st.dataframe(eda.top_correlations()),
            "6": lambda: display_outliers(eda, df),
            "7": lambda:  display_distributions(eda, df),
            "8": lambda: display_categorical(eda, df),
            "9": lambda: display_grouped(eda),
            "10": lambda: st.dataframe(eda.value_ranges()),
            "11": lambda: st.dataframe(eda.zero_negative_analysis()),
            "12": lambda: display_metrics(eda.duplicate_analysis()),
            "13": lambda: st.dataframe(eda.quartile_analysis()),
            "14": lambda: st.dataframe(eda.variability_analysis()),
            "15": lambda: display_visualizations(eda, df)
        }
        
        analysis_num = analysis. split(". ")[0]
        analysis_map[analysis_num]()

# Helper functions for EDA
def display_missing_values(eda):
    missing = eda.missing_values()
    if missing.empty:
        st.success("No missing values!")
    else:
        st. dataframe(missing)
        plot_figure(lambda ax: missing['Missing Count'].plot(kind='bar', ax=ax, color='coral'))

def plot_correlation(eda):
    corr = eda.correlation_matrix()
    if not corr.empty:
        plot_figure(lambda ax: sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f'), 
                   figsize=(12, 8))
    else:
        st.warning("Not enough numeric columns")

def display_outliers(eda, df):
    outliers = eda.detect_outliers()
    st.dataframe(outliers) if not outliers.empty else st.success("No outliers detected!")
    
    cols = st.multiselect("Box plot columns", eda.numeric_cols, default=eda.numeric_cols[: 3])
    if cols:
        plot_figure(lambda ax:  df[cols].boxplot(ax=ax), figsize=(12, 5))

def display_distributions(eda, df):
    st.dataframe(eda.distribution_analysis())
    if eda.numeric_cols:
        col = st.selectbox("Select column", eda.numeric_cols)
        plot_figure(lambda ax: df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black'))

def display_categorical(eda, df):
    if eda.categorical_cols:
        col = st.selectbox("Select categorical", eda.categorical_cols)
        cat_data = eda.categorical_analysis(col)
        st.dataframe(cat_data)
        plot_figure(lambda ax: cat_data['Count'].head(10).plot(kind='bar', ax=ax, color='teal'))
    else:
        st.warning("No categorical columns")

def display_grouped(eda):
    if eda.categorical_cols and eda.numeric_cols:
        group_col = st.selectbox("Group by", eda.categorical_cols)
        value_col = st.selectbox("Aggregate", eda.numeric_cols)
        st.dataframe(eda.grouped_stats(group_col, value_col))
    else:
        st.warning("Need both categorical and numeric columns")

def display_visualizations(eda, df):
    viz_type = st.selectbox("Visualization", ["Histogram", "Scatter Plot", "Box Plot", "Pair Plot"])
    
    if viz_type == "Histogram":
        col = st.selectbox("Column", eda.numeric_cols)
        plot_figure(lambda ax: df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black'))
    
    elif viz_type == "Scatter Plot": 
        x = st.selectbox("X-axis", eda.numeric_cols, index=0)
        y = st.selectbox("Y-axis", eda.numeric_cols, index=min(1, len(eda.numeric_cols)-1))
        plot_figure(lambda ax: ax.scatter(df[x], df[y], alpha=0.5), figsize=(10, 6))
    
    elif viz_type == "Box Plot":
        cols = st.multiselect("Columns", eda.numeric_cols, default=eda.numeric_cols[:3])
        if cols:
            plot_figure(lambda ax: df[cols].boxplot(ax=ax), figsize=(12, 5))
    
    elif viz_type == "Pair Plot":
        cols = st.multiselect("Columns (max 4)", eda.numeric_cols[: 4], default=eda.numeric_cols[:3])
        if len(cols) >= 2:
            fig = sns.pairplot(df[cols]. dropna())
            st.pyplot(fig)

# ===================== PREPROCESSING =====================
elif page == "Preprocessing":
    st.title("🔧 Data Preprocessing")
    st.markdown("---")
    
    if df is not None:
        st.subheader("Options")
        
        col1, col2 = st. columns(2)
        with col1:
            missing_strategy = st.selectbox("Missing Values", ["median", "mean", "most_frequent"])
            encoding_method = st.selectbox("Encoding", ["label", "onehot"])
        with col2:
            scaling_method = st.selectbox("Scaling", ["standard", "minmax"])
            outlier_method = st.selectbox("Outliers", ["clip", "remove", "none"])
        
        if st.button("Apply Preprocessing", type="primary"):
            preprocessor = DataPreprocessor(df)
            preprocessor.handle_missing(numeric_strategy=missing_strategy)
            preprocessor.encode_categorical(method=encoding_method)
            if outlier_method != 'none':
                preprocessor. handle_outliers(method=outlier_method)
            preprocessor.scale_features(method=scaling_method)
            
            st.session_state['processed_df'] = preprocessor.df
            st.session_state['preprocessor'] = preprocessor
            
            st.success("✅ Preprocessing completed!")
            
            st.subheader("Steps Applied")
            for i, step in enumerate(preprocessor.get_summary(), 1):
                st.write(f"{i}. {step}")
            
            st.subheader("Preview")
            st.dataframe(preprocessor.df.head())
            
            display_metrics({
                "Original":  f"{df.shape[0]} × {df.shape[1]}",
                "Processed": f"{preprocessor.df.shape[0]} × {preprocessor.df. shape[1]}"
            })

# ===================== ML MODEL =====================
elif page == "ML Model":
    st.title("🤖 Machine Learning Model")
    st.markdown("---")
    
    if df is not None:
        work_df = st.session_state.get('processed_df', df. copy())
        numeric_cols = work_df.select_dtypes(include=[np. number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
        else:
            st.subheader("Configuration")
            
            col1, col2 = st. columns(2)
            with col1:
                target_col = st.selectbox("Target Variable", numeric_cols)
                model_type = st.selectbox("Type", ["regression", "classification"])
            with col2:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
                models = ["Random Forest Regressor", "Linear Regression"] if model_type == "regression" else ["Random Forest Classifier", "Logistic Regression"]
                model_name = st.selectbox("Model", models)
            
            if st.button("Train Model", type="primary"):
                from sklearn.model_selection import train_test_split
                
                feature_cols = [c for c in numeric_cols if c != target_col]
                X = work_df[feature_cols]. dropna()
                y = work_df. loc[X.index, target_col]
                
                if model_type == "classification":
                    y = pd.cut(y, bins=3, labels=[0, 1, 2]).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                ml_model = MLModel(model_type=model_type)
                ml_model.train(X_train, y_train, model_name=model_name)
                metrics, predictions = ml_model.evaluate(X_test, y_test)
                
                # Store in session
                st.session_state. update({'ml_model': ml_model, 'feature_cols': feature_cols, 
                                        'X_test': X_test, 'y_test': y_test})
                
                st.success("✅ Model trained!")
                
                st.subheader("Performance")
                display_metrics(metrics)
                
                # Feature importance
                importance = ml_model.get_feature_importance()
                if importance is not None:
                    st. subheader("Feature Importance")
                    plot_figure(lambda ax: importance. head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax, color='steelblue'), figsize=(10, 6))
                
                # Actual vs Predicted
                st.subheader("Actual vs Predicted")
                plot_figure(lambda ax: [ax.scatter(y_test, predictions, alpha=0.5), 
                           ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')], figsize=(10, 6))
            
            st.markdown("---")
            st.subheader("🔮 Make Predictions")
            
            if 'ml_model' in st. session_state:
                ml_model = st.session_state['ml_model']
                feature_cols = st.session_state['feature_cols']
                
                st.write("Enter values:")
                input_data = {}
                cols = st.columns(3)
                
                for i, feat in enumerate(feature_cols):
                    with cols[i % 3]:
                        input_data[feat] = st.number_input(feat, value=float(work_df[feat].mean()),
                                                          min_value=float(work_df[feat].min()),
                                                          max_value=float(work_df[feat].max()))
                
                if st.button("Predict", type="secondary"):
                    prediction = ml_model.predict(input_data)
                    st. success(f"### 🎯 Predicted:  {prediction[0]:. 4f}")
            else:
                st.info("👆 Train a model first")

# ===================== CONCLUSION =====================
elif page == "Conclusion":
    st.title("📝 Conclusion")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Achievements
    
    ✅ **15 EDA analyses** | ✅ **Preprocessing pipeline** | ✅ **ML models** | ✅ **Interactive predictions**
    
    ### 🛠️ Tech Stack
    Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Streamlit
    
    ---
    **Course:** IDS F24 | **Instructor:** Dr M Nadeem Majeed
    """)
