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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os. path.join(current_dir, 'data', 'Air_Quality. csv'),
        os.path.join(current_dir, 'data', 'air_quality. csv'),
        'data/Air_Quality.csv',
        'data/air_quality.csv',
        'Air_Quality.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except: 
                continue
    st.error("❌ Could not find Air_Quality.csv")
    return None

df = load_data()

# Sidebar
st.sidebar.title("🌬️ Air Quality Analysis")
page = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Preprocessing", "ML Model", "Conclusion"])

# ===================== INTRODUCTION =====================
if page == "Introduction":
    st. title("🌬️ Air Quality Analysis & Prediction")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Project Overview
    This application analyzes air quality data to understand pollution patterns 
    and build predictive models using machine learning. 
    
    ### 🎯 Objectives
    - Perform comprehensive EDA (15 analyses)
    - Preprocess data for machine learning
    - Train and evaluate ML models
    - Make runtime predictions
    """)
    
    if df is not None:
        st. markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric", len(df. select_dtypes(include=[np.number]).columns))
        col4.metric("Categorical", len(df.select_dtypes(include=['object']).columns))
        
        # Find city/location column
        city_col = None
        for col in df.select_dtypes(include=['object']).columns:
            if 2 <= df[col].nunique() <= 50:
                city_col = col
                break
        
        if city_col: 
            st.markdown(f"### 🌍 Data by {city_col}")
            cities = df[city_col].unique()
            
            col1, col2 = st. columns([2, 1])
            with col1:
                selected = st.multiselect(
                    f"Select {city_col}(s) to view",
                    options=cities. tolist(),
                    default=cities[: 3].tolist() if len(cities) >= 3 else cities.tolist()
                )
            with col2:
                rows_option = st.selectbox("Rows to display", ["Sample (10 per city)", "50 rows", "100 rows", "All data"])
            
            if selected:
                filtered_df = df[df[city_col]. isin(selected)]
                st.write(f"**Total: {len(filtered_df):,} records from {len(selected)} location(s)**")
                
                if rows_option == "Sample (10 per city)":
                    sample_per_city = []
                    for city in selected: 
                        city_data = filtered_df[filtered_df[city_col] == city]. head(10)
                        sample_per_city.append(city_data)
                    display_df = pd.concat(sample_per_city)
                elif rows_option == "50 rows": 
                    display_df = filtered_df.head(50)
                elif rows_option == "100 rows":
                    display_df = filtered_df.head(100)
                else:
                    display_df = filtered_df
                
                st.dataframe(display_df, height=400)
                
                st.markdown("#### 📈 Records per Location")
                city_counts = filtered_df[city_col].value_counts()
                st.bar_chart(city_counts)
        else:
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(20))
        
        st. markdown("### 📋 Column Names")
        st.write(df.columns.tolist())

# ===================== EDA =====================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    if df is not None:
        eda = EDAAnalyzer(df)
        
        analysis = st.selectbox("Select Analysis", [
            "1. Summary Statistics",
            "2. Missing Values",
            "3. Data Types & Unique Values",
            "4. Correlation Matrix",
            "5. Top Correlations",
            "6. Outlier Detection",
            "7. Distribution Analysis",
            "8. Categorical Analysis",
            "9. Grouped Aggregation",
            "10. Value Ranges",
            "11. Zero/Negative Analysis",
            "12. Duplicate Analysis",
            "13. Quartile Analysis",
            "14. Variability Analysis",
            "15. Visualizations"
        ])
        
        st.markdown("---")
        
        if analysis == "1. Summary Statistics":
            st.subheader("Summary Statistics")
            st.dataframe(eda.summary_statistics())
        
        elif analysis == "2. Missing Values":
            st.subheader("Missing Value Analysis")
            missing = eda.missing_values()
            if missing.empty:
                st. success("No missing values found!")
            else:
                st.dataframe(missing)
                fig, ax = plt.subplots(figsize=(10, 5))
                missing['Missing Count'].plot(kind='bar', ax=ax, color='coral')
                plt.title("Missing Values by Column")
                plt. xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis == "3. Data Types & Unique Values": 
            st.subheader("Data Types and Unique Values")
            st.dataframe(eda.data_types_info())
        
        elif analysis == "4. Correlation Matrix": 
            st.subheader("Correlation Matrix")
            corr = eda.correlation_matrix()
            if not corr.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns. heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation")
        
        elif analysis == "5. Top Correlations": 
            st.subheader("Top Feature Correlations")
            top_corr = eda.top_correlations()
            if not top_corr.empty:
                st.dataframe(top_corr)
            else:
                st.warning("Not enough features for correlation analysis")
        
        elif analysis == "6. Outlier Detection":
            st.subheader("Outlier Detection (IQR Method)")
            outliers = eda.detect_outliers()
            if outliers.empty:
                st. success("No significant outliers detected!")
            else:
                st.dataframe(outliers)
                st.subheader("Box Plots")
                default_cols = eda.numeric_cols[: 3] if len(eda.numeric_cols) >= 3 else eda.numeric_cols
                cols = st.multiselect("Select columns for box plot", eda.numeric_cols, default=default_cols)
                if cols:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    df[cols].boxplot(ax=ax)
                    plt. xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif analysis == "7. Distribution Analysis":
            st.subheader("Feature Distribution Analysis")
            st.dataframe(eda.distribution_analysis())
            if eda.numeric_cols:
                col = st.selectbox("Select column for histogram", eda.numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis == "8. Categorical Analysis":
            st.subheader("Categorical Analysis")
            if eda.categorical_cols:
                col = st.selectbox("Select categorical column", eda.categorical_cols)
                cat_data = eda.categorical_analysis(col)
                st.dataframe(cat_data)
                fig, ax = plt.subplots(figsize=(10, 5))
                cat_data['Count'].head(10).plot(kind='bar', ax=ax, color='teal')
                plt.title(f"Top 10 values in {col}")
                plt. xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No categorical columns found")
        
        elif analysis == "9. Grouped Aggregation":
            st.subheader("Grouped Aggregation")
            if eda.categorical_cols and eda.numeric_cols:
                group_col = st.selectbox("Group by", eda.categorical_cols)
                value_col = st.selectbox("Aggregate column", eda.numeric_cols)
                grouped = eda.grouped_stats(group_col, value_col)
                st.dataframe(grouped)
            else:
                st.warning("Need both categorical and numeric columns")
        
        elif analysis == "10. Value Ranges":
            st.subheader("Value Range Analysis")
            st.dataframe(eda.value_ranges())
        
        elif analysis == "11. Zero/Negative Analysis":
            st.subheader("Zero and Negative Value Analysis")
            st.dataframe(eda.zero_negative_analysis())
        
        elif analysis == "12. Duplicate Analysis":
            st.subheader("Duplicate Analysis")
            dup = eda.duplicate_analysis()
            col1, col2 = st. columns(2)
            col1.metric("Total Duplicates", dup['Total Duplicates'])
            col2.metric("Duplicate %", f"{dup['Duplicate %']}%")
        
        elif analysis == "13. Quartile Analysis":
            st.subheader("Quartile Analysis")
            st.dataframe(eda.quartile_analysis())
        
        elif analysis == "14. Variability Analysis": 
            st.subheader("Coefficient of Variation Analysis")
            st.dataframe(eda.variability_analysis())
        
        elif analysis == "15. Visualizations":
            st.subheader("Data Visualizations")
            viz_type = st.selectbox("Select visualization", ["Histogram", "Scatter Plot", "Box Plot", "Pair Plot"])
            
            if viz_type == "Histogram":
                col = st.selectbox("Select column", eda.numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
                plt.title(f"Histogram of {col}")
                plt.tight_layout()
                st.pyplot(fig)
            
            elif viz_type == "Scatter Plot":
                col1_sel = st.selectbox("X-axis", eda.numeric_cols, index=0)
                col2_sel = st.selectbox("Y-axis", eda.numeric_cols, index=min(1, len(eda.numeric_cols)-1))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[col1_sel], df[col2_sel], alpha=0.5)
                plt.xlabel(col1_sel)
                plt.ylabel(col2_sel)
                plt.title(f"{col1_sel} vs {col2_sel}")
                plt. tight_layout()
                st. pyplot(fig)
            
            elif viz_type == "Box Plot":
                default_cols = eda.numeric_cols[:3] if len(eda.numeric_cols) >= 3 else eda.numeric_cols
                cols = st.multiselect("Select columns", eda.numeric_cols, default=default_cols)
                if cols:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    df[cols].boxplot(ax=ax)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif viz_type == "Pair Plot":
                default_cols = eda.numeric_cols[:3] if len(eda.numeric_cols) >= 3 else eda.numeric_cols
                cols = st.multiselect("Select columns (max 4)", eda.numeric_cols[: 4], default=default_cols)
                if len(cols) >= 2:
                    fig = sns.pairplot(df[cols]. dropna())
                    st.pyplot(fig)

# ===================== PREPROCESSING =====================
elif page == "Preprocessing": 
    st.title("🔧 Data Preprocessing")
    st.markdown("---")
    
    if df is not None:
        st.subheader("Preprocessing Options")
        
        col1, col2 = st. columns(2)
        with col1:
            missing_strategy = st.selectbox("Missing Value Strategy", ["median", "mean", "most_frequent"])
            encoding_method = st.selectbox("Encoding Method", ["label", "onehot"])
        with col2:
            scaling_method = st.selectbox("Scaling Method", ["standard", "minmax"])
            outlier_method = st.selectbox("Outlier Handling", ["clip", "remove", "none"])
        
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
            
            st.subheader("Preprocessing Steps Applied")
            for i, step in enumerate(preprocessor.get_summary(), 1):
                st.write(f"{i}. {step}")
            
            st.subheader("Processed Data Preview")
            st.dataframe(preprocessor.df. head())
            
            col1, col2 = st. columns(2)
            col1.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
            col2.metric("Processed Shape", f"{preprocessor. df.shape[0]} × {preprocessor.df.shape[1]}")

# ===================== ML MODEL =====================
elif page == "ML Model":
    st.title("🤖 Machine Learning Model")
    st.markdown("---")
    
    if df is not None:
        work_df = st.session_state.get('processed_df', df. copy())
        numeric_cols = work_df.select_dtypes(include=[np. number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Not enough numeric columns for modeling")
        else:
            st.subheader("Model Configuration")
            
            col1, col2 = st. columns(2)
            with col1:
                target_col = st.selectbox("Select Target Variable", numeric_cols)
                model_type = st.selectbox("Model Type", ["regression", "classification"])
            with col2:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
               # NEW CODE
 if model_type == "regression":
    model_name = st. selectbox("Select Model", ["Random Forest", "Linear Regression", "Gradient Boosting"])
else:
    model_name = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Gradient Boosting"])
            
            if st.button("Train Model", type="primary"):
                feature_cols = [c for c in numeric_cols if c != target_col]
                X = work_df[feature_cols]. dropna()
                y = work_df. loc[X. index, target_col]
                
                if model_type == "classification":
                    y = pd.cut(y, bins=3, labels=[0, 1, 2]).astype(int)
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                ml_model = MLModel(model_type=model_type)
                ml_model.train(X_train, y_train, model_name=model_name)
                metrics, predictions = ml_model.evaluate(X_test, y_test)
                
                st.session_state['ml_model'] = ml_model
                st.session_state['feature_cols'] = feature_cols
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success("✅ Model trained successfully!")
                
                st.subheader("Model Performance")
                metric_cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    metric_cols[i].metric(name, value)
                
                importance = ml_model.get_feature_importance()
                if importance is not None:
                    st. subheader("Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance.head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax, color='steelblue')
                    plt.title("Top 10 Feature Importances")
                    plt.tight_layout()
                    st. pyplot(fig)
                
                st.subheader("Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, predictions, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel("Actual")
                plt. ylabel("Predicted")
                plt.title("Actual vs Predicted Values")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("🔮 Make Runtime Predictions")
            
            if 'ml_model' in st. session_state:
                ml_model = st.session_state['ml_model']
                feature_cols = st.session_state['feature_cols']
                
                st.write("Enter values for prediction:")
                
                input_data = {}
                cols = st.columns(3)
                for i, feat in enumerate(feature_cols):
                    with cols[i % 3]:
                        min_val = float(work_df[feat].min())
                        max_val = float(work_df[feat].max())
                        mean_val = float(work_df[feat].mean())
                        input_data[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=mean_val)
                
                if st.button("Predict", type="secondary"):
                    prediction = ml_model.predict(input_data)
                    st. success(f"### 🎯 Predicted Value: {prediction[0]:.4f}")
            else:
                st.info("👆 Train a model first to make predictions")

# ===================== CONCLUSION =====================
elif page == "Conclusion": 
    st.title("📝 Conclusion")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Takeaways
    
    1. **Data Understanding**: Comprehensive EDA revealed important patterns in air quality data
    2. **Data Quality**: Identified and handled missing values, outliers, and inconsistencies
    3. **Feature Engineering**: Applied encoding and scaling to prepare data for ML
    4. **Model Performance**: Built and evaluated machine learning models
    5. **Interactive Predictions**: Real-time predictions based on user input
    
    ### 📊 Project Summary
    - ✅ Performed **15 different EDA analyses**
    - ✅ Applied complete **data preprocessing pipeline**
    - ✅ Trained **ML models** with evaluation metrics
    - ✅ Built **interactive Streamlit application**
    - ✅ Enabled **runtime predictions**
    
    ### 🛠️ Technologies Used
    - Python, Pandas, NumPy
    - Scikit-learn
    - Matplotlib, Seaborn
    - Streamlit
    """)
    
    st.markdown("---")

    st.markdown("**Course:** IDS F24 | **Instructor:** Dr M Nadeem Majeed")
