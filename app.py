"""
Air Quality Analysis - Streamlit Application
Course: IDS F24 | Instructor: Dr M Nadeem Majeed
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src. eda import EDAAnalyzer
from src.preprocessing import DataPreprocessor
from src.model import MLModel

st.set_page_config(page_title="Air Quality Analysis", page_icon="🌬️", layout="wide")

@st.cache_data
def load_data():
    for path in ['data/Air_Quality. csv', 'data/air_quality. csv', 'Air_Quality.csv']:
        if os.path.exists(path):
            try:  return pd.read_csv(path)
            except: continue
    st.error("❌ Could not find Air_Quality.csv")
    return None

df = load_data()
st.sidebar.title("🌬️ Air Quality Analysis")
page = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Preprocessing", "ML Model", "Conclusion"])

def metrics(d): 
    cols = st.columns(len(d))
    for i, (k, v) in enumerate(d.items()): cols[i].metric(k, v)

def plot(func, size=(10, 5)):
    fig, ax = plt.subplots(figsize=size)
    func(ax)
    plt.tight_layout()
    st.pyplot(fig)

# ===================== INTRODUCTION =====================
if page == "Introduction":
    st.title("🌬️ Air Quality Analysis & Prediction")
    st.markdown("### 📌 Overview\nComprehensive EDA, Preprocessing, ML Models, and Predictions\n\n### 🎯 Features\n✅ 15 EDA analyses | ✅ Preprocessing | ✅ ML models | ✅ Runtime predictions")
    
    if df is not None:
        st.markdown("### 📊 Dataset")
        metrics({"Rows": len(df), "Columns": len(df.columns), "Numeric": len(df.select_dtypes(include=[np.number]).columns), "Categorical": len(df. select_dtypes(include=['object']).columns)})
        
        city_col = next((c for c in df.select_dtypes(include=['object']).columns if 2 <= df[c].nunique() <= 50), None)
        
        if city_col:
            st.markdown(f"### 🌍 {city_col} Data")
            cities = df[city_col].unique()
            col1, col2 = st. columns([2, 1])
            selected = col1.multiselect(f"Select {city_col}", cities. tolist(), default=cities[:3]. tolist() if len(cities) >= 3 else cities. tolist())
            rows_opt = col2.selectbox("Rows", ["Sample (10/city)", "50", "100", "All"])
            
            if selected: 
                fdf = df[df[city_col].isin(selected)]
                st.write(f"**{len(fdf):,} records from {len(selected)} location(s)**")
                display_df = pd.concat([fdf[fdf[city_col]==c]. head(10) for c in selected]) if rows_opt == "Sample (10/city)" else fdf.head(50 if rows_opt == "50" else 100 if rows_opt == "100" else len(fdf))
                st.dataframe(display_df, height=400)
                st.markdown("#### 📈 Records per Location")
                st.bar_chart(fdf[city_col].value_counts())
        else:
            st.dataframe(df.head(20))

# ===================== EDA =====================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    if df is not None:
        eda = EDAAnalyzer(df)
        analysis = st.selectbox("Select Analysis", [f"{i+1}. {n}" for i, n in enumerate(["Summary Statistics", "Missing Values", "Data Types", "Correlation Matrix", "Top Correlations", "Outlier Detection", "Distribution", "Categorical", "Grouped Stats", "Value Ranges", "Zero/Negative", "Duplicates", "Quartiles", "Variability", "Visualizations"])])
        st.markdown("---")
        num = analysis.split(". ")[0]
        
        if num == "1":  st.dataframe(eda.summary_statistics())
        elif num == "2": 
            miss = eda.missing_values()
            st.dataframe(miss) if not miss.empty else st.success("No missing values!")
            if not miss.empty: plot(lambda ax: miss['Missing Count'].plot(kind='bar', ax=ax, color='coral'))
        elif num == "3":  st.dataframe(eda. data_types_info())
        elif num == "4": 
            corr = eda. correlation_matrix()
            plot(lambda ax: sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f'), size=(12, 8)) if not corr.empty else st.warning("Not enough numeric columns")
        elif num == "5":  st.dataframe(eda. top_correlations())
        elif num == "6": 
            out = eda.detect_outliers()
            st.dataframe(out) if not out.empty else st.success("No outliers!")
            cols = st.multiselect("Box plot", eda.numeric_cols, default=eda.numeric_cols[: 3])
            if cols: plot(lambda ax: df[cols].boxplot(ax=ax), size=(12, 5))
        elif num == "7": 
            st.dataframe(eda.distribution_analysis())
            if eda.numeric_cols:
                col = st.selectbox("Column", eda.numeric_cols)
                plot(lambda ax: df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black'))
        elif num == "8": 
            if eda.categorical_cols:
                col = st.selectbox("Categorical", eda.categorical_cols)
                cat = eda.categorical_analysis(col)
                st.dataframe(cat)
                plot(lambda ax: cat['Count'].head(10).plot(kind='bar', ax=ax, color='teal'))
            else:  st.warning("No categorical columns")
        elif num == "9": 
            if eda.categorical_cols and eda.numeric_cols:
                st.dataframe(eda.grouped_stats(st.selectbox("Group by", eda.categorical_cols), st.selectbox("Aggregate", eda.numeric_cols)))
            else: st.warning("Need categorical and numeric columns")
        elif num == "10": st.dataframe(eda.value_ranges())
        elif num == "11":  st.dataframe(eda. zero_negative_analysis())
        elif num == "12": metrics(eda.duplicate_analysis())
        elif num == "13": st.dataframe(eda.quartile_analysis())
        elif num == "14": st.dataframe(eda.variability_analysis())
        elif num == "15": 
            viz = st.selectbox("Type", ["Histogram", "Scatter", "Box", "Pair"])
            if viz == "Histogram": 
                col = st.selectbox("Column", eda.numeric_cols)
                plot(lambda ax: df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black'))
            elif viz == "Scatter":
                x, y = st.selectbox("X", eda.numeric_cols, index=0), st.selectbox("Y", eda.numeric_cols, index=min(1, len(eda.numeric_cols)-1))
                plot(lambda ax: [ax.scatter(df[x], df[y], alpha=0.5), ax.set_xlabel(x), ax.set_ylabel(y)], size=(10, 6))
            elif viz == "Box":
                cols = st.multiselect("Columns", eda.numeric_cols, default=eda.numeric_cols[: 3])
                if cols:  plot(lambda ax: df[cols].boxplot(ax=ax), size=(12, 5))
            elif viz == "Pair": 
                cols = st.multiselect("Columns (max 4)", eda.numeric_cols[: 4], default=eda.numeric_cols[:3])
                if len(cols) >= 2: st.pyplot(sns.pairplot(df[cols]. dropna()))

# ===================== PREPROCESSING =====================
elif page == "Preprocessing": 
    st.title("🔧 Data Preprocessing")
    st.markdown("---")
    
    if df is not None: 
        col1, col2 = st. columns(2)
        miss_strat = col1.selectbox("Missing", ["median", "mean", "most_frequent"])
        encode = col1.selectbox("Encoding", ["label", "onehot"])
        scale = col2.selectbox("Scaling", ["standard", "minmax"])
        outlier = col2.selectbox("Outliers", ["clip", "remove", "none"])
        
        if st.button("Apply Preprocessing", type="primary"):
            prep = DataPreprocessor(df)
            prep.handle_missing(numeric_strategy=miss_strat)
            prep.encode_categorical(method=encode)
            if outlier != 'none':  prep.handle_outliers(method=outlier)
            prep.scale_features(method=scale)
            
            st.session_state. update({'processed_df': prep.df, 'preprocessor': prep})
            st.success("✅ Preprocessing completed!")
            
            for i, step in enumerate(prep.get_summary(), 1): st.write(f"{i}. {step}")
            st.dataframe(prep.df. head())
            metrics({"Original": f"{df.shape[0]}×{df.shape[1]}", "Processed": f"{prep. df.shape[0]}×{prep.df.shape[1]}"})

# ===================== ML MODEL =====================
elif page == "ML Model":
    st.title("🤖 Machine Learning Model")
    st.markdown("---")
    
    if df is not None:
        wdf = st.session_state. get('processed_df', df. copy())
        ncols = wdf.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(ncols) < 2:
            st.error("Need 2+ numeric columns")
        else:
            col1, col2 = st. columns(2)
            target = col1.selectbox("Target", ncols)
            mtype = col1.selectbox("Type", ["regression", "classification"])
            tsize = col2.slider("Test Size", 0.1, 0.4, 0.2)
            mname = col2.selectbox("Model", ["Random Forest Regressor", "Linear Regression"] if mtype == "regression" else ["Random Forest Classifier", "Logistic Regression"])
            
            if st.button("Train Model", type="primary"):
                from sklearn.model_selection import train_test_split
                
                fcols = [c for c in ncols if c != target]
                X = wdf[fcols]. dropna()
                y = wdf. loc[X. index, target]
                if mtype == "classification":  y = pd.cut(y, bins=3, labels=[0, 1, 2]).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=42)
                
                ml = MLModel(model_type=mtype)
                ml.train(X_train, y_train, model_name=mname)
                mets, preds = ml.evaluate(X_test, y_test)
                
                st.session_state.update({'ml_model': ml, 'feature_cols': fcols, 'X_test': X_test, 'y_test': y_test})
                
                st.success("✅ Model trained!")
                metrics(mets)
                
                imp = ml.get_feature_importance()
                if imp is not None:  
                    st.subheader("Feature Importance")
                    plot(lambda ax: imp.head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax, color='steelblue'), size=(10, 6))
                
                st.subheader("Actual vs Predicted")
                plot(lambda ax: [ax.scatter(y_test, preds, alpha=0.5), ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--'), ax.set_xlabel("Actual"), ax.set_ylabel("Predicted")], size=(10, 6))
            
            st.markdown("---")
            st.subheader("🔮 Make Predictions")
            
            if 'ml_model' in st. session_state:
                ml = st.session_state['ml_model']
                fcols = st.session_state['feature_cols']
                
                inp = {}
                cols = st.columns(3)
                for i, f in enumerate(fcols):
                    with cols[i % 3]:
                        inp[f] = st.number_input(f, value=float(wdf[f].mean()), min_value=float(wdf[f].min()), max_value=float(wdf[f].max()))
                
                if st.button("Predict", type="secondary"):
                    pred = ml.predict(inp)
                    st.success(f"### 🎯 Predicted:  {pred[0]:. 4f}")
            else:
                st.info("👆 Train a model first")

# ===================== CONCLUSION =====================
elif page == "Conclusion":
    st.title("📝 Conclusion")
    st.markdown("---")
    st.markdown("""
    ### 🎯 Achievements
    ✅ **15 EDA analyses** | ✅ **Preprocessing pipeline** | ✅ **ML models** | ✅ **Predictions**
    
    ### 🛠️ Tech Stack
    Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Streamlit
    
    ---
    **Course:** IDS F24 | **Instructor:** Dr M Nadeem Majeed
    """)
