"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, df):
        self.df = df. copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.label_encoders = {}
        self.scaler = None
        self.steps = []
    
    # Handle Missing Values
    def handle_missing(self, numeric_strategy='median', categorical_strategy='mode'):
        # Numeric columns
        if self.numeric_cols:
            imputer = SimpleImputer(strategy=numeric_strategy)
            self.df[self.numeric_cols] = imputer.fit_transform(self.df[self.numeric_cols])
            self.steps.append(f"Imputed numeric columns with {numeric_strategy}")
        
        # Categorical columns
        for col in self.categorical_cols:
            if self.df[col].isnull().sum() > 0:
                fill_val = self.df[col].mode()[0] if categorical_strategy == 'mode' else 'Unknown'
                self.df[col]. fillna(fill_val, inplace=True)
        if self.categorical_cols:
            self.steps.append(f"Filled categorical missing values with {categorical_strategy}")
        
        return self.df
    
    # Encode Categorical Variables
    def encode_categorical(self, method='label'):
        if method == 'label':
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col]. astype(str))
                self.label_encoders[col] = le
            self.steps. append(f"Label encoded:  {self.categorical_cols}")
        elif method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=self.categorical_cols)
            self.steps.append(f"One-hot encoded: {self.categorical_cols}")
        return self.df
    
    # Scale Features
    def scale_features(self, method='standard', columns=None):
        if columns is None:
            columns = self. numeric_cols
        columns = [c for c in columns if c in self.df.columns]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        self.steps.append(f"Applied {method} scaling")
        return self.df
    
    # Handle Outliers
    def handle_outliers(self, method='clip'):
        for col in self.numeric_cols:
            if col not in self.df.columns:
                continue
            Q1, Q3 = self. df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            
            if method == 'clip': 
                self.df[col] = self.df[col].clip(lower, upper)
            elif method == 'remove':
                self. df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        self.steps.append(f"Handled outliers using {method}")
        return self.df
    
    # Train-Test Split
    def split_data(self, target_col, test_size=0.2, random_state=42):
        feature_cols = [c for c in self.df.select_dtypes(include=[np. number]).columns if c != target_col]
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.steps.append(f"Split data:  {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    # Get preprocessing summary
    def get_summary(self):
        return self.steps