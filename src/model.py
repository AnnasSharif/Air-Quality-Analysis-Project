"""
Machine Learning Model Module
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn. metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix)
import pickle


class MLModel:
    def __init__(self, model_type='regression'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = {}
    
    def get_models(self):
        if self.model_type == 'regression':
            return {
                'Linear Regression': LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
            }
        else:
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
            }
    
    def train(self, X_train, y_train, model_name='Random Forest Regressor'):
        models = self.get_models()
        self.model = models. get(model_name)
        self.feature_names = X_train.columns. tolist()
        self.model.fit(X_train, y_train)
        return self. model
    
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        if self.model_type == 'regression':
            self.metrics = {
                'R² Score': round(r2_score(y_test, predictions), 4),
                'RMSE': round(np.sqrt(mean_squared_error(y_test, predictions)), 4),
                'MAE': round(mean_absolute_error(y_test, predictions), 4)
            }
        else: 
            self.metrics = {
                'Accuracy': round(accuracy_score(y_test, predictions), 4),
                'Precision': round(precision_score(y_test, predictions, average='weighted', zero_division=0), 4),
                'Recall': round(recall_score(y_test, predictions, average='weighted', zero_division=0), 4),
                'F1 Score':  round(f1_score(y_test, predictions, average='weighted', zero_division=0), 4)
            }
        
        return self.metrics, predictions
    
    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        return self.model.predict(input_data)
    
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self. model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance
        return None
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_names}, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['features']