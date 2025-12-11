"""
Exploratory Data Analysis Module - 15 Analyses
"""

import pandas as pd
import numpy as np


class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Summary Statistics
    def summary_statistics(self):
        stats = self.df[self.numeric_cols].describe().T
        stats['median'] = self.df[self.numeric_cols].median()
        stats['mode'] = self.df[self. numeric_cols].mode().iloc[0]
        stats['skewness'] = self.df[self.numeric_cols].skew()
        return stats. round(3)
    
    # 2. Missing Value Analysis
    def missing_values(self):
        missing = pd.DataFrame({
            'Missing Count': self.df.isnull().sum(),
            'Missing %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        return missing[missing['Missing Count'] > 0]. sort_values('Missing Count', ascending=False)
    
    # 3. Data Types and Unique Values
    def data_types_info(self):
        return pd.DataFrame({
            'Data Type': self.df.dtypes,
            'Unique Values': self.df.nunique(),
            'Unique %': (self.df.nunique() / len(self.df) * 100).round(2)
        })
    
    # 4. Correlation Analysis
    def correlation_matrix(self):
        if len(self.numeric_cols) >= 2:
            return self.df[self.numeric_cols].corr().round(3)
        return pd.DataFrame()
    
    # 5. Top Correlations
    def top_correlations(self, n=10):
        corr = self.correlation_matrix()
        if corr.empty:
            return pd.DataFrame()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                pairs.append({
                    'Feature 1': corr.columns[i],
                    'Feature 2': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })
        return pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False).head(n)
    
    # 6. Outlier Detection (IQR)
    def detect_outliers(self):
        outliers = {}
        for col in self.numeric_cols:
            Q1, Q3 = self.df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if count > 0:
                outliers[col] = {'Count': count, 'Percentage':  round(count/len(self.df)*100, 2)}
        return pd.DataFrame(outliers).T if outliers else pd.DataFrame()
    
    # 7. Feature Distribution
    def distribution_analysis(self):
        dist = []
        for col in self.numeric_cols:
            skew = self.df[col]. skew()
            dist.append({
                'Column':  col,
                'Mean': self.df[col].mean(),
                'Std':  self.df[col].std(),
                'Skewness':  skew,
                'Distribution': 'Normal' if abs(skew) < 0.5 else ('Right Skewed' if skew > 0 else 'Left Skewed')
            })
        return pd.DataFrame(dist).round(3)
    
    # 8. Categorical Analysis
    def categorical_analysis(self, col):
        if col in self.categorical_cols:
            vc = self.df[col].value_counts()
            return pd. DataFrame({'Count': vc, 'Percentage': (vc/len(self.df)*100).round(2)})
        return pd.DataFrame()
    
    # 9. Grouped Aggregation
    def grouped_stats(self, group_col, value_col):
        if group_col in self.df.columns and value_col in self.numeric_cols:
            return self. df.groupby(group_col)[value_col].agg(['mean', 'median', 'std', 'min', 'max']).round(3)
        return pd.DataFrame()
    
    # 10. Value Range Analysis
    def value_ranges(self):
        ranges = []
        for col in self.numeric_cols:
            ranges.append({
                'Column': col,
                'Min': self.df[col].min(),
                'Max': self.df[col]. max(),
                'Range': self.df[col].max() - self.df[col].min()
            })
        return pd.DataFrame(ranges).round(3)
    
    # 11. Zero/Negative Analysis
    def zero_negative_analysis(self):
        analysis = []
        for col in self.numeric_cols:
            analysis.append({
                'Column':  col,
                'Zeros':  (self.df[col] == 0).sum(),
                'Negatives': (self.df[col] < 0).sum()
            })
        return pd.DataFrame(analysis)
    
    # 12. Duplicate Analysis
    def duplicate_analysis(self):
        return {
            'Total Duplicates': self.df.duplicated().sum(),
            'Duplicate %': round(self.df.duplicated().sum() / len(self.df) * 100, 2)
        }
    
    # 13. Quartile Analysis
    def quartile_analysis(self):
        return self.df[self.numeric_cols].quantile([0.25, 0.5, 0.75]).T. round(3)
    
    # 14. Coefficient of Variation
    def variability_analysis(self):
        var = []
        for col in self.numeric_cols:
            mean = self.df[col].mean()
            cv = (self.df[col]. std() / mean * 100) if mean != 0 else 0
            var. append({'Column': col, 'CV %': round(cv, 2)})
        return pd.DataFrame(var).sort_values('CV %', ascending=False)
    
    # 15. Basic Info
    def basic_info(self):
        return {
            'Rows': len(self.df),
            'Columns': len(self.df.columns),
            'Numeric Features': len(self.numeric_cols),
            'Categorical Features':  len(self.categorical_cols),
            'Total Missing':  self.df.isnull().sum().sum(),
            'Memory (MB)': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }