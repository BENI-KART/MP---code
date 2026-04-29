"""
Machine Learning Based Road Accident Severity Prediction System
Complete Implementation Example

This script demonstrates the complete pipeline for building an accident severity
prediction model using various machine learning algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score, roc_auc_score)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE

# Save/Load Models
import pickle


class AccidentSeverityPredictor:
    """
    A complete pipeline for accident severity prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_columns = None
        
    def load_data(self, filepath):
        """
        Load accident data from CSV file
        """
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df):
        """
        Perform initial data exploration
        """
        print("\n=== DATA EXPLORATION ===")
        print("\nDataset Info:")
        print(df.info())
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nStatistical Summary:")
        print(df.describe())
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])
        
        print("\nAccident Severity Distribution:")
        print(df['Accident_Severity'].value_counts())
        
        return df
    
    def visualize_data(self, df):
        """
        Create visualizations for EDA
        """
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Severity Distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        severity_counts = df['Accident_Severity'].value_counts()
        plt.bar(severity_counts.index, severity_counts.values, color=['red', 'orange', 'yellow'])
        plt.title('Accident Severity Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        
        # 2. Weather Conditions vs Severity
        plt.subplot(2, 3, 2)
        weather_severity = pd.crosstab(df['Weather_Conditions'], df['Accident_Severity'])
        weather_severity.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Weather vs Severity', fontsize=12, fontweight='bold')
        plt.xlabel('Weather Conditions')
        plt.ylabel('Count')
        plt.legend(title='Severity')
        plt.xticks(rotation=45)
        
        # 3. Time of Day Distribution
        plt.subplot(2, 3, 3)
        if 'Hour' in df.columns:
            df['Hour'].hist(bins=24, edgecolor='black')
            plt.title('Accidents by Hour of Day', fontsize=12, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Frequency')
        
        # 4. Road Type vs Severity
        plt.subplot(2, 3, 4)
        road_severity = pd.crosstab(df['Road_Type'], df['Accident_Severity'])
        road_severity.plot(kind='bar', ax=plt.gca())
        plt.title('Road Type vs Severity', fontsize=12, fontweight='bold')
        plt.xlabel('Road Type')
        plt.ylabel('Count')
        plt.legend(title='Severity')
        plt.xticks(rotation=45)
        
        # 5. Correlation Heatmap (numeric features only)
        plt.subplot(2, 3, 5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
        
        # 6. Number of Vehicles Distribution
        plt.subplot(2, 3, 6)
        if 'Number_of_Vehicles' in df.columns:
            df['Number_of_Vehicles'].value_counts().sort_index().plot(kind='bar')
            plt.title('Number of Vehicles Involved', fontsize=12, fontweight='bold')
            plt.xlabel('Number of Vehicles')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'eda_visualizations.png'")
        plt.close()
        
    def clean_data(self, df):
        """
        Clean and preprocess the data
        """
        print("\n=== DATA CLEANING ===")
        
        # Remove duplicates
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows")
        
        # Handle missing values
        # For categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled {col} with mode: {mode_value}")
        
        # For numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled {col} with median: {median_value}")
        
        # Remove outliers using IQR method for numerical columns
        for col in numerical_cols:
            if col != 'Accident_Severity':  # Don't remove outliers from target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    print(f"Removed {outliers} outliers from {col}")
        
        print(f"\nFinal dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones
        """
        print("\n=== FEATURE ENGINEERING ===")
        
        # Time-based features
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
            df['Hour'] = df['Time'].dt.hour
            df['Is_Rush_Hour'] = df['Hour'].apply(
                lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
            )
            print("Created: Hour, Is_Rush_Hour")
        
        # Date-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df['Season'] = df['Month'].apply(self._get_season)
            print("Created: DayOfWeek, Month, Is_Weekend, Season")
        
        # Interaction features
        if 'Number_of_Vehicles' in df.columns and 'Number_of_Lanes' in df.columns:
            df['Vehicle_Density'] = df['Number_of_Vehicles'] / (df['Number_of_Lanes'] + 1)
            print("Created: Vehicle_Density")
        
        # Severity indicators
        if 'Number_of_Casualties' in df.columns:
            df['High_Casualty'] = df['Number_of_Casualties'].apply(lambda x: 1 if x > 2 else 0)
            print("Created: High_Casualty")
        
        # Weather severity
        if 'Weather_Conditions' in df.columns:
            weather_severity_map = {
                'Fine': 0, 'Raining': 1, 'Snowing': 2, 'Fog': 1, 'Other': 1
            }
            df['Weather_Severity'] = df['Weather_Conditions'].map(weather_severity_map)
            print("Created: Weather_Severity")
        
        return df
    
    def _get_season(self, month):
        """Helper function to determine season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def encode_features(self, df):
        """
        Encode categorical variables
        """
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Accident_Severity']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df
    
    def prepare_features(self, df, target_col='Accident_Severity'):
        """
        Prepare features and target for modeling
        """
        print("\n=== PREPARING FEATURES ===")
        
        # Select feature columns (encoded categoricals + numerical)
        feature_cols = [col for col in df.columns if col.endswith('_Encoded')]
        
        # Add numerical features
        numerical_features = ['Speed_limit', 'Number_of_Vehicles', 
                            'Number_of_Casualties', 'Hour', 'Is_Rush_Hour',
                            'Is_Weekend', 'Vehicle_Density', 'High_Casualty']
        
        for feat in numerical_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """
        Split data and scale features
        """
        print("\n=== SPLITTING AND SCALING DATA ===")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance
        """
        print("\n=== HANDLING CLASS IMBALANCE ===")
        print(f"Original class distribution:\n{pd.Series(y_train).value_counts()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"\nAfter SMOTE:\n{pd.Series(y_resampled).value_counts()}")
            return X_resampled, y_resampled
        
        return X_train, y_train
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train and compare multiple models
        """
        print("\n=== TRAINING MODELS ===")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                    random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                           random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), 
                                          max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='f1_weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Detailed model evaluation
        """
        print("\n=== MODEL EVALUATION ===")
        
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Visualize Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.close()
        
        return accuracy
    
    def compare_models(self, results):
        """
        Compare all trained models
        """
        print("\n=== MODEL COMPARISON ===")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'F1-Score': [results[m]['f1_score'] for m in results.keys()],
            'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
            'CV Std': [results[m]['cv_std'] for m in results.keys()]
        })
        
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0].set_xlim([0.5, 1.0])
        
        # F1-Score comparison
        axes[1].barh(comparison_df['Model'], comparison_df['F1-Score'], color='coral')
        axes[1].set_xlabel('F1-Score')
        axes[1].set_title('Model F1-Score Comparison', fontweight='bold')
        axes[1].set_xlim([0.5, 1.0])
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nModel comparison saved as 'model_comparison.png'")
        plt.close()
        
        # Best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = results[best_model_name]['model']
        print(f"\nBest Model: {best_model_name}")
        
        return best_model, best_model_name
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Hyperparameter tuning for Random Forest
        """
        print("\n=== HYPERPARAMETER OPTIMIZATION ===")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            verbose=1,
            n_jobs=-1
        )
        
        print("Running GridSearchCV... (this may take a while)")
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def feature_importance(self, model, feature_names):
        """
        Analyze feature importance (for tree-based models)
        """
        if hasattr(model, 'feature_importances_'):
            print("\n=== FEATURE IMPORTANCE ===")
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(importance_df.head(10))
            
            # Visualize
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'].head(10), 
                    importance_df['Importance'].head(10))
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance saved as 'feature_importance.png'")
            plt.close()
            
            return importance_df
        else:
            print("Feature importance not available for this model")
            return None
    
    def save_model(self, model, filename='accident_model.pkl'):
        """
        Save trained model
        """
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved as '{filename}'")
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Scaler saved as 'scaler.pkl'")
    
    def load_model(self, filename='accident_model.pkl'):
        """
        Load trained model
        """
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from '{filename}'")
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded from 'scaler.pkl'")
        
        return self.model
    
    def predict(self, features):
        """
        Make prediction on new data
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)
        
        return {
            'severity': prediction[0],
            'probability': probability[0]
        }


def generate_sample_data(n_samples=10000):
    """
    Generate sample accident data for demonstration
    """
    np.random.seed(42)
    
    data = {
        'Date': pd.date_range('2022-01-01', periods=n_samples, freq='h'),
        'Time': [f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:00" 
                 for _ in range(n_samples)],
        'Weather_Conditions': np.random.choice(['Fine', 'Raining', 'Snowing', 'Fog'], 
                                               n_samples, p=[0.6, 0.25, 0.1, 0.05]),
        'Road_Type': np.random.choice(['Motorway', 'A-Road', 'B-Road', 'Residential'], 
                                     n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'Light_Conditions': np.random.choice(['Daylight', 'Darkness', 'Dusk'], 
                                            n_samples, p=[0.6, 0.3, 0.1]),
        'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet', 'Ice'], 
                                                    n_samples, p=[0.6, 0.3, 0.1]),
        'Urban_or_Rural_Area': np.random.choice(['Urban', 'Rural'], 
                                               n_samples, p=[0.7, 0.3]),
        'Speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], n_samples),
        'Number_of_Vehicles': np.random.randint(1, 5, n_samples),
        'Number_of_Casualties': np.random.randint(1, 4, n_samples),
        'Number_of_Lanes': np.random.randint(1, 4, n_samples)
    }
    
    # Generate target based on features (simplified logic)
    severity = []
    for i in range(n_samples):
        score = 0
        
        # Weather impact
        if data['Weather_Conditions'][i] in ['Snowing', 'Fog']:
            score += 2
        elif data['Weather_Conditions'][i] == 'Raining':
            score += 1
        
        # Speed impact
        score += data['Speed_limit'][i] / 35
        
        # Casualties impact
        score += data['Number_of_Casualties'][i] * 0.5
        
        # Light conditions
        if data['Light_Conditions'][i] == 'Darkness':
            score += 1
        
        # Random factor
        score += np.random.normal(0, 1)
        
        # Classify
        if score < 2:
            severity.append(3)  # Slight
        elif score < 4:
            severity.append(2)  # Serious
        else:
            severity.append(1)  # Fatal
    
    data['Accident_Severity'] = severity
    
    df = pd.DataFrame(data)
    return df


def main():
    """
    Main execution pipeline
    """
    print("=" * 70)
    print("MACHINE LEARNING BASED ROAD ACCIDENT SEVERITY PREDICTION SYSTEM")
    print("=" * 70)
    
    # Initialize predictor
    predictor = AccidentSeverityPredictor()
    
    # Generate sample data (replace with actual data loading)
    print("\nGenerating sample data...")
    df = generate_sample_data(n_samples=10000)
    df.to_csv('sample_accident_data.csv', index=False)
    print("Sample data saved as 'sample_accident_data.csv'")
    
    # Explore data
    df = predictor.explore_data(df)
    
    # Visualize
    predictor.visualize_data(df)
    
    # Clean data
    df = predictor.clean_data(df)
    
    # Feature engineering
    df = predictor.feature_engineering(df)
    
    # Encode features
    df = predictor.encode_features(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Split and scale
    X_train, X_test, y_train, y_test = predictor.split_and_scale(X, y)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = predictor.handle_imbalance(X_train, y_train)
    
    # Train models
    results = predictor.train_models(X_train_balanced, y_train_balanced, 
                                    X_test, y_test)
    
    # Compare models
    best_model, best_model_name = predictor.compare_models(results)
    
    # Evaluate best model
    predictor.evaluate_model(best_model, X_test, y_test)
    
    # Feature importance
    predictor.feature_importance(best_model, predictor.feature_columns)
    
    # Optimize (optional - commented out as it takes time)
    # optimized_model = predictor.optimize_hyperparameters(X_train_balanced, y_train_balanced)
    # predictor.evaluate_model(optimized_model, X_test, y_test)
    
    # Save model
    predictor.model = best_model
    predictor.save_model(best_model)
    
    # Example prediction
    print("\n=== EXAMPLE PREDICTION ===")
    sample_features = X_test[0]
    result = predictor.predict(sample_features)
    print(f"Predicted Severity: {result['severity']}")
    print(f"Probabilities: {result['probability']}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
