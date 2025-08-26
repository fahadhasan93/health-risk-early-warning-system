import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

class HealthRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names = None
        self.explainer = None
        
    def load_data(self, file_path):
        """Load and preprocess the health risk dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded: {self.data.shape[0]} patients, {self.data.shape[1]} features")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Handle categorical variables - convert to numeric for XGBoost compatibility
        df['Consciousness'] = df['Consciousness'].astype('category')
        df['On_Oxygen'] = df['On_Oxygen'].astype(int)  # Convert to int instead of category
        
        # One-hot encode consciousness
        consciousness_dummies = pd.get_dummies(df['Consciousness'], prefix='Consciousness')
        df = pd.concat([df, consciousness_dummies], axis=1)
        df.drop('Consciousness', axis=1, inplace=True)
        
        # Encode target variable
        df['Risk_Level_Encoded'] = self.label_encoder.fit_transform(df['Risk_Level'])
        
        # Select features for modeling
        feature_columns = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 
                          'Systolic_BP', 'Heart_Rate', 'Temperature', 'On_Oxygen'] + \
                         [col for col in df.columns if col.startswith('Consciousness_')]
        
        self.feature_names = feature_columns
        X = df[feature_columns]
        y = df['Risk_Level_Encoded']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        
        print(f"Data preprocessed. Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Feature names: {self.feature_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("Training models...")
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1 Score: {results[best_model_name]['f1']:.4f}")
        
        return results
    
    def evaluate_best_model(self):
        """Evaluate the best performing model in detail"""
        if self.best_model is None:
            print("No best model selected. Please train models first.")
            return
        
        print("\nDetailed evaluation of best model:")
        
        # Make predictions
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            y_pred = self.best_model.predict(self.X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(self.X_test_scaled) if hasattr(self.best_model, 'predict_proba') else None
        else:
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        return y_pred, y_pred_proba
    
    def create_shap_explainer(self):
        """Create SHAP explainer for model interpretability"""
        print("Creating SHAP explainer...")
        
        if self.best_model is None:
            print("No best model available for SHAP explanation.")
            return
        
        # Use training data for SHAP explainer
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            background_data = self.X_train_scaled[:100]  # Use subset for background
            self.explainer = shap.KernelExplainer(self.best_model.predict_proba, background_data)
        else:
            background_data = self.X_train[:100]  # Use subset for background
            self.explainer = shap.TreeExplainer(self.best_model)
        
        print("SHAP explainer created successfully.")
        return self.explainer
    
    def predict_risk(self, patient_data):
        """Predict risk level for new patient data"""
        if self.best_model is None:
            print("No trained model available.")
            return None
        
        # Preprocess patient data
        processed_data = self.preprocess_patient_data(patient_data)
        
        # Make prediction
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            risk_proba = self.best_model.predict_proba(processed_data.reshape(1, -1))[0]
        else:
            risk_proba = self.best_model.predict_proba(processed_data.reshape(1, -1))[0]
        
        risk_level = self.best_model.predict(processed_data.reshape(1, -1))[0]
        risk_level_name = self.label_encoder.inverse_transform([risk_level])[0]
        
        # Create risk breakdown
        risk_breakdown = {
            'predicted_risk': risk_level_name,
            'probabilities': dict(zip(self.label_encoder.classes_, risk_proba)),
            'escalation_probability': self.calculate_escalation_probability(risk_proba)
        }
        
        return risk_breakdown
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess single patient data for prediction"""
        # Extract features in the same order as training
        features = []
        
        # Continuous features
        features.extend([
            float(patient_data['Respiratory_Rate']),
            float(patient_data['Oxygen_Saturation']),
            float(patient_data['O2_Scale']),
            float(patient_data['Systolic_BP']),
            float(patient_data['Heart_Rate']),
            float(patient_data['Temperature']),
            int(patient_data['On_Oxygen'])  # Ensure it's an integer
        ])
        
        # Consciousness one-hot encoding
        consciousness = patient_data['Consciousness']
        for level in ['A', 'P', 'C', 'V', 'U']:
            features.append(1 if consciousness == level else 0)
        
        features = np.array(features, dtype=float)
        
        # Scale if using scaled models
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        return features
    
    def calculate_escalation_probability(self, risk_proba):
        """Calculate probability of escalation from current risk level"""
        # This is a simplified calculation - in practice, you might want more sophisticated logic
        normal_prob = risk_proba[0] if len(risk_proba) > 0 else 0
        low_prob = risk_proba[1] if len(risk_proba) > 1 else 0
        medium_prob = risk_proba[2] if len(risk_proba) > 2 else 0
        high_prob = risk_proba[3] if len(risk_proba) > 3 else 0
        
        # Probability of being at medium or high risk
        escalation_prob = medium_prob + high_prob
        
        return {
            'normal_to_high': normal_prob * high_prob,
            'low_to_high': low_prob * high_prob,
            'medium_to_high': medium_prob * high_prob,
            'overall_escalation': escalation_prob
        }
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else:
            return None
        
        feature_importance = dict(zip(self.feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a previously trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")

def main():
    """Main function to demonstrate the model training and evaluation"""
    # Initialize the predictor
    predictor = HealthRiskPredictor()
    
    # Load data
    data = predictor.load_data('Health_Risk_Dataset.csv')
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train models
    results = predictor.train_models()
    
    # Evaluate best model
    predictor.evaluate_best_model()
    
    # Create SHAP explainer
    predictor.create_shap_explainer()
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\nFeature Importance:")
        for feature, imp in list(importance.items())[:10]:
            print(f"{feature}: {imp:.4f}")
    
    # Save the model
    predictor.save_model('hrews_model.pkl')
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main()
