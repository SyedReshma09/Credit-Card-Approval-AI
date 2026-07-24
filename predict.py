import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

class CreditCardPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        try:
            model_path = 'model/credit_model.pkl'
            scaler_path = 'model/scaler.pkl'
            encoders_path = 'model/label_encoders.pkl'
            features_path = 'model/feature_columns.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoders = joblib.load(encoders_path)
                self.feature_columns = joblib.load(features_path)
                print("[INFO] Model loaded successfully!")
                return True
            else:
                print("[WARNING] Model not found. Please train the model first.")
                return False
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, form_data):
        """Preprocess single input for prediction"""
        # Create DataFrame with a single row
        df = pd.DataFrame([form_data])
        
        # Calculate derived features
        df['AGE'] = -df['DAYS_BIRTH'] / 365
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['INCOME_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        
        # Select features in correct order
        df = df[self.feature_columns]
        
        # Fill missing values
        df = df.fillna('Unknown')
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown categories
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                if 'Unknown' not in le.classes_:
                    # Add 'Unknown' to classes if not present
                    classes = list(le.classes_) + ['Unknown']
                    le.classes_ = np.array(classes)
                df[col] = le.transform(df[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def predict(self, form_data):
        """Make prediction for a single application"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        try:
            # Preprocess input
            X = self.preprocess_input(form_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get probability of positive class (credit approval)
            prob_positive = probability[1] if len(probability) > 1 else probability[0]
            
            # Determine approval status
            if prediction == 1:
                status = "Approved"
                confidence = prob_positive * 100
            else:
                status = "Not Approved"
                confidence = (1 - prob_positive) * 100
            
            return status, f"{confidence:.1f}%"
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

# Example usage
if __name__ == "__main__":
    predictor = CreditCardPredictor()
    
    # Sample input
    sample_data = {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 1,
        'AMT_INCOME_TOTAL': 250000,
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'DAYS_BIRTH': -35*365,
        'DAYS_EMPLOYED': -5*365,
        'FLAG_WORK_PHONE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': 'Office staff',
        'CNT_FAM_MEMBERS': 3
    }
    
    try:
        status, confidence = predictor.predict(sample_data)
        print(f"\nPrediction: {status}")
        print(f"Confidence: {confidence}")
    except Exception as e:
        print(f"Error: {e}")