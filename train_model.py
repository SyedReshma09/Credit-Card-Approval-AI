import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CreditCardTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self):
        """Load and merge datasets"""
        print("Loading datasets...")
        
        # Load application records
        app_df = pd.read_csv('dataset/application_record.csv')
        print(f"Application records: {app_df.shape}")
        
        # Load credit records
        credit_df = pd.read_csv('dataset/credit_record.csv')
        print(f"Credit records: {credit_df.shape}")
        
        return app_df, credit_df
    
    def preprocess_data(self, app_df, credit_df):
        """Preprocess and merge data"""
        print("Preprocessing data...")
        
        # Create target variable: Good credit status
        # Consider status '0' as good (no default), others as bad
        credit_df['TARGET'] = credit_df['STATUS'].apply(lambda x: 1 if x == '0' else 0)
        
        # Aggregate credit records by ID
        credit_agg = credit_df.groupby('ID').agg({
            'TARGET': 'mean',  # Proportion of good status
            'STATUS': lambda x: len(x)  # Number of records
        }).rename(columns={'TARGET': 'credit_score', 'STATUS': 'record_count'})
        
        # Merge with application data
        merged_df = app_df.merge(credit_agg, on='ID', how='inner')
        print(f"Merged dataset: {merged_df.shape}")
        
        return merged_df
    
    def feature_engineering(self, df):
        """Create and select features"""
        print("Engineering features...")
        
        # Convert days to years
        df['AGE'] = -df['DAYS_BIRTH'] / 365
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        
        # Income per family member
        df['INCOME_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        
        # Children ratio
        df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        
        # Select features for modeling
        features = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
            'AGE', 'EMPLOYMENT_YEARS', 'INCOME_PER_FAMILY', 'CHILDREN_RATIO'
        ]
        
        self.feature_columns = features
        return df[features + ['credit_score']]
    
    def prepare_features(self, df):
        """Prepare features for training"""
        print("Preparing features...")
        
        X = df.drop('credit_score', axis=1)
        # Convert continuous credit_score to binary target (1 for Approved, 0 for Not Approved)
        y = (df['credit_score'] >= 0.5).astype(int)
        
        # Handle missing values
        X = X.fillna('Unknown')
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        return X_scaled, y
    
    def train_model(self):
        """Train the model"""
        print("Starting model training...")
        
        # Load and prepare data
        app_df, credit_df = self.load_data()
        merged_df = self.preprocess_data(app_df, credit_df)
        processed_df = self.feature_engineering(merged_df)
        
        # Prepare features
        X, y = self.prepare_features(processed_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return accuracy
    
    def save_model(self):
        """Save model and preprocessing objects"""
        print("\nSaving model...")
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, 'model/credit_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'model/scaler.pkl')
        
        # Save label encoders
        joblib.dump(self.label_encoders, 'model/label_encoders.pkl')
        
        # Save feature columns
        joblib.dump(self.feature_columns, 'model/feature_columns.pkl')
        
        print("Model saved successfully!")
    
    def run(self):
        """Run the complete training pipeline"""
        try:
            accuracy = self.train_model()
            self.save_model()
            print(f"\n[SUCCESS] Training completed successfully!")
            print(f"Model accuracy: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            print(f"\n[ERROR] Error during training: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = CreditCardTrainer()
    trainer.run()