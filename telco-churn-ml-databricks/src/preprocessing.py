

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class TelcoPreprocessor:
    """
    Comprehensive preprocessing pipeline for Telco Churn dataset.
    
    Attributes:
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (dict): Dictionary of label encoders for categorical features
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_features = []
        self.categorical_features = []
        self.target_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """
        Load the Telco Customer Churn dataset.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset: handle missing values, convert data types.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (handles whitespace)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Handle missing values in TotalCharges
        missing_count = df_clean['TotalCharges'].isnull().sum()
        if missing_count > 0:
            print(f"⚠ Found {missing_count} missing values in TotalCharges")
            # Fill with median or drop (depends on business logic)
            df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
            print(f"✓ Missing values filled with median")
        
        # Remove customerID (not a feature)
        if 'customerID' in df_clean.columns:
            df_clean.drop('customerID', axis=1, inplace=True)
            print(" Removed customerID column")
        
        print(f" Data cleaned: {df_clean.shape}")
        return df_clean
    
    def identify_feature_types(self, df, target_col='Churn'):
        """
        Identify numerical and categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
        """
        # Identify numerical features
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_col in self.numerical_features:
            self.numerical_features.remove(target_col)
        
        # Identify categorical features
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        if target_col in self.categorical_features:
            self.categorical_features.remove(target_col)
        
        print(f"✓ Numerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"✓ Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        print(f" Encoded {len(self.categorical_features)} categorical features")
        return df_encoded
    
    def encode_target(self, y):
        """
        Encode target variable (Churn: Yes/No -> 1/0).
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            np.array: Encoded target
        """
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"✓ Target encoded: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        return y_encoded
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Scaled training and test sets (if provided)
        """
        X_train_scaled = X_train.copy()
        
        # Fit and transform training data
        X_train_scaled[self.numerical_features] = self.scaler.fit_transform(
            X_train[self.numerical_features]
        )
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[self.numerical_features] = self.scaler.transform(
                X_test[self.numerical_features]
            )
            print(f"✓ Features scaled (train and test)")
            return X_train_scaled, X_test_scaled
        
        print(f"✓ Features scaled (train only)")
        return X_train_scaled
    
    def prepare_data_for_clustering(self, df, target_col='Churn'):
        """
        Prepare data specifically for clustering (without target variable).
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column to exclude
            
        Returns:
            tuple: (X_scaled, y)
        """
        df_clean = self.clean_data(df)
        self.identify_feature_types(df_clean, target_col)
        
        # Separate features and target
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X)
        
        # Scale all features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Encode target for evaluation
        y_encoded = self.encode_target(y)
        
        print(f" Data prepared for clustering: {X_scaled.shape}")
        return X_scaled, y_encoded
    
    def prepare_train_test_split(self, df, target_col='Churn', test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline with train-test split.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Clean data
        df_clean = self.clean_data(df)
        
        # Identify feature types
        self.identify_feature_types(df_clean, target_col)
        
        # Separate features and target
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X)
        
        # Encode target
        y_encoded = self.encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f" Train-test split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def get_feature_names(preprocessor):
    """
    Get all feature names after preprocessing.
    
    Args:
        preprocessor (TelcoPreprocessor): Fitted preprocessor
        
    Returns:
        list: List of feature names
    """
    return preprocessor.numerical_features + preprocessor.categorical_features


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Telco Churn Preprocessing Module")
    print("="*60)
    
    preprocessor = TelcoPreprocessor()
    df = preprocessor.load_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(df)
    
    print("\n" + "="*60)
    print(f"Final shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print("="*60)
