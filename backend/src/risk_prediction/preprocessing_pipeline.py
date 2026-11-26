#!/usr/bin/env python3
"""
Preprocessing Pipeline for Genetic Algorithm Feature Selection
Follows best practices to ensure data quality without harming GA effectiveness
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class GAPreprocessingPipeline:
    def __init__(self, handle_missing='simple', scale_features=False, encode_categorical=True, feature_engineering=True, remove_correlated=False, random_state=42):
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.encode_categorical = encode_categorical
        self.feature_engineering = feature_engineering
        self.remove_correlated = remove_correlated
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, df, target_col='cardio'):
        """
        Preprocess data for GA feature selection
        
        Args:
            df: Input DataFrame with cardiovascular features
            target_col: Name of target column
            
        Returns:
            X: Preprocessed feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        print("🔧 Starting GA-optimized preprocessing pipeline...")
        
        # Step 1: Data validation
        print("   - Validating data structure...")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Step 2: Handle missing values
        print("   - Handling missing values...")
        df_clean = self._handle_missing_values(df)
        
        # Step 3: Encode categorical variables
        if self.encode_categorical:
            print("   - Encoding categorical variables...")
            df_clean = self._encode_categorical(df_clean)
        
        # Step 4: Feature engineering (before GA)
        if self.feature_engineering:
            print("   - Engineering cardiovascular features...")
            df_clean = self._engineer_features(df_clean)
        
        # Step 5: Prepare final dataset
        print("   - Preparing final dataset...")
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
        
        # Store feature names for GA
        self.feature_names = X.columns.tolist()
        
        # Step 6: Scale features (optional, only if needed)
        if self.scale_features:
            print("   - Scaling features...")
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Step 7: Final validation
        print("   - Final validation...")
        if X.isnull().any().any():
            raise ValueError("NaN values detected after preprocessing")
        
        self.is_fitted = True
        print(f"✅ Preprocessing completed! Final shape: {X.shape}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Samples: {len(X)}")
        
        return X.values, y.values, self.feature_names

    def _handle_missing_values(self, df):
        """Handle missing values using appropriate strategy"""
        df_clean = df.copy()
        
        # Check for missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"     Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            if self.handle_missing == 'knn':
                # Use KNN imputation for better quality
                self.imputer = KNNImputer(n_neighbors=5, random_state=self.random_state)
                df_clean = pd.DataFrame(
                    self.imputer.fit_transform(df_clean),
                    columns=df_clean.columns,
                    index=df_clean.index
                )
                print("     Applied KNN imputation")
            else:
                # Use simple imputation (median for numerical, mode for categorical)
                for col in df_clean.columns:
                    if df_clean[col].dtype in ['object', 'category']:
                        # Categorical: use mode
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                        df_clean[col].fillna(mode_val, inplace=True)
                    else:
                        # Numerical: use median
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                print("     Applied simple imputation (median/mode)")
        else:
            print("     No missing values found")
        
        return df_clean

    def _encode_categorical(self, df):
        """Encode categorical variables for GA compatibility"""
        df_encoded = df.copy()
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype in ['object', 'category']:
                # Create label encoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"     Encoded '{col}' -> {len(le.classes_)} categories")
        
        return df_encoded

    def _engineer_features(self, df):
        """Engineer cardiovascular-specific features"""
        df_engineered = df.copy()
        
        # BMI calculation (if height and weight exist)
        if 'height' in df_engineered.columns and 'weight' in df_engineered.columns:
            # Check if BMI already exists
            if 'BMI' not in df_engineered.columns:
                # Convert height to meters and weight to kg if needed
                height_col = 'height'
                weight_col = 'weight'
                
                # Check if data is already scaled (small range values)
                height_range = df_engineered[height_col].max() - df_engineered[height_col].min()
                weight_range = df_engineered[weight_col].max() - df_engineered[weight_col].min()
                
                if height_range < 100:  # Likely already in meters
                    height_m = df_engineered[height_col]
                else:  # Likely in cm, convert to meters
                    height_m = df_engineered[height_col] / 100
                
                if weight_range < 100:  # Likely already in kg
                    weight_kg = df_engineered[weight_col]
                else:  # Likely in grams, convert to kg
                    weight_kg = df_engineered[weight_col] / 1000
                
                # Calculate BMI
                df_engineered['BMI'] = weight_kg / (height_m ** 2)
                print("     - Added BMI feature")
            else:
                print("     - BMI feature already exists, skipping")
        
        # Age groups (if age exists)
        if 'age' in df_engineered.columns:
            # Check if age is already scaled
            age_range = df_engineered['age'].max() - df_engineered['age'].min()
            if age_range > 50:  # Likely actual age values
                # Fill NaN values before binning
                age_median = df_engineered['age'].median()
                df_engineered['age'] = df_engineered['age'].fillna(age_median)
                
                df_engineered['age_group'] = pd.cut(
                    df_engineered['age'], 
                    bins=[0, 30, 50, 70, 100], 
                    labels=[0, 1, 2, 3]
                )
                df_engineered['age_group'] = df_engineered['age_group'].astype(int)
                print("     - Added age_group feature")
            else:
                print("     - Age appears to be scaled, skipping age_group creation")
        
        # Blood pressure categories (if systolic and diastolic exist)
        if 'ap_hi' in df_engineered.columns and 'ap_lo' in df_engineered.columns:
            # Check if BP values are already scaled
            ap_hi_range = df_engineered['ap_hi'].max() - df_engineered['ap_hi'].min()
            ap_lo_range = df_engineered['ap_lo'].max() - df_engineered['ap_lo'].min()
            
            if ap_hi_range > 50 and ap_lo_range > 30:  # Likely actual BP values
                # Fill NaN values before categorization
                ap_hi_median = df_engineered['ap_hi'].median()
                ap_lo_median = df_engineered['ap_lo'].median()
                df_engineered['ap_hi'] = df_engineered['ap_hi'].fillna(ap_hi_median)
                df_engineered['ap_lo'] = df_engineered['ap_lo'].fillna(ap_lo_median)
                
                # Create BP categories
                conditions = [
                    (df_engineered['ap_hi'] < 120) & (df_engineered['ap_lo'] < 80),
                    (df_engineered['ap_hi'] >= 120) & (df_engineered['ap_hi'] < 130) & (df_engineered['ap_lo'] < 80),
                    (df_engineered['ap_hi'] >= 130) | (df_engineered['ap_lo'] >= 80)
                ]
                choices = [0, 1, 2]
                df_engineered['bp_category'] = np.select(conditions, choices, default=2)
                print("     - Added bp_category feature")
            else:
                print("     - BP values appear to be scaled, skipping bp_category creation")
        
        return df_engineered

    def transform(self, df):
        """Transform new data using fitted preprocessing"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        
        # Apply the same preprocessing steps
        df_clean = self._handle_missing_values(df)
        
        if self.encode_categorical:
            df_clean = self._encode_categorical(df_clean)
        
        if self.feature_engineering:
            df_clean = self._engineer_features(df_clean)
        
        # Ensure same features as training
        missing_features = set(self.feature_names) - set(df_clean.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the features used in training
        X = df_clean[self.feature_names]
        
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X.values

def create_ga_preprocessing_pipeline(handle_missing='simple', scale_features=False, feature_engineering=True):
    """Create a preprocessing pipeline optimized for GA feature selection"""
    return GAPreprocessingPipeline(
        handle_missing=handle_missing,
        scale_features=scale_features,
        encode_categorical=True,
        feature_engineering=feature_engineering,
        remove_correlated=False,
        random_state=42
    )

if __name__ == "__main__":
    print("GA Preprocessing Pipeline for Cardiovascular Disease")
    print("This module provides preprocessing optimized for GA feature selection")
    print("Features:")
    print("- Missing value handling (simple/KNN)")
    print("- Categorical encoding")
    print("- Cardiovascular feature engineering")
    print("- GA-compatible data preparation")
