#!/usr/bin/env python3
"""
Domain-Specific Feature Engineering for Cardiovascular Risk Prediction
Adds medical knowledge-based features and interactions to improve model performance.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def add_domain_features(df):
    """
    Add domain-specific features based on cardiovascular medical knowledge.
    
    Args:
        df: DataFrame with original features
        
    Returns:
        DataFrame with additional engineered features
    """
    df_eng = df.copy()
    
    print("🔧 Adding domain-specific features...")
    
    # 1. Pulse Pressure (arterial stiffness indicator)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df_eng['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        print("   ✓ Pulse Pressure = ap_hi - ap_lo")
    
    # 2. Mean Arterial Pressure (cardiac output indicator)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df_eng['MAP'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
        print("   ✓ Mean Arterial Pressure = ap_lo + (ap_hi - ap_lo)/3")
    
    # 3. BMI categories (if BMI exists)
    if 'bmi' in df.columns:
        bmi_cat = pd.cut(df['bmi'], 
                        bins=[0, 18.5, 25, 30, 40, 100], 
                        labels=[0, 1, 2, 3, 4], 
                        include_lowest=True)
        df_eng['bmi_category'] = bmi_cat.cat.codes  # Convert to numeric codes
        print("   ✓ BMI Categories (0=Underweight, 1=Normal, 2=Overweight, 3=Obese, 4=Morbid)")
    
    # 4. Age groups (clinical relevance)
    if 'age' in df.columns:
        age_cat = pd.cut(df['age'], 
                        bins=[0, 35, 45, 55, 65, 100], 
                        labels=[0, 1, 2, 3, 4], 
                        include_lowest=True)
        df_eng['age_group'] = age_cat.cat.codes  # Convert to numeric codes
        print("   ✓ Age Groups (0=<35, 1=35-44, 2=45-54, 3=55-64, 4=65+)")
    
    # 5. Blood pressure categories
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        # Normal: <120/80, Elevated: 120-129/<80, Stage1: 130-139/80-89, Stage2: ≥140/≥90
        bp_stage = np.zeros(len(df))
        bp_stage[(df['ap_hi'] >= 130) | (df['ap_lo'] >= 80)] = 1  # Stage 1
        bp_stage[(df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)] = 2  # Stage 2
        bp_stage[(df['ap_hi'] >= 160) | (df['ap_lo'] >= 100)] = 3  # Stage 3
        df_eng['bp_stage'] = bp_stage
        print("   ✓ Blood Pressure Stages (0=Normal, 1=Stage1, 2=Stage2, 3=Stage3)")
    
    # 6. Feature Interactions (clinically meaningful combinations)
    print("   🔗 Adding feature interactions...")
    
    # Age × BMI (compounding risk factors)
    if 'age' in df.columns and 'bmi' in df.columns:
        df_eng['age_bmi'] = df['age'] * df['bmi']
        print("     ✓ age × bmi")
    
    # Cholesterol × Glucose (metabolic syndrome indicators)
    if 'cholesterol' in df.columns and 'gluc' in df.columns:
        df_eng['chol_gluc'] = df['cholesterol'] * df['gluc']
        print("     ✓ cholesterol × gluc")
    
    # Blood pressure interaction
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df_eng['bp_interaction'] = df['ap_hi'] * df['ap_lo']
        print("     ✓ ap_hi × ap_lo")
    
    # Age × Blood pressure (age-related hypertension)
    if 'age' in df.columns and 'ap_hi' in df.columns:
        df_eng['age_bp'] = df['age'] * df['ap_hi']
        print("     ✓ age × ap_hi")
    
    # BMI × Blood pressure (obesity-hypertension link)
    if 'bmi' in df.columns and 'ap_hi' in df.columns:
        df_eng['bmi_bp'] = df['bmi'] * df['ap_hi']
        print("     ✓ bmi × ap_hi")
    
    # 7. Risk score components (clinical guidelines)
    if all(col in df.columns for col in ['age', 'gender', 'smoke', 'cholesterol', 'ap_hi']):
        # Simplified risk score based on major factors
        risk_score = np.zeros(len(df))
        risk_score += (df['age'] - 30) * 0.1  # Age factor
        risk_score += (df['gender'] - 1) * 2  # Male factor
        risk_score += df['smoke'] * 5  # Smoking factor
        risk_score += (df['cholesterol'] - 1) * 2  # Cholesterol factor
        risk_score += (df['ap_hi'] - 120) * 0.05  # BP factor
        df_eng['clinical_risk_score'] = risk_score
        print("   ✓ Clinical Risk Score (age + gender + smoking + cholesterol + BP)")
    
    # 8. Lifestyle risk index
    if all(col in df.columns for col in ['smoke', 'alco', 'active']):
        lifestyle_risk = df['smoke'] * 3 + df['alco'] * 1 - df['active'] * 2
        df_eng['lifestyle_risk'] = lifestyle_risk
        print("   ✓ Lifestyle Risk Index (smoking + alcohol - activity)")
    
    print(f"✅ Feature engineering complete! Added {len(df_eng.columns) - len(df.columns)} new features")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Total features: {len(df_eng.columns)}")
    
    return df_eng

def main():
    parser = argparse.ArgumentParser(description="Add domain-specific features to cardiovascular dataset")
    parser.add_argument('--input', type=str, default='cardio_train_clean.csv', 
                       help='Input CSV file')
    parser.add_argument('--output', type=str, default='cardio_engineered.csv',
                       help='Output CSV file with engineered features')
    parser.add_argument('--show-features', action='store_true',
                       help='Show all feature names and types')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📊 Loading dataset: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Shape: {df.shape}")
    
    # Add engineered features
    df_engineered = add_domain_features(df)
    
    # Show feature information
    if args.show_features:
        print("\n📋 All Features:")
        for i, col in enumerate(df_engineered.columns, 1):
            dtype = df_engineered[col].dtype
            unique_vals = df_engineered[col].nunique()
            print(f"   {i:2d}. {col:<25} ({dtype}, {unique_vals} unique values)")
    
    # Save engineered dataset
    print(f"\n💾 Saving engineered dataset: {args.output}")
    df_engineered.to_csv(args.output, index=False)
    
    # Show summary statistics for new features
    new_features = [col for col in df_engineered.columns if col not in df.columns]
    if new_features:
        print(f"\n📈 New Feature Summary:")
        print(df_engineered[new_features].describe().round(3))
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Ready for GA feature selection or direct model training")

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
# The shipped training CSVs (cardio_train_clean.csv, cardio_engineered.csv) hold
# the continuous columns already standardised (z-scores). The saved models were
# trained on that z-score space, so raw user input (age in years, weight in kg,
# BP in mmHg ...) must be converted to the same space before it reaches the
# models. Values are (mean, std) recovered from the training data; age, ap_hi
# and ap_lo are exact, height, weight and BMI are close estimates.
RAW_TO_Z_STATS = {
    'age': (52.8286, 6.7688),
    'height': (164.35, 8.12),
    'weight': (74.10, 12.82),
    'ap_hi': (126.6786, 16.6822),
    'ap_lo': (81.311, 9.4282),
    'BMI': (27.40, 5.10),
}


def standardize_raw_inputs(user_data):
    """Return a one-row DataFrame in the model's training (z-score) space."""
    row = {
        'age': user_data['age'], 'gender': user_data['gender'],
        'height': user_data['height'], 'weight': user_data['weight'],
        'ap_hi': user_data['ap_hi'], 'ap_lo': user_data['ap_lo'],
        'cholesterol': user_data['cholesterol'], 'gluc': user_data['gluc'],
        'smoke': user_data['smoke'], 'alco': user_data['alco'],
        'active': user_data['active'], 'BMI': user_data['bmi'],
    }
    for col, (mean, std) in RAW_TO_Z_STATS.items():
        row[col] = (float(row[col]) - mean) / std
    return pd.DataFrame([row])


def engineer_single_sample(user_data):
    """Training-identical 21 features for one person, from raw user input."""
    import io, contextlib
    df = standardize_raw_inputs(user_data)
    with contextlib.redirect_stdout(io.StringIO()):
        df = add_domain_features(df)
    cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
            'gluc', 'smoke', 'alco', 'active', 'BMI', 'pulse_pressure', 'MAP',
            'age_group', 'bp_stage', 'chol_gluc', 'bp_interaction', 'age_bp',
            'clinical_risk_score', 'lifestyle_risk']
    return df[cols].astype(float).values
