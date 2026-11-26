#!/usr/bin/env python3
"""
Cardiovascular Risk Calculator
Input: Person's health data
Output: Risk Score (0-100%)
"""

import os
import pandas as pd
import numpy as np
import argparse
import sys

# Ensure Windows consoles don't crash on unicode/emoji
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
import warnings
warnings.filterwarnings('ignore')

# Default decision threshold for positive class
DEFAULT_THRESHOLD = 0.369

# Import our models (optional)
try:
    from cardiovascular_transformer import CardiovascularTransformer
except Exception:
    CardiovascularTransformer = None
try:
    from genetic_feature_selection import GeneticFeatureSelector
except Exception:
    GeneticFeatureSelector = None
try:
    from particle_swarm_optimization import ParticleSwarmOptimizer
except Exception:
    ParticleSwarmOptimizer = None
try:
    from genetic_transformer import TransformerClassifier
except Exception:
    TransformerClassifier = None
try:
    import sys
    import os
    # Add current directory to path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from comprehensive_stacking_ensemble import ComprehensiveStackingEnsemble
except Exception as e:
    print(f"⚠️ Could not import ComprehensiveStackingEnsemble: {e}")
    ComprehensiveStackingEnsemble = None

class CardiovascularRiskCalculator:
    """
    Cardiovascular Risk Calculator with Risk Score (0-100%)
    """
    
    def __init__(self):
        self.models = {}
        self.default_threshold = DEFAULT_THRESHOLD
        self.feature_names = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'
        ]
        # Optional stacking ensemble (meta-learner + base models)
        self.stacking_ensemble = None
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models and data"""
        print("🔄 Loading cardiovascular risk models...")
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'cardio_train_clean.csv')
        
        # Load the dataset
        print(f"📊 Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"📈 Dataset loaded: {len(df)} records")
        self.scaler = self._fit_scaler(df)
        
        # Precompute statistical centroids in scaled space for efficiency
        try:
            feature_columns = [col for col in df.columns if col != 'cardio']
            X = df[feature_columns].values
            X_scaled = self.scaler.transform(X)
            y = df['cardio'].values
            self.centroids_ = {}
            if np.any(y == 1):
                self.centroids_['cardio'] = X_scaled[y == 1].mean(axis=0)
            if np.any(y == 0):
                self.centroids_['non_cardio'] = X_scaled[y == 0].mean(axis=0)
        except Exception as e:
            print(f"⚠️ Could not precompute centroids: {e}")
        
        # Initialize models (we'll use pre-trained weights if available)
        try:
            if CardiovascularTransformer is not None:
                self.models['transformer'] = CardiovascularTransformer(
                    num_features=12, d_model=64, nhead=4, num_layers=2
                )
            if GeneticFeatureSelector is not None:
                self.models['genetic'] = GeneticFeatureSelector(
                    population_size=20, generations=10
                )
            if ParticleSwarmOptimizer is not None:
                self.models['pso'] = ParticleSwarmOptimizer(
                    n_particles=10, n_iterations=20
                )
            if TransformerClassifier is not None:
                self.models['classifier'] = TransformerClassifier(
                    input_dim=12, epochs=50, lr=0.001
                )
            if self.models:
                print("✅ Optional models initialized where available.")
            else:
                print("ℹ️ No optional model modules found; proceeding with rule/statistical methods.")
        except Exception as e:
            print(f"⚠️ Model initialization error: {e}. Using simplified risk calculation...")

        # Try to load trained stacking ensemble (meta-learner + base models)
        if ComprehensiveStackingEnsemble is not None:
            try:
                print("🔍 Attempting to load stacking ensemble...")
                stacking = ComprehensiveStackingEnsemble()
                load_result = stacking.load_trained_models()
                print(f"📊 Load result: {load_result}")
                if load_result:
                    self.stacking_ensemble = stacking
                    print("✅ Stacking ensemble loaded successfully and will be used for model-based risk.")
                else:
                    print("ℹ️ Stacking ensemble models not found; continuing without meta-learner.")
            except Exception as e:
                import traceback
                print(f"⚠️ Could not load stacking ensemble: {e}")
                print(f"📋 Full error: {traceback.format_exc()}")
        else:
            print("⚠️ ComprehensiveStackingEnsemble is None")
    
    def _fit_scaler(self, df):
        """Fit scaler on training data"""
        from sklearn.preprocessing import StandardScaler
        feature_columns = [col for col in df.columns if col != 'cardio']
        X = df[feature_columns].values
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler
    
    def calculate_bmi(self, height_cm, weight_kg):
        """Calculate BMI from height and weight"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)
    
    def get_user_input(self):
        """Get cardiovascular health data from user"""
        print("\n" + "="*60)
        print("🏥 CARDIOVASCULAR RISK ASSESSMENT")
        print("="*60)
        print("Please provide the following health information:")
        print()
        
        # Basic information
        age = int(input("📊 Age (years): "))
        gender = int(input("👤 Gender (1 = Female, 2 = Male): "))
        height = float(input("📏 Height (cm): "))
        weight = float(input("⚖️ Weight (kg): "))
        
        # Blood pressure
        ap_hi = int(input("🩸 Systolic Blood Pressure (mmHg): "))
        ap_lo = int(input("🩸 Diastolic Blood Pressure (mmHg): "))
        
        # Health markers
        cholesterol = int(input("🩺 Cholesterol (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): "))
        gluc = int(input("🩺 Glucose (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): "))
        
        # Lifestyle factors
        smoke = int(input("🚬 Smoking (0 = No, 1 = Yes): "))
        alco = int(input("🍺 Alcohol Intake (0 = No, 1 = Yes): "))
        active = int(input("🏃 Physical Activity (0 = No, 1 = Yes): "))
        
        # Calculate BMI
        bmi = self.calculate_bmi(height, weight)
        print(f"📊 Calculated BMI: {bmi}")
        
        return {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'bmi': bmi
        }
    
    def calculate_risk_score(self, user_data, verbose=True):
        """Calculate cardiovascular risk score (0-100%)"""
        if verbose:
            print("\n🔬 Analyzing cardiovascular risk...")
        
        # Ensure BMI and active flags are present/consistent in user_data
        if 'bmi' not in user_data or user_data.get('bmi') is None:
            try:
                h = float(user_data.get('height', 0))
                w = float(user_data.get('weight', 0))
                if h > 0 and w > 0:
                    user_data['bmi'] = self.calculate_bmi(h, w)
                else:
                    user_data['bmi'] = 0.0
            except Exception:
                user_data['bmi'] = 0.0

        if 'activity_hours_per_week' in user_data and 'active' not in user_data:
            try:
                hrs = float(user_data.get('activity_hours_per_week', 0))
            except (TypeError, ValueError):
                hrs = 0.0
            user_data['active'] = 1 if hrs > 0 else 0

        # Map user_data into feature vector using self.feature_names
        feature_values = []
        for name in self.feature_names:
            val = user_data.get(name)
            if name == 'bmi':
                # Already ensured above
                val = user_data.get('bmi', 0.0)
            elif val is None:
                # Fallback for missing values (e.g., active inferred above)
                val = 0.0
            try:
                feature_values.append(float(val))
            except (TypeError, ValueError):
                feature_values.append(0.0)

        features = np.asarray(feature_values, dtype=float).reshape(1, -1)
        # The dataset (and so the scaler/centroids) is in z-score space; convert raw input first
        from feature_engineering import standardize_raw_inputs
        features = standardize_raw_inputs(user_data).values.astype(float).reshape(1, -1)

        # Scale features for statistical / model components
        features_scaled = self.scaler.transform(features)

        # 1. Rule-based risk calculation (uses raw user_data)
        rule_based_risk = float(self._calculate_rule_based_risk(user_data))

        # 2. Statistical risk calculation (distance to centroid)
        statistical_risk = float(self._calculate_statistical_risk(features_scaled))

        # 3. Model-based risk calculation from stacking ensemble (if available)
        model_risk = self._calculate_model_risk(user_data)

        # Combine components: emphasize model risk when available, otherwise
        # use a balanced combination of rule-based and statistical components.
        components = [rule_based_risk, statistical_risk]
        weights = [0.4, 0.6]

        if model_risk is not None:
            components.append(float(model_risk))
            # Give the learned model substantial influence
            weights = [0.3, 0.3, 0.4]

        final_risk = float(np.average(components, weights=weights))

        # Ensure risk is between 0-100%
        final_risk = max(0.0, min(100.0, final_risk))

        if verbose:
            print(f"  Rule-based component: {rule_based_risk:.1f}%")
            print(f"  Statistical component: {statistical_risk:.1f}%")
            if model_risk is not None:
                print(f"  Model-based component (stacking ensemble): {float(model_risk):.1f}%")
            print(f"  → Combined risk score: {final_risk:.1f}%")

        return round(final_risk, 1)
    
    def _calculate_rule_based_risk(self, data):
        """Calculate risk based on medical guidelines"""
        risk = 0.0
        
        # Age factor
        if data['age'] >= 65:
            risk += 25
        elif data['age'] >= 55:
            risk += 20
        elif data['age'] >= 45:
            risk += 15
        elif data['age'] >= 35:
            risk += 10
        
        # Gender factor
        if data['gender'] == 2:  # Male
            risk += 7.242
        elif data['gender'] == 1 and data['age'] >= 45:
            risk += 4.13
        
        # Blood pressure factor
        if data['ap_hi'] >= 180 or data['ap_lo'] >= 110:
            risk += 25
        elif data['ap_hi'] >= 160 or data['ap_lo'] >= 100:
            risk += 20
        elif data['ap_hi'] >= 140 or data['ap_lo'] >= 90:
            risk += 15
        
        # BMI factor
        if data['bmi'] >= 40:
            risk += 20
        elif data['bmi'] >= 35:
            risk += 15
        elif data['bmi'] >= 30:
            risk += 10
        elif data['bmi'] >= 25:
            risk += 5
        
        # Cholesterol factor
        if data['cholesterol'] == 3:
            risk += 15
        elif data['cholesterol'] == 2:
            risk += 10
        
        # Glucose factor
        if data['gluc'] == 3:
            risk += 15
        elif data['gluc'] == 2:
            risk += 10
        
        # Lifestyle factors
        if data['smoke'] == 1:
            risk += 20
        if data['alco'] == 1:
            risk += 10
        if data['active'] == 0:
            risk += 10
        
        return min(100, risk)
    
    def _calculate_statistical_risk(self, features_scaled):
        """Calculate risk using statistical methods (distance to cardio centroid)."""
        try:
            cardio_centroid = self.centroids_.get('cardio') if hasattr(self, 'centroids_') else None
            if cardio_centroid is None:
                return 50
            # Euclidean distance to centroid in scaled space
            diff = features_scaled[0] - cardio_centroid
            dist = float(np.sqrt(np.dot(diff, diff)))
            # Map distance to 0-100 risk; 0 distance -> 100 risk, farther -> lower risk
            risk = (1 - dist / 10.0) * 100.0
            return max(0, min(100, risk))
        except Exception:
            return 50
    
    def _calculate_model_risk(self, user_data):
        """Calculate risk using the trained stacking ensemble (0-100), or None if unavailable.

        This uses the meta-model probability as the core signal so that
        high-risk combinations (e.g., high BP, high cholesterol/glucose,
        smoking, inactivity, high BMI) yield clearly higher risk scores
        than healthy profiles.
        """
        try:
            # Prefer trained stacking ensemble (meta-learner over base models)
            if hasattr(self, 'stacking_ensemble') and self.stacking_ensemble is not None:
                result = self.stacking_ensemble.predict_single_sample(user_data)
                prob = result.get('probability')
                if prob is not None:
                    return max(0.0, min(100.0, float(prob) * 100.0))

            # If no ensemble is available, return None so callers can
            # gracefully fall back to rule-based/statistical components.
            return None
        except Exception as e:
            print(f"⚠️ Model risk calculation error: {e}")
            return None
    
    def get_risk_category(self, risk_score):
        """Get risk category based on score"""
        if risk_score < 20:
            return "🟢 LOW RISK", "Your cardiovascular risk is low. Maintain healthy lifestyle!"
        elif risk_score < 40:
            return "🟡 MODERATE RISK", "Your cardiovascular risk is moderate. Consider lifestyle changes."
        elif risk_score < 60:
            return "🟠 HIGH RISK", "Your cardiovascular risk is high. Consult a healthcare provider."
        else:
            return "🔴 VERY HIGH RISK", "Your cardiovascular risk is very high. Seek immediate medical attention."
    
    def generate_recommendations(self, user_data, risk_score):
        """Generate personalized recommendations"""
        recommendations = []
        
        if user_data['bmi'] >= 25:
            recommendations.append("📉 Consider weight management through diet and exercise")
        
        if user_data['ap_hi'] >= 140 or user_data['ap_lo'] >= 90:
            recommendations.append("🩺 Monitor blood pressure regularly and consider lifestyle changes")
        
        if user_data['smoke'] == 1:
            recommendations.append("🚭 Quit smoking - this is the most important change you can make")
        
        if user_data['active'] == 0:
            recommendations.append("🏃 Start regular physical activity (150 minutes/week)")
        
        if user_data['cholesterol'] >= 2:
            recommendations.append("🥗 Adopt a heart-healthy diet low in saturated fats")
        
        if user_data['gluc'] >= 2:
            recommendations.append("🍎 Monitor blood sugar and consider dietary changes")
        
        if risk_score >= 40:
            recommendations.append("👨‍⚕️ Schedule regular check-ups with your healthcare provider")
        
        return recommendations
    
    def display_results(self, user_data, risk_score, risk_category, recommendations, predicted_label=None, threshold_used=None):
        """Display comprehensive results"""
        print("\n" + "="*60)
        print("📊 CARDIOVASCULAR RISK ASSESSMENT RESULTS")
        print("="*60)
        
        # Risk Score Display
        print(f"\n🎯 RISK SCORE: {risk_score}%")
        print(f"📋 RISK CATEGORY: {risk_category[0]}")
        print(f"💡 {risk_category[1]}")
        if predicted_label is not None:
            print(f"🔎 Predicted Label (threshold {threshold_used if threshold_used is not None else self.default_threshold:.3f}): {'Positive (1)' if predicted_label == 1 else 'Negative (0)'}")
        
        # Health Summary
        print(f"\n📋 HEALTH SUMMARY:")
        print(f"   Age: {user_data['age']} years")
        print(f"   Gender: {'Male' if user_data['gender'] == 2 else 'Female'}")
        print(f"   BMI: {user_data['bmi']} ({'Normal' if user_data['bmi'] < 25 else 'Overweight' if user_data['bmi'] < 30 else 'Obese'})")
        print(f"   Blood Pressure: {user_data['ap_hi']}/{user_data['ap_lo']} mmHg")
        print(f"   Cholesterol: {['Normal', 'Above Normal', 'Well Above Normal'][user_data['cholesterol']-1]}")
        print(f"   Glucose: {['Normal', 'Above Normal', 'Well Above Normal'][user_data['gluc']-1]}")
        print(f"   Smoking: {'Yes' if user_data['smoke'] == 1 else 'No'}")
        print(f"   Physical Activity: {'Yes' if user_data['active'] == 1 else 'No'}")
        
        # Recommendations
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Disclaimer
        print(f"\n⚠️ DISCLAIMER:")
        print("   This assessment is for informational purposes only.")
        print("   It is not a substitute for professional medical advice.")
        print("   Always consult with a healthcare provider for proper diagnosis and treatment.")
        
        print("\n" + "="*60)

def main():
    """Main function to run the cardiovascular risk calculator"""
    parser = argparse.ArgumentParser(description="Cardiovascular Risk Calculator")
    parser.add_argument('--eval', action='store_true', help='Run evaluation on dataset')
    parser.add_argument('--data', type=str, default='cardio_train_clean.csv', help='Path to dataset CSV')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold for positive class (0-1)')
    args = parser.parse_args()
    threshold_value = DEFAULT_THRESHOLD if args.threshold is None else float(args.threshold)
    
    if args.eval:
        # Non-interactive evaluation mode
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
        print("🏥 Evaluation mode: computing metrics on dataset")
        # Initialize calculator (fits scaler, centroids)
        calculator = CardiovascularRiskCalculator()
        # Load dataset
        df = pd.read_csv(args.data)
        feature_columns = [c for c in df.columns if c != 'cardio']
        # Ensure BMI exists; if not, compute from height and weight
        if 'bmi' not in df.columns and {'height', 'weight'}.issubset(df.columns):
            df['bmi'] = (df['weight'] / ((df['height'] / 100.0) ** 2)).round(1)
            if 'bmi' not in feature_columns:
                feature_columns.append('bmi')
        # Prepare predictions
        y_true = df['cardio'].values
        y_probs = []
        for _, row in df.iterrows():
            user_data = {
                'age': float(row['age']),
                'gender': int(row['gender']),
                'height': float(row['height']),
                'weight': float(row['weight']),
                'ap_hi': float(row['ap_hi']),
                'ap_lo': float(row['ap_lo']),
                'cholesterol': int(row['cholesterol']),
                'gluc': int(row['gluc']),
                'smoke': int(row['smoke']) if 'smoke' in row else int(row['smoke']),
                'alco': int(row['alco']) if 'alco' in row else int(row['alco']),
                'active': int(row['active']) if 'active' in row else int(row['active']),
                'bmi': float(row['bmi']) if 'bmi' in row else calculator.calculate_bmi(float(row['height']), float(row['weight']))
            }
            risk = calculator.calculate_risk_score(user_data, verbose=False)
            y_probs.append(risk / 100.0)
        y_probs = np.asarray(y_probs)
        # Helper to compute metrics for a given threshold
        def compute_metrics_at_threshold(thresh):
            yp = (y_probs >= thresh).astype(int)
            acc_v = accuracy_score(y_true, yp)
            precision_v, recall_v, f1_v, _ = precision_recall_fscore_support(y_true, yp, average='binary', zero_division=0)
            cm_v = confusion_matrix(y_true, yp)
            youden_v = recall_v + (cm_v[0,0] / (cm_v[0,0] + cm_v[0,1] if (cm_v[0,0] + cm_v[0,1]) else 1)) - 1.0
            return {
                'threshold': thresh,
                'accuracy': acc_v,
                'precision': precision_v,
                'recall': recall_v,
                'f1': f1_v,
                'youden': youden_v,
                'cm': cm_v
            }
        # Base metrics at 0.5 or user-provided threshold
        base_metrics = compute_metrics_at_threshold(threshold_value)
        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, y_probs)
        except Exception:
            auc = float('nan')
        # Threshold tuning (scan unique probabilities plus grid)
        unique_scores = np.unique(y_probs)
        grid = np.linspace(0.05, 0.95, 19)
        candidates = np.unique(np.concatenate([unique_scores, grid]))
        best_f1 = None
        best_youden = None
        for t in candidates:
            m = compute_metrics_at_threshold(float(t))
            if best_f1 is None or m['f1'] > best_f1['f1']:
                best_f1 = m
            if best_youden is None or m['youden'] > best_youden['youden']:
                best_youden = m
        # Print metrics
        print(f"\n📏 Metrics on {len(df)} records (ROC-AUC: {auc:.4f}):")
        print(f"   At threshold {base_metrics['threshold']:.3f} -> Acc {base_metrics['accuracy']:.4f}, Prec {base_metrics['precision']:.4f}, Rec {base_metrics['recall']:.4f}, F1 {base_metrics['f1']:.4f}")
        print("   Confusion Matrix (TN, FP | FN, TP):")
        print(base_metrics['cm'])
        print(f"   Best F1 at {best_f1['threshold']:.3f} -> Acc {best_f1['accuracy']:.4f}, Prec {best_f1['precision']:.4f}, Rec {best_f1['recall']:.4f}, F1 {best_f1['f1']:.4f}")
        print(f"   Best Youden at {best_youden['threshold']:.3f} -> Acc {best_youden['accuracy']:.4f}, Prec {best_youden['precision']:.4f}, Rec {best_youden['recall']:.4f}, F1 {best_youden['f1']:.4f}")
        # Dataset-wide analysis
        print("\n📊 Dataset analysis:")
        # Class balance
        class_counts = df['cardio'].value_counts().to_dict()
        print(f"   Class balance: {class_counts}")
        # Feature correlations with target
        try:
            corr = df.corr(numeric_only=True)['cardio'].sort_values(ascending=False)
            print("   Top correlations with target:")
            for feature, value in corr.drop('cardio', errors='ignore').head(10).items():
                print(f"      {feature}: {value:.3f}")
        except Exception:
            pass
        # Risk distribution summary
        print("   Risk distribution (percentiles):")
        for p in [0, 10, 25, 50, 75, 90, 100]:
            val = np.percentile(y_probs * 100.0, p)
            print(f"      p{p}: {val:.1f}%")
        return
    
    # Interactive mode
    print("🏥 Welcome to the Cardiovascular Risk Calculator!")
    print("This tool will assess your cardiovascular disease risk based on your health data.")
    
    # Initialize calculator
    calculator = CardiovascularRiskCalculator()
    
    try:
        # Get user input
        user_data = calculator.get_user_input()
        
        # Calculate risk score
        risk_score = calculator.calculate_risk_score(user_data)
        
        # Get risk category and recommendations
        risk_category = calculator.get_risk_category(risk_score)
        recommendations = calculator.generate_recommendations(user_data, risk_score)
        
        # Display results
        calculator.display_results(user_data, risk_score, risk_category, recommendations)
        
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using the Cardiovascular Risk Calculator!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or contact support.")

if __name__ == "__main__":
    main()
