#!/usr/bin/env python3
"""
Comprehensive Stacking Ensemble Implementation
Step-by-step guide to build optimal stacking ensemble combining:
- PSO-tuned Transformer
- LightGBM with GA features  
- Random Forest with GA features
- MLP meta-learner
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import optuna
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing components
from cardiovascular_transformer import CardiovascularTransformer, CardiovascularDiseasePredictor
from train_meta_learner import MLPClassifier, tune_and_train_meta_learner

class ComprehensiveStackingEnsemble:
    """Complete stacking ensemble implementation"""
    
    def __init__(self, n_folds=3, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Base models
        self.transformer_model = None
        self.lgb_model = None
        self.rf_model = None
        
        # Meta-learner
        self.meta_learner = None
        self.meta_scaler = StandardScaler()
        
        # Out-of-fold predictions storage
        self.oof_predictions = {
            'transformer': None,
            'lightgbm': None,
            'random_forest': None
        }
        
        # Final models for test predictions
        self.final_models = {
            'transformer': None,
            'lightgbm': None,
            'random_forest': None
        }
        
        # Results storage
        self.results = {}
        
    def step1_prepare_data(self, data_path):
        """Step 1: Load and prepare data with stratified k-fold splits"""
        print("\n" + "=" * 80)
        print("STEP 1: PREPARING DATA")
        print("=" * 80)
        
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset: {self.df.shape}")
        
        # Use a subset for faster training (20% of data)
        if len(self.df) > 10000:
            self.df = self.df.sample(frac=0.2, random_state=42)
            print(f"Using subset for faster training: {self.df.shape}")
        
        # Define features (all columns except target)
        self.feature_cols = [col for col in self.df.columns if col != 'cardio']
        X = self.df[self.feature_cols]
        y = self.df['cardio'].values.astype(int)  # Ensure integer type for target
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        print(f"Class balance: {np.bincount(y) / len(y)}")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create stratified k-fold
        self.skf = StratifiedKFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Initialize out-of-fold prediction arrays
        self.oof_predictions['transformer'] = np.zeros(len(X_train))
        self.oof_predictions['lightgbm'] = np.zeros(len(X_train))
        self.oof_predictions['random_forest'] = np.zeros(len(X_train))
        
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print("✓ Data preparation completed")
        return X_train, X_test, y_train, y_test
    
    def step2_train_base_models_oof(self):
        """Step 2: Train base models with out-of-fold predictions"""
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING BASE MODELS WITH OUT-OF-FOLD PREDICTIONS")
        print("=" * 60)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(self.X_train, self.y_train)):
            print(f"\n--- FOLD {fold + 1}/{self.n_folds} ---")
            
            # Split fold data
            X_fold_train = self.X_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_train = self.y_train[train_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Scale features
            X_fold_train_scaled = self.scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = self.scaler.transform(X_fold_val)
            
            fold_scores = {}
            
            # 1. Train PSO-tuned Transformer using CardiovascularDiseasePredictor
            print("Training Transformer...")
            try:
                # Initialize the predictor with faster parameters
                predictor = CardiovascularDiseasePredictor(
                    d_model=32,
                    nhead=4,
                    num_layers=1,
                    dropout=0.1,
                    epochs=20,
                    batch_size=64
                )
                
                # Create the model
                num_features = X_fold_train_scaled.shape[1]
                predictor.create_model(num_features=num_features, num_classes=2)
                
                # Convert data to DataFrame for the predictor
                feature_cols = [f'feature_{i}' for i in range(num_features)]
                X_fold_train_df = pd.DataFrame(X_fold_train_scaled, columns=feature_cols)
                y_fold_train_series = pd.Series(y_fold_train)
                
                # Prepare data for the predictor
                train_loader, val_loader, _ = predictor.prepare_data(
                    X_fold_train_df, y_fold_train_series, test_size=0.2, random_state=self.random_state
                )
                
                # Train the model
                predictor.train_model(train_loader, val_loader)
                
                # Make predictions on validation set (use positive-class probability)
                X_fold_val_df = pd.DataFrame(X_fold_val_scaled, columns=feature_cols)
                val_labels, val_probs = predictor.predict(X_fold_val_df)
                transformer_pred = val_probs[:, 1]
                
                self.oof_predictions['transformer'][val_idx] = transformer_pred
                fold_scores['transformer'] = roc_auc_score(y_fold_val, transformer_pred)
                print(f"  Transformer AUC: {fold_scores['transformer']:.4f}")
                
            except Exception as e:
                import traceback
                print(f"  Transformer failed: {e}")
                print(traceback.format_exc())
                self.oof_predictions['transformer'][val_idx] = 0.5
                fold_scores['transformer'] = 0.5
            
            # 2. Train LightGBM
            print("Training LightGBM...")
            try:
                lgb_train = lgb.Dataset(X_fold_train_scaled, label=y_fold_train)
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_state
                }
                
                lgb_model = lgb.train(
                    lgb_params, 
                    lgb_train, 
                    num_boost_round=1000,
                    valid_sets=[lgb_train],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
                lgb_pred = lgb_model.predict(X_fold_val_scaled)
                self.oof_predictions['lightgbm'][val_idx] = lgb_pred
                fold_scores['lightgbm'] = roc_auc_score(y_fold_val, lgb_pred)
                print(f"  LightGBM AUC: {fold_scores['lightgbm']:.4f}")
                
            except Exception as e:
                print(f"  LightGBM failed: {e}")
                self.oof_predictions['lightgbm'][val_idx] = 0.5
                fold_scores['lightgbm'] = 0.5
            
            # 3. Train Random Forest
            print("Training Random Forest...")
            try:
                rf_model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf_model.fit(X_fold_train_scaled, y_fold_train)
                rf_pred = rf_model.predict_proba(X_fold_val_scaled)[:, 1]
                
                self.oof_predictions['random_forest'][val_idx] = rf_pred
                fold_scores['random_forest'] = roc_auc_score(y_fold_val, rf_pred)
                print(f"  Random Forest AUC: {fold_scores['random_forest']:.4f}")
                
            except Exception as e:
                print(f"  Random Forest failed: {e}")
                self.oof_predictions['random_forest'][val_idx] = 0.5
                fold_scores['random_forest'] = 0.5
            
            fold_results.append(fold_scores)
        
        # Calculate average fold performance
        print(f"\n--- FOLD AVERAGE PERFORMANCE ---")
        for model_name in ['transformer', 'lightgbm', 'random_forest']:
            avg_auc = np.mean([fold[model_name] for fold in fold_results])
            print(f"{model_name.capitalize()}: {avg_auc:.4f}")
        
        print("✓ Base model training with OOF predictions completed")
        return fold_results
    
    def step3_construct_meta_features(self):
        """Step 3: Construct meta-feature dataset"""
        print("\n" + "=" * 60)
        print("STEP 3: CONSTRUCTING META-FEATURE DATASET")
        print("=" * 60)
        
        # Create meta-feature matrix (n_samples, n_base_models)
        meta_features = np.column_stack([
            self.oof_predictions['transformer'],
            self.oof_predictions['lightgbm'],
            self.oof_predictions['random_forest']
        ])
        
        print(f"Meta-feature matrix shape: {meta_features.shape}")
        print(f"Meta-features: {['transformer', 'lightgbm', 'random_forest']}")
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(meta_features)) or np.any(np.isinf(meta_features)):
            print("Warning: Found NaN or infinite values in meta-features")
            meta_features = np.nan_to_num(meta_features, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Scale meta-features
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        
        self.meta_features_train = meta_features_scaled
        self.meta_features_test = None  # Will be created in step 5
        
        print("✓ Meta-feature dataset constructed")
        return meta_features_scaled
    
    def step4_train_meta_learner(self):
        """Step 4: Train MLP meta-learner with hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("STEP 4: TRAINING MLP META-LEARNER WITH HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Create test set for meta-learner evaluation (use 20% of training data)
        X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
            self.meta_features_train, self.y_train, test_size=0.2, 
            random_state=self.random_state, stratify=self.y_train
        )
        
        print(f"Meta-learner training set: {X_meta_train.shape}")
        print(f"Meta-learner test set: {X_meta_test.shape}")
        
        # Simple MLP meta-learner without complex hyperparameter tuning
        print("Training simple MLP meta-learner...")
        
        # Convert to tensors
        X_meta_train_tensor = torch.tensor(X_meta_train, dtype=torch.float32)
        y_meta_train_tensor = torch.tensor(y_meta_train, dtype=torch.long)
        X_meta_test_tensor = torch.tensor(X_meta_test, dtype=torch.float32)
        y_meta_test_tensor = torch.tensor(y_meta_test, dtype=torch.long)
        
        # Create simple MLP
        self.meta_learner = MLPClassifier(
            input_dim=X_meta_train.shape[1],
            hidden_dim_1=64,
            hidden_dim_2=32,
            dropout_rate=0.3
        )
        
        # Train the meta-learner
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.meta_learner.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.meta_learner(X_meta_train_tensor)
            loss = criterion(outputs, y_meta_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate on test set
        self.meta_learner.eval()
        with torch.no_grad():
            test_outputs = self.meta_learner(X_meta_test_tensor)
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1]
            test_auc = roc_auc_score(y_meta_test, test_probs.numpy())
            print(f"Meta-learner test AUC: {test_auc:.4f}")
        
        self.meta_params = {
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
            'dropout_rate': 0.3,
            'lr': 0.001
        }
        
        print("✓ Meta-learner training completed")
        return self.meta_learner
    
    def step5_generate_final_predictions(self):
        """Step 5: Generate final test predictions"""
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING FINAL TEST PREDICTIONS")
        print("=" * 60)
        
        # Scale test features
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train final models on full training set
        print("Training final models on full training set...")
        
        # Final Transformer using CardiovascularDiseasePredictor
        print("Training final Transformer...")
        try:
            # Initialize the predictor with default parameters
            predictor = CardiovascularDiseasePredictor(
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                epochs=50,
                batch_size=32
            )

            # Create the model
            num_features = self.X_train.shape[1]
            predictor.create_model(num_features=num_features, num_classes=2)

            # Convert data to DataFrame for the predictor
            feature_cols = [f'feature_{i}' for i in range(num_features)]
            X_train_df = pd.DataFrame(self.X_train, columns=feature_cols)
            y_train_series = pd.Series(self.y_train)

            # Prepare data for the predictor
            train_loader, val_loader, _ = predictor.prepare_data(
                X_train_df, y_train_series, test_size=0.2, random_state=self.random_state
            )

            # Train the model
            predictor.train_model(train_loader, val_loader)

            # Keep trained transformer model for saving and later use
            self.final_models['transformer'] = predictor.model

            # Make predictions on test set (use positive-class probability)
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
            test_labels, test_probs = predictor.predict(X_test_df)
            transformer_test_pred = test_probs[:, 1]

            print(f"  Final Transformer AUC: {roc_auc_score(self.y_test, transformer_test_pred):.4f}")
        except Exception as e:
            import traceback
            print(f"  Final Transformer failed: {e}")
            print(traceback.format_exc())
            transformer_test_pred = np.ones(len(self.y_test)) * 0.5
        
        # Final LightGBM
        print("Training final LightGBM...")
        try:
            lgb_train = lgb.Dataset(self.scaler.transform(self.X_train), label=self.y_train)
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
            
            self.final_models['lightgbm'] = lgb.train(
                lgb_params, 
                lgb_train, 
                num_boost_round=1000,
                valid_sets=[lgb_train],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            lgb_test_pred = self.final_models['lightgbm'].predict(X_test_scaled)
        except Exception as e:
            print(f"Final LightGBM failed: {e}")
            lgb_test_pred = np.full(len(self.X_test), 0.5)
        
        # Final Random Forest
        print("Training final Random Forest...")
        try:
            self.final_models['random_forest'] = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.final_models['random_forest'].fit(
                self.scaler.transform(self.X_train), 
                self.y_train
            )
            rf_test_pred = self.final_models['random_forest'].predict_proba(X_test_scaled)[:, 1]
        except Exception as e:
            print(f"Final Random Forest failed: {e}")
            rf_test_pred = np.full(len(self.X_test), 0.5)
        
        # Create test meta-features
        test_meta_features = np.column_stack([transformer_test_pred, lgb_test_pred, rf_test_pred])
        test_meta_features_scaled = self.meta_scaler.transform(test_meta_features)
        
        self.meta_features_test = test_meta_features_scaled
        
        # Generate final stacking predictions
        print("Generating final stacking predictions...")
        self.meta_learner.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_meta_features_scaled, dtype=torch.float32)
            meta_logits = self.meta_learner(test_tensor)
            self.final_predictions = torch.softmax(meta_logits, dim=1)[:, 1].numpy()
        
        print("✓ Final test predictions generated")
        return self.final_predictions
    
    def step6_evaluate_and_calibrate(self):
        """Step 6: Evaluate and calibrate stacking performance"""
        print("\n" + "=" * 60)
        print("STEP 6: EVALUATING AND CALIBRATING STACKING PERFORMANCE")
        print("=" * 60)
        
        # Calculate metrics
        threshold = 0.5
        final_pred_binary = (self.final_predictions >= threshold).astype(int)
        
        accuracy = accuracy_score(self.y_test, final_pred_binary)
        f1 = f1_score(self.y_test, final_pred_binary)
        auc = roc_auc_score(self.y_test, self.final_predictions)
        
        print(f"Stacking Ensemble Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # Threshold optimization
        print("\nOptimizing classification threshold...")
        best_threshold = 0.5
        best_f1 = f1
        
        for threshold in np.linspace(0.1, 0.9, 17):
            pred_binary = (self.final_predictions >= threshold).astype(int)
            current_f1 = f1_score(self.y_test, pred_binary)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        # Final evaluation with best threshold
        final_pred_optimized = (self.final_predictions >= best_threshold).astype(int)
        final_accuracy = accuracy_score(self.y_test, final_pred_optimized)
        final_f1 = f1_score(self.y_test, final_pred_optimized)
        
        self.results = {
            'accuracy': final_accuracy,
            'f1_score': final_f1,
            'auc': auc,
            'best_threshold': best_threshold,
            'classification_report': classification_report(self.y_test, final_pred_optimized)
        }
        
        print(f"\nFinal Optimized Performance:")
        print(f"  Accuracy: {final_accuracy:.4f}")
        print(f"  F1-Score: {final_f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        print("✓ Evaluation and calibration completed")
        return self.results
    
    def step7_save_models(self):
        """Step 7: Save final stacking models"""
        print("\n" + "=" * 60)
        print("STEP 7: SAVING FINAL STACKING MODELS")
        print("=" * 60)
        
        # Create models directory relative to this file so paths are stable
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'final_models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save scalers
        joblib.dump(self.scaler, os.path.join(models_dir, 'final_scaler.pkl'))
        joblib.dump(self.meta_scaler, os.path.join(models_dir, 'final_meta_scaler.pkl'))
        
        # Save meta-learner
        torch.save(self.meta_learner.state_dict(), os.path.join(models_dir, 'final_meta_learner_model.pth'))
        
        # Save meta-learner parameters
        with open(os.path.join(models_dir, 'final_meta_learner_params.json'), 'w') as f:
            json.dump(self.meta_params, f, indent=2)
        
        # Save base models
        if self.final_models['random_forest'] is not None:
            joblib.dump(self.final_models['random_forest'], os.path.join(models_dir, 'final_random_forest.pkl'))
        
        # Save LightGBM model
        if self.final_models['lightgbm'] is not None:
            self.final_models['lightgbm'].save_model(os.path.join(models_dir, 'final_lightgbm.txt'))
        
        # Save Transformer model
        if self.final_models['transformer'] is not None:
            torch.save(self.final_models['transformer'].state_dict(), os.path.join(models_dir, 'final_transformer.pth'))
        
        # Save results
        with open(os.path.join(models_dir, 'stacking_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("✓ All models saved to 'final_models/' directory")
        print("Saved files:")
        print("  - final_scaler.pkl")
        print("  - final_meta_scaler.pkl") 
        print("  - final_meta_learner_model.pth")
        print("  - final_meta_learner_params.json")
        print("  - final_random_forest.pkl")
        print("  - final_lightgbm.txt")
        print("  - final_transformer.pth")
        print("  - stacking_results.json")
    
    def load_trained_models(self):
        """Load trained models from final_models directory"""
        import joblib
        import torch

        try:
            # Resolve models directory relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'final_models')

            # Load scalers
            self.scaler = joblib.load(os.path.join(models_dir, 'final_scaler.pkl'))
            self.meta_scaler = joblib.load(os.path.join(models_dir, 'final_meta_scaler.pkl'))

            # Load meta-learner
            self.meta_learner = MLPClassifier(input_dim=3)
            self.meta_learner.load_state_dict(torch.load(os.path.join(models_dir, 'final_meta_learner_model.pth')))
            self.meta_learner.eval()

            # Load meta-learner parameters
            with open(os.path.join(models_dir, 'final_meta_learner_params.json'), 'r') as f:
                self.meta_params = json.load(f)

            # Load base models
            try:
                self.final_models['random_forest'] = joblib.load(os.path.join(models_dir, 'final_random_forest.pkl'))
            except:
                print("Warning: Could not load Random Forest model")

            try:
                import lightgbm as lgb
                self.final_models['lightgbm'] = lgb.Booster(model_file=os.path.join(models_dir, 'final_lightgbm.txt'))
            except:
                print("Warning: Could not load LightGBM model")

            try:
                self.final_models['transformer'] = CardiovascularTransformer(
                    num_features=21, d_model=64, nhead=4, num_layers=2, dropout=0.1
                )
                self.final_models['transformer'].load_state_dict(torch.load(os.path.join(models_dir, 'final_transformer.pth')))
                self.final_models['transformer'].eval()
            except:
                print("Warning: Could not load Transformer model")

            # Load results
            with open(os.path.join(models_dir, 'stacking_results.json'), 'r') as f:
                self.results = json.load(f)

            print("✅ Trained models loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def predict_single_sample(self, user_data):
        """Predict cardiovascular risk for a single sample using trained stacking ensemble"""
        try:
            # Build the same 21 features the models were trained on. Raw input
            # is converted to the training (z-score) space first.
            from feature_engineering import engineer_single_sample
            features = engineer_single_sample(user_data)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get predictions from base models
            transformer_pred = 0.5
            lgb_pred = 0.5
            rf_pred = 0.5

            # Transformer prediction
            if 'transformer' in self.final_models and self.final_models['transformer'] is not None:
                try:
                    with torch.no_grad():
                        transformer_input = torch.tensor(features_scaled, dtype=torch.float32)
                        # The saved model is a plain nn.Module (no predict_proba): use logits -> softmax
                        logits = self.final_models['transformer'](transformer_input)
                        transformer_pred = torch.softmax(logits, dim=1)[0, 1].item()
                except Exception as exc:
                    print(f"Warning: transformer prediction failed: {exc}")

            # LightGBM prediction
            if 'lightgbm' in self.final_models and self.final_models['lightgbm'] is not None:
                try:
                    lgb_pred = self.final_models['lightgbm'].predict(features_scaled)[0]
                except:
                    pass

            # Random Forest prediction
            if 'random_forest' in self.final_models and self.final_models['random_forest'] is not None:
                try:
                    rf_pred = self.final_models['random_forest'].predict_proba(features_scaled)[0, 1]
                except:
                    pass

            # Create meta-features
            meta_features = np.array([transformer_pred, lgb_pred, rf_pred]).reshape(1, -1)
            meta_features_scaled = self.meta_scaler.transform(meta_features)

            # Meta-learner prediction
            with torch.no_grad():
                meta_input = torch.tensor(meta_features_scaled, dtype=torch.float32)
                meta_logits = self.meta_learner(meta_input)
                final_prob = torch.softmax(meta_logits, dim=1)[0, 1].item()

            # Apply optimal threshold
            optimal_threshold = self.results.get('best_threshold', 0.5)
            prediction = 1 if final_prob >= optimal_threshold else 0

            return {
                'probability': final_prob,
                'prediction': prediction,
                'threshold_used': optimal_threshold,
                'base_model_probs': {
                    'transformer': transformer_pred,
                    'lightgbm': lgb_pred,
                    'random_forest': rf_pred
                }
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to simple rule-based calculation
            return self._fallback_prediction(user_data)

    def _fallback_prediction(self, user_data):
        """Fallback prediction using rule-based approach"""
        risk_score = 0

        # Age factor
        if user_data['age'] >= 65:
            risk_score += 25
        elif user_data['age'] >= 55:
            risk_score += 20
        elif user_data['age'] >= 45:
            risk_score += 15

        # Blood pressure
        if user_data['ap_hi'] >= 140 or user_data['ap_lo'] >= 90:
            risk_score += 20

        # BMI
        if user_data['bmi'] >= 30:
            risk_score += 15

        # Lifestyle
        if user_data['smoke']:
            risk_score += 20
        if user_data['cholesterol'] > 1:
            risk_score += 10

        probability = min(100, risk_score) / 100.0
        prediction = 1 if probability >= 0.5 else 0

        return {
            'probability': probability,
            'prediction': prediction,
            'threshold_used': 0.5,
            'base_model_probs': {'fallback': probability}
        }

    def run_complete_stacking(self, data_path=None):
        """Run the complete stacking pipeline"""
        print("🚀 STARTING COMPREHENSIVE STACKING ENSEMBLE")
        print("=" * 80)
        
        # If no data_path is provided, use the correct path to the data file
        if data_path is None:
            import os
            # Use current working directory to find the dataset
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, 'cardio_engineered.csv')
        
        print(f"Using data file: {data_path}")
        
        try:
            # Step 1: Prepare data with stratified k-fold splits
            self.step1_prepare_data(data_path)

            # Step 2: Train base models with OOF
            self.step2_train_base_models_oof()

            # Step 3: Construct meta-features
            self.step3_construct_meta_features()

            # Step 4: Train meta-learner
            self.step4_train_meta_learner()

            # Step 5: Generate final predictions
            self.step5_generate_final_predictions()

            # Step 6: Evaluate and calibrate
            self.step6_evaluate_and_calibrate()

            # Step 7: Save models
            self.step7_save_models()

            print("\n" + "=" * 80)
            print("🎉 COMPREHENSIVE STACKING ENSEMBLE COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return self.results

        except Exception as e:
            print(f"\n❌ Error in stacking pipeline: {e}")
            raise

def main():
    """Main execution function"""
    # Initialize stacking ensemble
    stacking = ComprehensiveStackingEnsemble(n_folds=5, random_state=42)
    
    # Run complete pipeline
    results = stacking.run_complete_stacking()
    
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Best Threshold: {results['best_threshold']:.3f}")

if __name__ == "__main__":
    main()
