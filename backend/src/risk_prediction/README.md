# Cardiovascular Risk Prediction System

This folder contains all the essential files for cardiovascular risk prediction using advanced machine learning techniques.

## Files Overview

### Core Prediction Models
- `comprehensive_stacking_ensemble.py` - Main ensemble model combining Transformer, LightGBM, Random Forest with MLP meta-learner
- `cardiovascular_transformer.py` - Transformer-based model for cardiovascular risk prediction
- `cardiovascular_risk_calculator.py` - Risk calculation utilities and prediction functions

### Interactive Interface
- `xai_dashboard.py` - Streamlit web dashboard for interactive risk assessment
- `xai_demo.py` - Demonstration script for prediction examples

### Feature Selection and Optimization
- `app.py` - Genetic algorithm feature selection
- `genetic_feature_selection.py` - Genetic algorithm implementation
- `particle_swarm_optimization.py` - PSO optimization for model parameters

### Data Processing
- `data_preparation.py` - Data preprocessing utilities
- `preprocessing_pipeline.py` - Complete data processing pipeline
- `feature_engineering.py` - Feature engineering utilities

### Model Training and Testing
- `train_meta_learner.py` - Training script for the meta-learner
- `test_all_models.py` - Comprehensive model testing
- `quick_evaluate_models.py` - Quick model evaluation

### Data Files
- `cardio_train_clean.csv` - Cleaned training dataset
- `cardio_engineered.csv` - Feature-engineered dataset
- `cardio_ga_selected.csv` - Dataset with genetically selected features
- `best_cardiovascular_model.pth` - Saved trained model

### Dependencies
- `requirements.txt` - Python package requirements

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the interactive dashboard:
   ```bash
   streamlit run xai_dashboard.py
   ```

3. Run genetic feature selection:
   ```bash
   python app.py
   ```

4. Test all models:
   ```bash
   python test_all_models.py
   ```

## Key Features

- **Ensemble Learning**: Combines multiple ML models for better accuracy
- **Genetic Algorithm**: Optimizes feature selection
- **Interactive Dashboard**: Web-based risk assessment tool
- **Transformer Architecture**: Advanced deep learning for pattern recognition
- **Comprehensive Testing**: Multiple evaluation metrics and validation

## Selected Features (GA Results)

The genetic algorithm selected the top 3 most important features:
1. age
2. ap_hi (systolic blood pressure)
3. cholesterol

These features provide optimal predictive performance while reducing model complexity.
