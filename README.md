# CheckYourHeartHealth: Transformer-Based Heart Disease Prediction Model

## Overview

CheckYourHeartHealth is an advanced machine learning system for cardiovascular disease risk prediction. It combines cutting-edge deep learning techniques (Transformers), evolutionary algorithms (Genetic Algorithm for feature selection), and swarm intelligence (Particle Swarm Optimization for hyperparameter tuning) to provide accurate, explainable heart disease risk assessments.

## Key Features

- **Multi-Modal Input**: Accepts health and lifestyle data via web app, command line, or API
- **Advanced Feature Engineering**: Automatically computes medical features like BMI, pulse pressure, MAP, and risk interactions
- **Intelligent Feature Selection**: Uses Genetic Algorithms to identify the most predictive features
- **Hyperparameter Optimization**: Employs Particle Swarm Optimization for optimal model tuning
- **Ensemble Learning**: Combines Random Forest, LightGBM, and Transformer models with stacking
- **Explainable AI**: Provides SHAP and LIME explanations for model decisions
- **What-If Analysis**: Allows users to explore how lifestyle changes affect risk
- **Personalized Recommendations**: Generates tailored health advice based on risk factors
- **Comprehensive Reporting**: Saves results, explanations, and PDF reports

## Architecture

```
Input Data → Data Cleaning → Feature Engineering → Feature Selection (GA) → Model Training (PSO-tuned) → Stacking Ensemble → Risk Scoring → XAI Explanations
```

## How It Works

### Step 1: Data Input
Users provide health information such as:
- age, gender, height, weight
- blood pressure (systolic/diastolic)
- cholesterol and glucose levels
- lifestyle factors (smoking, alcohol, physical activity)

### Step 2: Feature Engineering
The system automatically computes additional medical features:
- BMI (Body Mass Index)
- Pulse pressure (ap_hi - ap_lo)
- Mean Arterial Pressure (MAP)
- Age groups and BMI categories
- Blood pressure stages
- Feature interactions (e.g., age × BMI)

### Step 3: Feature Selection with Genetic Algorithm
- GA explores different feature subsets
- Selects the optimal combination that maximizes predictive power while minimizing complexity
- Ensures model stability and interpretability

### Step 4: Hyperparameter Tuning with Particle Swarm Optimization
- PSO optimizes model parameters (e.g., number of trees, learning rate)
- Searches efficiently through parameter space for best performance
- Applied to Random Forest, LightGBM, and Transformer models

### Step 5: Ensemble Model Training
- **Random Forest**: Robust baseline model
- **LightGBM**: Gradient boosting for efficiency
- **Transformer**: Deep learning model for complex pattern recognition
- **Stacking Meta-Learner**: Combines predictions for superior accuracy

### Step 6: Risk Assessment
- Produces calibrated probability scores (0-100%)
- Applies decision threshold for risk categorization
- Generates personalized recommendations

### Step 7: Explainable AI
- SHAP values show feature importance
- LIME provides local explanations
- What-if scenarios demonstrate impact of changes

## Example Usage

**Input:**
```
age=45, gender=Male, height=170 cm, weight=75 kg
ap_hi=130, ap_lo=85, cholesterol=2, glucose=1
smoke=0, alco=0, active=1
(BMI computed ≈ 25.9)
```

**Process:**
1. Feature engineering adds pulse_pressure=45, MAP, BMI category, etc.
2. GA selects optimal feature subset
3. PSO tunes model hyperparameters
4. Ensemble predicts risk probability: 0.32 (32%)
5. Threshold (0.35) classifies as "MODERATE RISK"

**Output:**
- Risk Score: 32.0%
- Category: Moderate Risk
- Recommendations: Monitor blood pressure, maintain healthy weight, stay active
- Explanations: Shows which factors contributed to the score

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
cd CheckYourHeartHealth/backend
pip install -r requirements.txt
```

### Running the Application
```bash
# Web App
streamlit run app.py

# Command Line Interface
python src/cardiovascular_risk_calculator.py --input your_data.json

# API Server
python src/app.py  # Flask API
```

## Project Structure

```
CheckYourHeartHealth/
├── backend/
│   ├── app.py                    # Main Streamlit web application
│   ├── requirements.txt          # Python dependencies
│   ├── models/                   # Trained model files
│   ├── data/                     # Datasets and processed data
│   ├── src/                      # Source code
│   │   ├── cardiovascular_risk_calculator.py    # Main calculator class
│   │   ├── cardiovascular_transformer.py         # Transformer model
│   │   ├── genetic_feature_selection.py          # GA feature selection
│   │   ├── particle_swarm_optimization.py        # PSO optimization
│   │   ├── feature_engineering.py               # Feature creation
│   │   ├── preprocessing_pipeline.py            # Data preprocessing
│   │   └── [other modules]
│   ├── tests/                    # Unit tests
│   ├── visualizations/           # Generated plots
│   └── results/                  # Analysis results
├── docs/                         # Documentation
└── README.md                     # This file
```

## Models & Algorithms

### Primary Models
- **Transformer Classifier**: Handles sequential and complex feature relationships
- **Random Forest**: Ensemble baseline with high interpretability
- **LightGBM**: Efficient gradient boosting for large datasets

### Optimization Algorithms
- **Genetic Algorithm (GA)**: Evolutionary feature selection
- **Particle Swarm Optimization (PSO)**: Hyperparameter tuning

### Explainability Tools
- **SHAP**: Global and local feature importance
- **LIME**: Model-agnostic local explanations

## API Usage

### Risk Calculation
```python
from cardiovascular_risk_calculator import CardiovascularRiskCalculator

calculator = CardiovascularRiskCalculator()
user_data = {
    'age': 45, 'gender': 2, 'height': 170, 'weight': 75,
    'ap_hi': 130, 'ap_lo': 85, 'cholesterol': 2, 'gluc': 1,
    'smoke': 0, 'alco': 0, 'active': 1, 'bmi': 25.9
}
risk_score = calculator.calculate_risk_score(user_data)
```

### Feature Selection
```python
from genetic_feature_selection import GeneticFeatureSelector

selector = GeneticFeatureSelector()
best_features = selector.select_features(X_train, y_train)
```

## Deployment

### Local Deployment
- Run `streamlit run app.py` for web interface
- Use Flask API for programmatic access

### Cloud Deployment
- Compatible with Heroku, AWS, Google Cloud
- Docker containerization available
- Scalable for high-traffic applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with scikit-learn, PyTorch, and other open-source ML libraries
- Inspired by medical research on cardiovascular risk assessment
- Designed for educational and research purposes

## Contact

For questions or collaborations, please open an issue on GitHub or contact the maintainers.

## Quick start

```bash
git clone https://github.com/jpoojaj/transformer-heart-disease-prediction-system.git
cd transformer-heart-disease-prediction-system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env        # then edit .env
cd backend
streamlit run app.py
```

The first time you open the app, register a user (the local SQLite database `heart_health.db` is created automatically).

## Use your own keys

This repo does **not** contain any API keys. The `.env` file is ignored by git.
Copy `.env.example` to `.env` and fill in **your own** keys (Google client ID/secret, Supabase URL/key). Do not use anyone else's keys and never commit your `.env`.

## Notes

- Risk models live in `backend/src/risk_prediction/final_models/`. They were trained on standardised (z-score) features, so the app converts raw inputs (age in years, weight in kg, BP in mmHg) to that space before predicting (`feature_engineering.py`).
- The 3D heart viewer needs the model files (`.obj`, textures) in `backend/heart-master/models/`. They are not in this repo because of size.
- ECG classification needs separate ECG model files; without them the app falls back to a default result.
- For educational use only. Not medical advice.
