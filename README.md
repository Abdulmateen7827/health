# Mental Health Detection using XGBoost

This project implements a machine learning system for detecting mental health disorders using XGBoost algorithm. The system analyzes 17 essential behavioral symptoms to classify patients into different mental health categories.

## 🚀 Quick Start

Get started in minutes with our beautiful Streamlit web application:

```bash
# Option 1: Quick start (recommended)
python quick_start.py

# Option 2: Manual setup
pip install -r requirements.txt
python mental_health_detection.py
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## Dataset Description

The dataset contains 120 psychology patients with 17 essential symptoms used by psychiatrists to diagnose:
- **Bipolar Type-1**: Manic episodes with or without depressive episodes
- **Bipolar Type-2**: Hypomanic episodes with depressive episodes  
- **Depression**: Major Depressive Disorder
- **Normal**: Individuals with minor mental problems seeking therapy for counseling and personal development

### Features (17 Behavioral Symptoms)

**Categorical Features:**
- Sadness, Euphoric, Exhausted, Sleep disorder
- Mood Swing, Suicidal thoughts, Anorexia, Authority Respect
- Try-Explanation, Aggressive Response, Ignore & Move-On
- Nervous Break-down, Admit Mistakes, Overthinking

**Numerical Features (1-10 scale):**
- Sexual Activity, Concentration, Optimism

## Features

- **Data Preprocessing**: Automatic encoding of categorical variables and scaling of numerical features
- **XGBoost Model**: State-of-the-art gradient boosting algorithm for classification
- **Model Evaluation**: Comprehensive evaluation with accuracy, classification report, and confusion matrix
- **Visualizations**: 
  - Confusion matrix heatmap
  - Feature importance analysis
  - Target distribution plots
- **Cross-validation**: 5-fold cross-validation for robust performance assessment
- **Model Persistence**: Save and load trained models for later use
- **Prediction Interface**: Make predictions on new patient data
- **🎨 Beautiful Web Interface**: Professional Streamlit app with intuitive UI

## 🎨 Streamlit Web Application

### ✨ Beautiful Features

- **Professional UI/UX**: Modern, clean interface with responsive design
- **Interactive Forms**: Easy-to-use input fields for all 17 symptoms
- **Real-time Predictions**: Instant results with confidence scores
- **Advanced Visualizations**: Interactive charts using Plotly
- **Mobile Responsive**: Works perfectly on desktop and mobile devices
- **Prediction History**: Track previous assessments during the session
- **Model Analytics**: Feature importance and performance metrics

### 🔍 Patient Assessment Interface

The web app provides intuitive input controls for all symptoms:

**Emotional & Mood Symptoms:**
- Dropdown menus for Sadness, Euphoric episodes, Exhaustion, Sleep disorders
- Yes/No options for Mood swings, Suicidal thoughts, Anorexia, Authority respect

**Behavioral & Cognitive Symptoms:**
- Yes/No options for Try explanation, Aggressive response, Ignore & move on
- Yes/No options for Nervous breakdown, Admit mistakes, Overthinking

**Functional Assessment:**
- Slider controls (1-10 scale) for Sexual activity, Concentration, Optimism

### 📊 Results & Analytics

- **Prediction Display**: Clear diagnosis with confidence percentage
- **Confidence Chart**: Interactive bar chart showing confidence for all classes
- **Detailed Metrics**: Confidence scores for each mental health category
- **Interpretation Guide**: Professional guidance based on confidence levels
- **Feature Importance**: Visual representation of symptom importance

### 🛡️ Safety & Ethics

- **Medical Disclaimers**: Prominent warnings about research use only
- **Professional Guidance**: Emphasis on consulting mental health professionals
- **No Data Storage**: Session-based processing (no permanent patient data)
- **Confidentiality**: All data processed locally

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🚀 Quick Start (Recommended)

```bash
python quick_start.py
```

This script will:
- Check and install missing dependencies
- Train the model if needed
- Launch the Streamlit app automatically

### Manual Setup

#### Training the Model

Run the main training script:
```bash
python mental_health_detection.py
```

The script will:
1. Load and preprocess the dataset
2. Train an XGBoost model
3. Evaluate model performance
4. Generate visualizations
5. Save the trained model and preprocessing components

#### Launching the Web App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Making Predictions

After training, you can use the saved model to make predictions:
```bash
python load_and_predict.py
```

This script demonstrates how to:
- Load the saved model
- Preprocess new patient data
- Make predictions with confidence scores

## Output Files

The training script generates:
- `confusion_matrix.png`: Confusion matrix showing prediction accuracy
- `feature_importance.png`: Feature importance ranking
- `target_distribution.png`: Distribution of mental health diagnoses
- `mental_health_model.joblib`: Saved model with all preprocessing components
- `model_info.joblib`: Model metadata and information

## Model Performance

The XGBoost model typically achieves:
- High accuracy on the test set
- Good generalization through cross-validation
- Clear feature importance insights
- Balanced performance across all mental health categories

## Technical Details

- **Algorithm**: XGBoost (eXtreme Gradient Boosting)
- **Hyperparameters**: 
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- **Data Split**: 80% training, 20% testing (stratified)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Model Persistence**: Uses joblib for efficient model serialization
- **Web Framework**: Streamlit with Plotly visualizations

## Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment

See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions:

- **Streamlit Cloud** (Recommended): Easy deployment with GitHub integration
- **Heroku**: Cloud platform deployment
- **Docker**: Containerized deployment
- **Custom Server**: Self-hosted deployment

## Making Predictions on New Data

You can use the saved model to predict mental health diagnoses for new patients:

```python
import joblib

# Load the saved model
model_components = joblib.load('mental_health_model.joblib')

# Example patient data
patient_data = {
    'Sadness': 'Usually',
    'Euphoric': 'Seldom',
    'Exhausted': 'Usually',
    'Sleep dissorder': 'Sometimes',
    'Mood Swing': 'YES',
    'Suicidal thoughts': 'YES',
    'Anorxia': 'NO',
    'Authority Respect': 'NO',
    'Try-Explanation': 'YES',
    'Aggressive Response': 'NO',
    'Ignore & Move-On': 'NO',
    'Nervous Break-down': 'YES',
    'Admit Mistakes': 'YES',
    'Overthinking': 'YES',
    'Sexual Activity': '3 From 10',
    'Concentration': '2 From 10',
    'Optimisim': '4 From 10'
}

# Make prediction
result = predict_mental_health(model_components, patient_data)
print(f"Predicted Diagnosis: {result['prediction']}")
```

## Important Notes

- This is a research/demonstration project and should not be used for actual medical diagnosis
- The model is trained on a limited dataset and may not generalize to all populations
- Always consult qualified mental health professionals for actual diagnosis and treatment
- The dataset contains sensitive mental health information and should be handled with appropriate privacy considerations

## Contributing

Feel free to contribute by:
- Improving the model architecture
- Adding new evaluation metrics
- Enhancing visualizations
- Expanding the dataset analysis
- Adding new prediction features
- Improving the web interface

## License

This project is for educational and research purposes only. 