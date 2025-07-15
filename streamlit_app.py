import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .feature-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .stNumberInput > div > div > input {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the saved mental health detection model"""
    try:
        model_components = joblib.load('mental_health_model.joblib')
        return model_components
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'mental_health_model.joblib' exists in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_mental_health(model_components, patient_data):
    """Predict mental health diagnosis for a patient"""
    try:
        # Extract components
        model = model_components['model']
        scaler = model_components['scaler']
        label_encoder = model_components['label_encoder']
        feature_encoders = model_components['feature_encoders']
        feature_names = model_components['feature_names']
        
        # Preprocess patient data
        processed_features = []
        
        for feature in feature_names:
            if feature in patient_data:
                value = patient_data[feature]
                
                # Handle categorical features
                if feature in feature_encoders:
                    if value in feature_encoders[feature].classes_:
                        encoded_value = feature_encoders[feature].transform([value])[0]
                    else:
                        encoded_value = 0
                    processed_features.append(encoded_value)
                
                # Handle numerical features
                elif feature in ['Sexual Activity', 'Concentration', 'Optimisim']:
                    if isinstance(value, str) and 'From 10' in value:
                        numeric_value = int(value.split()[0])
                    else:
                        numeric_value = int(value)
                    processed_features.append(numeric_value)
                
                else:
                    processed_features.append(value)
            else:
                processed_features.append(0)
        
        # Scale features
        scaled_features = scaler.transform([processed_features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Convert back to original class name
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        confidence_scores = dict(zip(label_encoder.classes_, prediction_proba))
        
        return {
            'prediction': predicted_class,
            'confidence': confidence_scores,
            'prediction_proba': prediction_proba
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_confidence_chart(confidence_scores):
    """Create a beautiful confidence chart using Plotly"""
    classes = list(confidence_scores.keys())
    scores = list(confidence_scores.values())
    
    # Create color mapping
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=scores,
            marker_color=colors[:len(classes)],
            text=[f'{score:.1%}' for score in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence by Diagnosis",
        xaxis_title="Mental Health Diagnosis",
        yaxis_title="Confidence Score",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Mental Health Assessment using XGBoost</p>', unsafe_allow_html=True)
    
    # Load model
    model_components = load_model()
    
    if model_components is None:
        st.stop()
    
    # Sidebar for model info
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.info(f"**Model Type:** XGBoost")
        st.info(f"**Features:** {len(model_components['feature_names'])}")
        st.info(f"**Classes:** {len(model_components['target_classes'])}")
        
        st.markdown("## 🎯 Target Classes")
        for i, class_name in enumerate(model_components['target_classes']):
            st.write(f"• {class_name}")
        
        st.markdown("## ⚠️ Important Notice")
        st.warning("""
        This is a research/demonstration tool and should **NOT** be used for actual medical diagnosis.
        
        Always consult qualified mental health professionals for proper diagnosis and treatment.
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Patient Assessment", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Patient Symptom Assessment</h2>', unsafe_allow_html=True)
        
        # Create form for patient data
        with st.form("patient_assessment_form"):
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.markdown("### Emotional & Mood Symptoms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sadness = st.selectbox(
                    "Sadness Level",
                    ["Seldom", "Sometimes", "Usually", "Most-Often"],
                    help="How often does the patient experience sadness?"
                )
                
                euphoric = st.selectbox(
                    "Euphoric Episodes",
                    ["Seldom", "Sometimes", "Usually", "Most-Often"],
                    help="How often does the patient experience euphoric or elevated moods?"
                )
                
                exhausted = st.selectbox(
                    "Exhaustion Level",
                    ["Seldom", "Sometimes", "Usually", "Most-Often"],
                    help="How often does the patient feel exhausted?"
                )
                
                sleep_disorder = st.selectbox(
                    "Sleep Disorder",
                    ["Seldom", "Sometimes", "Usually", "Most-Often"],
                    help="How often does the patient experience sleep problems?"
                )
            
            with col2:
                mood_swing = st.selectbox(
                    "Mood Swings",
                    ["NO", "YES"],
                    help="Does the patient experience significant mood swings?"
                )
                
                suicidal_thoughts = st.selectbox(
                    "Suicidal Thoughts",
                    ["NO", "YES"],
                    help="Does the patient report suicidal thoughts?"
                )
                
                anorexia = st.selectbox(
                    "Anorexia",
                    ["NO", "YES"],
                    help="Does the patient show signs of anorexia?"
                )
                
                authority_respect = st.selectbox(
                    "Authority Respect",
                    ["NO", "YES"],
                    help="Does the patient show respect for authority?"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.markdown("### Behavioral & Cognitive Symptoms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try_explanation = st.selectbox(
                    "Try Explanation",
                    ["NO", "YES"],
                    help="Does the patient try to explain their behavior?"
                )
                
                aggressive_response = st.selectbox(
                    "Aggressive Response",
                    ["NO", "YES"],
                    help="Does the patient respond aggressively?"
                )
                
                ignore_move_on = st.selectbox(
                    "Ignore & Move On",
                    ["NO", "YES"],
                    help="Does the patient tend to ignore problems and move on?"
                )
                
                nervous_breakdown = st.selectbox(
                    "Nervous Breakdown",
                    ["NO", "YES"],
                    help="Has the patient experienced nervous breakdowns?"
                )
            
            with col2:
                admit_mistakes = st.selectbox(
                    "Admit Mistakes",
                    ["NO", "YES"],
                    help="Does the patient admit their mistakes?"
                )
                
                overthinking = st.selectbox(
                    "Overthinking",
                    ["NO", "YES"],
                    help="Does the patient tend to overthink situations?"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.markdown("### Functional Assessment (1-10 Scale)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sexual_activity = st.slider(
                    "Sexual Activity",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Rate the patient's sexual activity level (1=Very Low, 10=Very High)"
                )
            
            with col2:
                concentration = st.slider(
                    "Concentration",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Rate the patient's concentration ability (1=Very Poor, 10=Excellent)"
                )
            
            with col3:
                optimism = st.slider(
                    "Optimism",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Rate the patient's optimism level (1=Very Pessimistic, 10=Very Optimistic)"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button("🔮 Get Prediction", use_container_width=True)
        
        # Process prediction when form is submitted
        if submitted:
            # Prepare patient data
            patient_data = {
                'Sadness': sadness,
                'Euphoric': euphoric,
                'Exhausted': exhausted,
                'Sleep dissorder': sleep_disorder,
                'Mood Swing': mood_swing,
                'Suicidal thoughts': suicidal_thoughts,
                'Anorxia': anorexia,
                'Authority Respect': authority_respect,
                'Try-Explanation': try_explanation,
                'Aggressive Response': aggressive_response,
                'Ignore & Move-On': ignore_move_on,
                'Nervous Break-down': nervous_breakdown,
                'Admit Mistakes': admit_mistakes,
                'Overthinking': overthinking,
                'Sexual Activity': f"{sexual_activity} From 10",
                'Concentration': f"{concentration} From 10",
                'Optimisim': f"{optimism} From 10"
            }
            
            # Make prediction
            with st.spinner("Analyzing patient symptoms..."):
                result = predict_mental_health(model_components, patient_data)
            
            if result:
                # Display prediction results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"## 🎯 Predicted Diagnosis: **{result['prediction']}**")
                st.markdown(f"**Confidence:** {result['confidence'][result['prediction']]:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence chart
                st.plotly_chart(create_confidence_chart(result['confidence']), use_container_width=True)
                
                # Detailed confidence scores
                st.markdown("### 📊 Detailed Confidence Scores")
                col1, col2, col3, col4 = st.columns(4)
                
                for i, (class_name, confidence) in enumerate(result['confidence'].items()):
                    with [col1, col2, col3, col4][i]:
                        st.metric(
                            label=class_name,
                            value=f"{confidence:.1%}",
                            delta="High" if confidence > 0.7 else "Moderate" if confidence > 0.5 else "Low"
                        )
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                predicted_confidence = result['confidence'][result['prediction']]
                
                if predicted_confidence > 0.7:
                    st.success("**High Confidence Prediction** - The model is highly confident in this diagnosis based on the provided symptoms.")
                elif predicted_confidence > 0.5:
                    st.warning("**Moderate Confidence Prediction** - The model shows moderate confidence. Consider additional assessment.")
                else:
                    st.error("**Low Confidence Prediction** - The model shows low confidence. Professional evaluation is strongly recommended.")
                
                # Save prediction to session state for history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'prediction': result['prediction'],
                    'confidence': predicted_confidence,
                    'symptoms': patient_data
                })
    
    with tab2:
        st.markdown('<h2 class="sub-header">Model Performance & Analytics</h2>', unsafe_allow_html=True)
        
        # Load model info if available
        if os.path.exists('model_info.joblib'):
            model_info = joblib.load('model_info.joblib')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Type", model_info['model_type'])
            
            with col2:
                st.metric("Features", model_info['n_features'])
            
            with col3:
                st.metric("Classes", model_info['n_classes'])
            
            with col4:
                st.metric("Training Date", model_info['training_date'])
        
        # Feature importance visualization
        if model_components:
            st.markdown("### 🎯 Feature Importance")
            
            feature_importance = model_components['model'].feature_importances_
            feature_names = model_components['feature_names']
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Mental Health Detection",
                color='Importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction history
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.markdown("### 📈 Prediction History")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Display recent predictions
            st.dataframe(
                history_df[['timestamp', 'prediction', 'confidence']].tail(10),
                use_container_width=True
            )
            
            # Clear history button
            if st.button("Clear Prediction History"):
                st.session_state.prediction_history = []
                st.rerun()
    
    with tab3:
        st.markdown('<h2 class="sub-header">About the Mental Health Detection System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🧠 System Overview
        
        This AI-powered mental health detection system uses machine learning to analyze 17 essential behavioral symptoms 
        and provide preliminary assessments for mental health conditions.
        
        ### 📊 Dataset Information
        
        - **Dataset Size:** 120 psychology patients
        - **Features:** 17 behavioral symptoms
        - **Target Classes:** 4 mental health categories
        - **Algorithm:** XGBoost (eXtreme Gradient Boosting)
        
        ### 🎯 Target Conditions
        
        1. **Bipolar Type-1:** Manic episodes with or without depressive episodes
        2. **Bipolar Type-2:** Hypomanic episodes with depressive episodes
        3. **Depression:** Major Depressive Disorder
        4. **Normal:** Individuals with minor mental problems seeking therapy
        
        ### 🔬 Features Analyzed
        
        **Emotional & Mood Symptoms:**
        - Sadness, Euphoric episodes, Exhaustion, Sleep disorders
        - Mood swings, Suicidal thoughts, Anorexia, Authority respect
        
        **Behavioral & Cognitive Symptoms:**
        - Try explanation, Aggressive response, Ignore & move on
        - Nervous breakdown, Admit mistakes, Overthinking
        
        **Functional Assessment:**
        - Sexual activity, Concentration, Optimism (1-10 scale)
        
        ### ⚠️ Important Disclaimers
        
        - **This is a research/demonstration tool only**
        - **NOT intended for actual medical diagnosis**
        - **Always consult qualified mental health professionals**
        - **Results should not replace professional evaluation**
        
        ### 🛠️ Technical Details
        
        - **Framework:** Streamlit
        - **Machine Learning:** XGBoost
        - **Data Processing:** Scikit-learn
        - **Visualization:** Plotly
        - **Model Persistence:** Joblib
        
        ### 📝 Usage Guidelines
        
        1. Input patient symptoms accurately
        2. Review prediction confidence levels
        3. Use results as screening tool only
        4. Refer to professionals for diagnosis
        5. Maintain patient confidentiality
        """)

if __name__ == "__main__":
    main() 