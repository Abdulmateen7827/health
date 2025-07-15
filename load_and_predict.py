import joblib
import pandas as pd
import numpy as np

def load_saved_model(model_path='mental_health_model.joblib'):
    """Load the saved mental health detection model"""
    try:
        model_components = joblib.load(model_path)
        print("Model loaded successfully!")
        print(f"Model type: XGBoost")
        print(f"Number of features: {len(model_components['feature_names'])}")
        print(f"Target classes: {model_components['target_classes']}")
        return model_components
    except FileNotFoundError:
        print(f"Model file {model_path} not found! Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
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
                        print(f"Warning: Unknown value '{value}' for feature '{feature}', using default")
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
                print(f"Warning: Missing feature '{feature}', using default value 0")
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
        print(f"Error making prediction: {e}")
        return None

def main():
    """Main function to demonstrate model loading and prediction"""
    print("=" * 60)
    print("MENTAL HEALTH PREDICTION USING SAVED MODEL")
    print("=" * 60)
    
    # Load the saved model
    model_components = load_saved_model()
    
    if model_components is None:
        print("Please run the training script first to create the model file.")
        return
    
    # Example patient data (you can modify these values)
    example_patients = [
        {
            'name': 'Patient A',
            'data': {
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
        },
        {
            'name': 'Patient B',
            'data': {
                'Sadness': 'Sometimes',
                'Euphoric': 'Sometimes',
                'Exhausted': 'Sometimes',
                'Sleep dissorder': 'Seldom',
                'Mood Swing': 'NO',
                'Suicidal thoughts': 'NO',
                'Anorxia': 'NO',
                'Authority Respect': 'YES',
                'Try-Explanation': 'NO',
                'Aggressive Response': 'NO',
                'Ignore & Move-On': 'YES',
                'Nervous Break-down': 'NO',
                'Admit Mistakes': 'YES',
                'Overthinking': 'NO',
                'Sexual Activity': '5 From 10',
                'Concentration': '7 From 10',
                'Optimisim': '8 From 10'
            }
        }
    ]
    
    # Make predictions for example patients
    for patient in example_patients:
        print(f"\n{'='*50}")
        print(f"PREDICTING FOR: {patient['name']}")
        print(f"{'='*50}")
        
        result = predict_mental_health(model_components, patient['data'])
        
        if result:
            print(f"Predicted Diagnosis: {result['prediction']}")
            print(f"Confidence Scores:")
            for class_name, confidence in result['confidence'].items():
                print(f"  {class_name}: {confidence:.3f}")
            
            # Highlight the predicted class
            predicted_confidence = result['confidence'][result['prediction']]
            print(f"\nConfidence in prediction: {predicted_confidence:.3f} ({predicted_confidence*100:.1f}%)")
            
            # Provide interpretation
            if predicted_confidence > 0.7:
                print("High confidence prediction")
            elif predicted_confidence > 0.5:
                print("Moderate confidence prediction")
            else:
                print("Low confidence prediction - consider additional assessment")
        else:
            print("Failed to make prediction")
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 