#!/usr/bin/env python3
"""
Test script to verify model saving and loading functionality
"""

import os
import joblib
import pandas as pd
import numpy as np
from mental_health_detection import MentalHealthDetector

def test_model_saving():
    """Test the model saving functionality"""
    print("Testing Model Saving and Loading Functionality")
    print("=" * 50)
    
    # Check if model file exists
    model_file = 'mental_health_model.joblib'
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found.")
        print("Please run the training script first:")
        print("python mental_health_detection.py")
        return False
    
    try:
        # Load the saved model
        print("Loading saved model...")
        model_components = joblib.load(model_file)
        
        # Verify all components are present
        required_components = ['model', 'scaler', 'label_encoder', 'feature_encoders', 
                             'feature_names', 'target_classes']
        
        for component in required_components:
            if component not in model_components:
                print(f"Missing component: {component}")
                return False
        
        print("✓ All model components loaded successfully")
        print(f"✓ Model type: XGBoost")
        print(f"✓ Number of features: {len(model_components['feature_names'])}")
        print(f"✓ Target classes: {model_components['target_classes']}")
        
        # Test prediction functionality
        print("\nTesting prediction functionality...")
        
        # Create test patient data
        test_patient = {
            'Sadness': 'Sometimes',
            'Euphoric': 'Sometimes',
            'Exhausted': 'Sometimes',
            'Sleep dissorder': 'Sometimes',
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
            'Concentration': '5 From 10',
            'Optimisim': '5 From 10'
        }
        
        # Initialize detector and load model
        detector = MentalHealthDetector()
        if detector.load_model():
            # Make prediction
            result = detector.predict_single_patient(test_patient)
            
            if result:
                print("✓ Prediction successful")
                print(f"✓ Predicted diagnosis: {result['prediction']}")
                print(f"✓ Confidence scores: {result['confidence']}")
                return True
            else:
                print("✗ Prediction failed")
                return False
        else:
            print("✗ Failed to load model")
            return False
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def test_model_info():
    """Test model info file"""
    print("\nTesting Model Info File")
    print("=" * 30)
    
    info_file = 'model_info.joblib'
    if os.path.exists(info_file):
        try:
            model_info = joblib.load(info_file)
            print("✓ Model info loaded successfully")
            print(f"✓ Model type: {model_info['model_type']}")
            print(f"✓ Training date: {model_info['training_date']}")
            print(f"✓ Number of features: {model_info['n_features']}")
            print(f"✓ Number of classes: {model_info['n_classes']}")
            return True
        except Exception as e:
            print(f"✗ Error loading model info: {e}")
            return False
    else:
        print(f"✗ Model info file {info_file} not found")
        return False

def main():
    """Main test function"""
    print("MENTAL HEALTH MODEL TESTING")
    print("=" * 40)
    
    # Test model saving/loading
    model_test_passed = test_model_saving()
    
    # Test model info
    info_test_passed = test_model_info()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    if model_test_passed and info_test_passed:
        print("✓ All tests passed!")
        print("✓ Model saving and loading functionality is working correctly")
        print("✓ You can now use the model for predictions")
    else:
        print("✗ Some tests failed")
        print("✗ Please check the error messages above")
    
    return model_test_passed and info_test_passed

if __name__ == "__main__":
    main() 