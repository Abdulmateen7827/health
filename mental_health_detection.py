import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MentalHealthDetector:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_encoders = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the mental health dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Display target distribution
        print("\nTarget distribution:")
        print(df['Expert Diagnose'].value_counts())
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("\nEncoding categorical features...")
        
        # Features to encode
        categorical_features = [
            'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 
            'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
            'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
            'Nervous Break-down', 'Admit Mistakes', 'Overthinking'
        ]
        
        # Create a copy for encoding
        df_encoded = df.copy()
        
        # Encode each categorical feature
        for feature in categorical_features:
            if feature in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature])
                self.feature_encoders[feature] = le
                print(f"Encoded {feature}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return df_encoded
    
    def process_numerical_features(self, df):
        """Process numerical features (Sexual Activity, Concentration, Optimisim)"""
        print("\nProcessing numerical features...")
        
        df_processed = df.copy()
        
        # Extract numerical values from "X From 10" format
        numerical_features = ['Sexual Activity', 'Concentration', 'Optimisim']
        
        for feature in numerical_features:
            if feature in df_processed.columns:
                # Extract the number before "From 10"
                df_processed[feature] = df_processed[feature].str.extract('(\d+)').astype(int)
                print(f"Processed {feature}: range {df_processed[feature].min()}-{df_processed[feature].max()}")
        
        return df_processed
    
    def prepare_features_and_target(self, df):
        """Prepare features and target for modeling"""
        print("\nPreparing features and target...")
        
        # Features to use (excluding Patient Number and target)
        feature_columns = [
            'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 
            'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
            'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
            'Nervous Break-down', 'Admit Mistakes', 'Overthinking',
            'Sexual Activity', 'Concentration', 'Optimisim'
        ]
        
        X = df[feature_columns]
        y = df['Expert Diagnose']
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        print(f"Target distribution: {np.bincount(y_encoded)}")
        
        return X, y_encoded
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model and plot training/validation loss"""
        print("\nTraining XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Prepare eval set and evals_result
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        evals_result = {}
        
        # Train model with evaluation logging
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        evals_result = self.model.evals_result()
        
        # Plot training and validation loss
        train_loss = evals_result['validation_0']['mlogloss']
        val_loss = evals_result['validation_1']['mlogloss']
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('Training and Validation Loss (XGBoost)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_validation_loss.png', dpi=300)
        plt.close()
        print("Saved training/validation loss plot as "
              "'training_validation_loss.png'.")
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def save_model(self, model_path='mental_health_model.joblib'):
        """Save the trained model and preprocessing components"""
        print(f"\nSaving model to {model_path}...")
        
        # Create a dictionary with all components
        model_components = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_encoders': self.feature_encoders,
            'feature_names': [
                'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 
                'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
                'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
                'Nervous Break-down', 'Admit Mistakes', 'Overthinking',
                'Sexual Activity', 'Concentration', 'Optimisim'
            ],
            'target_classes': self.label_encoder.classes_.tolist()
        }
        
        # Save using joblib
        joblib.dump(model_components, model_path)
        print(f"Model saved successfully to {model_path}")
        
        # Also save model info
        model_info = {
            'model_type': 'XGBoost',
            'n_features': len(model_components['feature_names']),
            'n_classes': len(model_components['target_classes']),
            'feature_names': model_components['feature_names'],
            'target_classes': model_components['target_classes'],
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_info, 'model_info.joblib')
        print("Model info saved to model_info.joblib")
    
    def load_model(self, model_path='mental_health_model.joblib'):
        """Load a previously trained model"""
        print(f"Loading model from {model_path}...")
        
        try:
            model_components = joblib.load(model_path)
            
            self.model = model_components['model']
            self.scaler = model_components['scaler']
            self.label_encoder = model_components['label_encoder']
            self.feature_encoders = model_components['feature_encoders']
            
            print("Model loaded successfully!")
            print(f"Target classes: {self.label_encoder.classes_}")
            print(f"Number of features: {len(model_components['feature_names'])}")
            
            return True
            
        except FileNotFoundError:
            print(f"Model file {model_path} not found!")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single_patient(self, patient_data):
        """Predict mental health diagnosis for a single patient"""
        if self.model is None:
            print("Model not loaded! Please load or train a model first.")
            return None
        
        try:
            # Preprocess the input data
            processed_data = self.preprocess_single_patient(patient_data)
            
            # Scale the features
            scaled_data = self.scaler.transform([processed_data])
            
            # Make prediction
            prediction = self.model.predict(scaled_data)[0]
            prediction_proba = self.model.predict_proba(scaled_data)[0]
            
            # Convert back to original class name
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence scores
            confidence_scores = dict(zip(self.label_encoder.classes_, prediction_proba))
            
            return {
                'prediction': predicted_class,
                'confidence': confidence_scores,
                'prediction_proba': prediction_proba
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def preprocess_single_patient(self, patient_data):
        """Preprocess data for a single patient"""
        # Expected feature names
        feature_names = [
            'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 
            'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
            'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
            'Nervous Break-down', 'Admit Mistakes', 'Overthinking',
            'Sexual Activity', 'Concentration', 'Optimisim'
        ]
        
        processed_features = []
        
        for feature in feature_names:
            if feature in patient_data:
                value = patient_data[feature]
                
                # Handle categorical features
                if feature in self.feature_encoders:
                    if value in self.feature_encoders[feature].classes_:
                        encoded_value = self.feature_encoders[feature].transform([value])[0]
                    else:
                        print(f"Warning: Unknown value '{value}' for feature '{feature}'")
                        encoded_value = 0  # Default value
                    processed_features.append(encoded_value)
                
                # Handle numerical features
                elif feature in ['Sexual Activity', 'Concentration', 'Optimisim']:
                    if isinstance(value, str) and 'From 10' in value:
                        # Extract number from "X From 10" format
                        numeric_value = int(value.split()[0])
                    else:
                        numeric_value = int(value)
                    processed_features.append(numeric_value)
                
                else:
                    processed_features.append(value)
            else:
                print(f"Warning: Missing feature '{feature}', using default value 0")
                processed_features.append(0)
        
        return processed_features
    
    def evaluate_model(self, y_test, y_pred):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("=" * 50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix Shape: {cm.shape}")
        
        return cm
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        plt.title('Confusion Matrix - Mental Health Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Add accuracy text
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        feature_names = [
            'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 
            'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
            'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
            'Nervous Break-down', 'Admit Mistakes', 'Overthinking',
            'Sexual Activity', 'Concentration', 'Optimisim'
        ]
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(importance_df)), importance_df['Importance'])
        
        # Color bars based on importance
        colors = plt.cm.RdYlBu_r(importance_df['Importance'] / importance_df['Importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('XGBoost Feature Importance for Mental Health Detection', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importance_df['Importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_target_distribution(self, df):
        """Plot target distribution"""
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        target_counts = df['Expert Diagnose'].value_counts()
        bars = ax1.bar(target_counts.index, target_counts.values, 
                      color=sns.color_palette("husl", len(target_counts)))
        ax1.set_title('Distribution of Mental Health Diagnoses', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Diagnosis', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, target_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("husl", len(target_counts)))
        ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validation_evaluation(self, X, y):
        """Perform cross-validation evaluation"""
        print("\nCross-Validation Evaluation:")
        print("=" * 40)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def run_complete_analysis(self, file_path):
        """Run complete mental health detection analysis"""
        print("=" * 60)
        print("MENTAL HEALTH DETECTION USING XGBOOST")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)
        
        # Plot target distribution
        self.plot_target_distribution(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df)
        
        # Process numerical features
        df_processed = self.process_numerical_features(df_encoded)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(df_processed)
        
        # Train model
        X_test, y_test, y_pred, y_pred_proba = self.train_xgboost_model(X, y)
        
        # Evaluate model
        cm = self.evaluate_model(y_test, y_pred)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Cross-validation evaluation
        cv_scores = self.cross_validation_evaluation(X, y)
        
        # Save the trained model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("- confusion_matrix.png")
        print("- feature_importance.png") 
        print("- target_distribution.png")
        print("- mental_health_model.joblib (saved model)")
        print("- model_info.joblib (model information)")
        
        return {
            'model': self.model,
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cv_scores,
            'confusion_matrix': cm
        }

def main():
    """Main function to run the mental health detection"""
    # Initialize detector
    detector = MentalHealthDetector()
    
    # Run complete analysis
    results = detector.run_complete_analysis('Dataset-Mental-Disorders.csv')
    
    # Print final summary
    print(f"\nFinal Model Performance:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})")

if __name__ == "__main__":
    main() 