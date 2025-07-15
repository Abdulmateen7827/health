#!/usr/bin/env python3
"""
Quick Start Script for Mental Health Detection Streamlit App
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'sklearn', 'xgboost', 'joblib', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'sklearn':
            spec = importlib.util.find_spec('sklearn')
        else:
            spec = importlib.util.find_spec(package)
        
        if spec is None:
            missing_packages.append(package)
    
    return missing_packages

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        'mental_health_model.joblib',
        'model_info.joblib',
        'Dataset-Mental-Disorders.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def install_dependencies():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return False

def train_model():
    """Train the model if it doesn't exist"""
    print("Training the mental health detection model...")
    try:
        subprocess.check_call([sys.executable, "mental_health_detection.py"])
        print("✅ Model trained successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to train model. Please run manually:")
        print("python mental_health_detection.py")
        return False

def launch_streamlit():
    """Launch the Streamlit app"""
    print("🚀 Launching Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main function"""
    print("🧠 Mental Health Detection - Quick Start")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please install dependencies manually: pip install -r requirements.txt")
            return
    else:
        print("✅ All dependencies are installed!")
    
    # Check model files
    print("\nChecking model files...")
    missing_files = check_model_files()
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        
        if 'mental_health_model.joblib' in missing_files:
            response = input("Would you like to train the model now? (y/n): ")
            if response.lower() == 'y':
                if not train_model():
                    return
            else:
                print("Please train the model manually: python mental_health_detection.py")
                return
        
        if 'Dataset-Mental-Disorders.csv' in missing_files:
            print("❌ Dataset file not found. Please ensure 'Dataset-Mental-Disorders.csv' is in the current directory.")
            return
    else:
        print("✅ All model files are present!")
    
    # Launch app
    print("\n🎉 Everything is ready!")
    launch_streamlit()

if __name__ == "__main__":
    main() 