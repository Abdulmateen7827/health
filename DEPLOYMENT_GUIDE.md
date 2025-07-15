# 🚀 Streamlit Mental Health Detection App - Deployment Guide

This guide will help you deploy the beautiful Streamlit mental health detection application.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Trained Model**: Run the training script to generate `mental_health_model.joblib`
2. **Python Environment**: Python 3.8+ with all dependencies installed
3. **Dataset**: `Dataset-Mental-Disorders.csv` in the project directory

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python mental_health_detection.py
```

This will generate:
- `mental_health_model.joblib` - The trained model
- `model_info.joblib` - Model metadata
- Visualization files (PNG)

### 3. Verify Model Files

Ensure these files exist in your project directory:
- ✅ `mental_health_model.joblib`
- ✅ `model_info.joblib`
- ✅ `Dataset-Mental-Disorders.csv`

## 🚀 Running the Streamlit App

### Local Development

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Production Deployment

#### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**: Upload your project to a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and set the path to `streamlit_app.py`
3. **Deploy**: Click "Deploy" and wait for the build to complete

#### Option 2: Heroku

1. **Create `Procfile`**:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `setup.sh`**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy Streamlit app"
   git push heroku main
   ```

#### Option 3: Docker

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t mental-health-app .
   docker run -p 8501:8501 mental-health-app
   ```

## 🎨 App Features

### 🔍 Patient Assessment Tab
- **17 Input Fields**: All behavioral symptoms with intuitive controls
- **Categorical Inputs**: Dropdown menus for mood and behavioral symptoms
- **Slider Controls**: 1-10 scale for functional assessments
- **Real-time Prediction**: Instant results with confidence scores
- **Beautiful Visualizations**: Interactive charts using Plotly

### 📈 Model Performance Tab
- **Model Metrics**: Training date, features, classes
- **Feature Importance**: Interactive horizontal bar chart
- **Prediction History**: Track previous assessments
- **Analytics Dashboard**: Comprehensive model insights

### ℹ️ About Tab
- **System Overview**: Complete documentation
- **Technical Details**: Framework and algorithm information
- **Usage Guidelines**: Best practices and disclaimers
- **Medical Disclaimers**: Important safety information

## 🎯 Key Features

### ✨ Beautiful UI/UX
- **Modern Design**: Clean, professional interface
- **Responsive Layout**: Works on desktop and mobile
- **Color-coded Sections**: Easy navigation and understanding
- **Interactive Elements**: Hover effects and smooth transitions

### 📊 Advanced Visualizations
- **Confidence Charts**: Bar charts showing prediction confidence
- **Feature Importance**: Horizontal bar charts with color gradients
- **Real-time Updates**: Dynamic charts that update with predictions

### 🔒 Safety & Ethics
- **Medical Disclaimers**: Clear warnings about research use only
- **Professional Guidance**: Emphasis on consulting mental health professionals
- **Confidentiality**: Session-based storage (no permanent patient data)

## 🛡️ Security Considerations

### Data Privacy
- **No Data Storage**: Predictions are not permanently stored
- **Session-based**: Data only exists during the session
- **Local Processing**: All computations happen locally

### Medical Ethics
- **Research Tool**: Clearly marked as demonstration only
- **Professional Referral**: Always recommend professional consultation
- **Disclaimer Prominent**: Medical disclaimers visible throughout

## 🔧 Configuration Options

### Customizing the App

You can modify `streamlit_app.py` to:

1. **Change Colors**: Update CSS variables in the style section
2. **Add Features**: Include additional input fields or visualizations
3. **Modify Layout**: Adjust column layouts and spacing
4. **Custom Branding**: Add your organization's logo and colors

### Environment Variables

Set these environment variables for production:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
```

## 📱 Mobile Optimization

The app is optimized for mobile devices with:
- **Responsive Design**: Adapts to different screen sizes
- **Touch-friendly**: Large buttons and sliders
- **Fast Loading**: Optimized for mobile networks
- **Offline Capable**: Works without internet after initial load

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Ensure `mental_health_model.joblib` exists
   - Run training script first: `python mental_health_detection.py`

2. **Dependencies Missing**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Port Already in Use**:
   - Change port: `streamlit run streamlit_app.py --server.port=8502`
   - Kill existing process: `lsof -ti:8501 | xargs kill -9`

4. **Memory Issues**:
   - Reduce model complexity in training
   - Use smaller dataset for testing

### Performance Optimization

1. **Model Caching**: Uses `@st.cache_resource` for efficient loading
2. **Lazy Loading**: Components load only when needed
3. **Efficient Charts**: Plotly charts optimized for web

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are correctly installed
4. Verify model files are present and valid

## 🎉 Success!

Your beautiful Streamlit mental health detection app is now ready for deployment! 

The app provides:
- ✅ Professional, intuitive interface
- ✅ All 17 behavioral symptom inputs
- ✅ Real-time predictions with confidence scores
- ✅ Beautiful visualizations and analytics
- ✅ Mobile-responsive design
- ✅ Comprehensive documentation and disclaimers

Remember: This is a research/demonstration tool and should not be used for actual medical diagnosis. 