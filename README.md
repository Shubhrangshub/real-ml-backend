# 🤖 AutoML Master

**Automatically train and deploy machine learning models with no coding required!**

AutoML Master is a complete full-stack application that makes machine learning accessible to everyone. Simply upload your CSV data, select your target column, and let the system automatically train multiple ML models and select the best one for you.

## ✨ Features

### 🎯 Training
- **Automatic Model Selection**: Trains 5+ different algorithms in parallel
- **Smart Problem Detection**: Automatically detects classification vs regression
- **Cross-Validation**: Built-in 5-fold cross-validation for robust evaluation
- **Feature Importance**: Visualizes which features matter most
- **Multiple Algorithms**: 
  - Logistic Regression
  - Linear Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting

### 🔮 Predictions
- Real-time predictions on trained models
- Batch prediction support
- Probability scores for classification tasks
- JSON-based API

### 📊 Model Management
- View all trained models
- Delete unwanted models
- Model persistence with MongoDB
- In-memory caching for fast access

### 🎨 Beautiful UI
- Modern gradient design
- Responsive layout
- Interactive charts (Recharts)
- Real-time training progress
- Sample datasets included

## 🏗️ Architecture

### Backend (FastAPI)
- **Location**: `/app/backend/`
- **Port**: 8001
- **Features**:
  - RESTful API
  - Parallel model training
  - MongoDB integration
  - scikit-learn ML pipeline

### Frontend (React)
- **Location**: `/app/frontend/`
- **Port**: 3000
- **Features**:
  - Tailwind CSS styling
  - Axios for API calls
  - Recharts for visualizations
  - Responsive design

### Database (MongoDB)
- Stores trained models
- Training history
- Model metadata

## 🚀 Getting Started

### Prerequisites
All dependencies are already installed! The application is running on:
- Backend: http://localhost:8001
- Frontend: http://localhost:3000

### Quick Start

1. **Open the Application**
   - Navigate to http://localhost:3000 in your browser

2. **Try a Sample Dataset**
   - Click on one of the sample datasets (Loan Approval or House Prices)

3. **Select Target Column**
   - Choose which column you want to predict

4. **Train Models**
   - Click "Train Models" and wait for results

5. **Make Predictions**
   - Go to the "Make Predictions" tab
   - Select your trained model
   - Input data in JSON format
   - Get instant predictions!

## 📖 API Documentation

### Health Check
```bash
GET /api/health
```

### Train Models
```bash
POST /api/train
Content-Type: application/json

{
  "csv_text": "col1,col2,target\n1,2,0\n3,4,1",
  "target_column": "target",
  "algorithm": "auto",
  "problem_type": "auto"
}
```

### Make Predictions
```bash
POST /api/predict
Content-Type: application/json

{
  "model_id": "model-uuid-here",
  "data": [{"col1": 1, "col2": 2}]
}
```

### List Models
```bash
GET /api/models
```

### Delete Model
```bash
DELETE /api/models/{model_id}
```

## 🧪 Testing

### Backend Test
```bash
curl http://localhost:8001/api/health
```

### Train a Model
```bash
curl -X POST http://localhost:8001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "csv_text": "age,income,approved\n25,45000,0\n35,75000,1",
    "target_column": "approved",
    "algorithm": "auto"
  }'
```

## 📁 Project Structure

```
/app/
├── backend/
│   ├── server.py           # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── App.css        # Styles
│   │   └── index.js       # Entry point
│   ├── public/
│   │   └── index.html     # HTML template
│   ├── package.json       # Node dependencies
│   ├── tailwind.config.js # Tailwind configuration
│   └── .env              # Frontend environment
└── sample_data/
    ├── loan_approval.csv  # Sample classification data
    └── house_prices.csv   # Sample regression data
```

## 🔧 Configuration

### Backend Environment Variables
```env
MONGO_URL=mongodb://localhost:27017/
```

### Frontend Environment Variables
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

## 🛠️ Service Management

### Restart Services
```bash
sudo supervisorctl restart all
```

### Check Status
```bash
sudo supervisorctl status
```

### View Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.err.log

# Frontend logs
tail -f /var/log/supervisor/frontend.err.log
```

## 📊 Sample Datasets

### Loan Approval (Classification)
Predict whether a loan application will be approved based on:
- Age
- Income
- Credit Score
- Loan Amount

### House Prices (Regression)
Predict house prices based on:
- Size (sq ft)
- Number of bedrooms
- Age of house
- Location score

## 🎯 Use Cases

1. **Financial Services**: Credit risk assessment, fraud detection
2. **Real Estate**: Price prediction, property valuation
3. **Healthcare**: Disease prediction, patient risk scoring
4. **Marketing**: Customer churn prediction, campaign optimization
5. **Education**: Student performance prediction

## 🔐 Features in Detail

### Automatic Problem Type Detection
The system automatically determines if your problem is:
- **Classification**: Predicting categories (approved/rejected, yes/no)
- **Regression**: Predicting continuous values (prices, scores)

### Feature Engineering
- Automatic handling of missing values
- One-hot encoding for categorical variables
- Feature scaling (when appropriate)

### Model Evaluation
- Cross-validation scores
- Multiple metrics (Accuracy, F1, R², MAE, RMSE)
- Training time tracking
- Feature importance rankings

## 🎨 UI Components

### Training Dashboard
- CSV upload area
- Column selector
- Algorithm chooser
- Progress indicator
- Results visualization

### Prediction Interface
- Model selector
- JSON input area
- Prediction results
- Probability scores

### Model Management
- Card-based model display
- Quick delete functionality
- Model metadata viewing

## 🚀 Performance

- **Parallel Training**: All models train simultaneously
- **Fast Inference**: In-memory model caching
- **Efficient Storage**: MongoDB for persistence
- **Hot Reload**: Instant code updates during development

## 📝 Notes

- Models are stored both in-memory and in MongoDB
- The system supports both file upload and direct CSV paste
- All training happens in the backend with scikit-learn
- Frontend is fully responsive and mobile-friendly

## 🎉 Success!

Your AutoML Master application is now fully operational! Visit http://localhost:3000 to start training models.

---

Built with ❤️ using FastAPI, React, MongoDB, and scikit-learn
