# 🚀 Quick Start Guide - AutoML Master

## ✅ Application Status: READY

### Services Running:
- ✅ Backend API: http://localhost:8001
- ✅ Frontend UI: http://localhost:3000
- ✅ MongoDB: Connected and running
- ✅ All tests passed: 100%

---

## 🎯 How to Use

### Option 1: Use the Web Interface (Recommended)

1. **Open your browser**: http://localhost:3000

2. **Try a Sample Dataset**:
   - Click "Loan Approval (Classification)" or "House Prices (Regression)"
   - Data will automatically load

3. **Train Models**:
   - Select your target column (e.g., "approved" or "price")
   - Choose algorithm or leave on "Auto (Try All)"
   - Click "🚀 Train Models"
   - Wait ~2-5 seconds for results

4. **View Results**:
   - See best model metrics
   - Check feature importance chart
   - Review leaderboard of all models

5. **Make Predictions**:
   - Go to "🔮 Make Predictions" tab
   - Select a trained model
   - Enter data in JSON format:
     ```json
     [{"age": 30, "income": 60000, "credit_score": 700, "loan_amount": 20000}]
     ```
   - Click "Predict"

6. **Manage Models**:
   - Go to "📊 My Models" tab
   - View all trained models
   - Delete unwanted models

---

### Option 2: Use the API (For Developers)

#### 1. Health Check
```bash
curl http://localhost:8001/api/health
```

#### 2. Train a Model
```bash
curl -X POST http://localhost:8001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "csv_text": "age,income,approved\n25,45000,0\n35,75000,1\n45,95000,1",
    "target_column": "approved",
    "algorithm": "auto"
  }'
```

#### 3. Make Predictions
```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "YOUR_MODEL_ID",
    "data": [{"age": 30, "income": 60000}]
  }'
```

#### 4. List Models
```bash
curl http://localhost:8001/api/models
```

---

## 🧪 Run Tests

### Quick Test Script
```bash
bash /app/test_automl.sh
```

### Backend Test Suite
```bash
cd /app && python backend_test.py
```

---

## 🔧 Service Management

### Check Status
```bash
sudo supervisorctl status
```

### Restart Services
```bash
# Restart all
sudo supervisorctl restart all

# Restart specific service
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
```

### View Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.err.log

# Frontend logs
tail -f /var/log/supervisor/frontend.err.log
```

---

## 📊 Sample Datasets Included

### 1. Loan Approval (Classification)
- **Target**: approved (0/1)
- **Features**: age, income, credit_score, loan_amount
- **Use Case**: Predict loan approval

### 2. House Prices (Regression)
- **Target**: price
- **Features**: size, bedrooms, age, location_score
- **Use Case**: Predict house prices

---

## 🎨 Features

### Backend
- ✅ 5+ ML algorithms (Logistic, Linear, Decision Tree, Random Forest, Gradient Boosting)
- ✅ Parallel training (all models train simultaneously)
- ✅ Cross-validation (5-fold)
- ✅ Feature importance extraction
- ✅ MongoDB persistence
- ✅ RESTful API

### Frontend
- ✅ Beautiful gradient UI
- ✅ CSV upload & paste
- ✅ Interactive charts (Recharts)
- ✅ Real-time training progress
- ✅ Model management
- ✅ Responsive design

---

## 📁 Project Structure

```
/app/
├── backend/
│   ├── server.py              # FastAPI application
│   ├── requirements.txt       # Dependencies
│   └── .env                   # Configuration
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React app
│   │   └── index.js          # Entry point
│   ├── package.json          # Dependencies
│   └── tailwind.config.js    # Styling
├── sample_data/
│   ├── loan_approval.csv     # Sample data
│   └── house_prices.csv      # Sample data
├── test_automl.sh            # Quick test script
├── backend_test.py           # Backend tests
└── README.md                 # Full documentation
```

---

## 🎉 You're All Set!

Visit **http://localhost:3000** to start using AutoML Master!

Need help? Check `/app/README.md` for full documentation.
