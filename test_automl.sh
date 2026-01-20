#!/bin/bash

echo "🤖 AutoML Master - Quick Test Script"
echo "======================================"
echo ""

# Test 1: Health Check
echo "1️⃣ Testing Backend Health..."
curl -s http://localhost:8001/api/health | python -m json.tool
echo ""
echo ""

# Test 2: Train a Classification Model
echo "2️⃣ Training Classification Model (Loan Approval)..."
TRAIN_RESULT=$(curl -s -X POST http://localhost:8001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "csv_text": "age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1\n26,48000,660,12000,0\n40,92000,770,45000,1",
    "target_column": "approved",
    "algorithm": "random_forest"
  }')

echo "$TRAIN_RESULT" | python -m json.tool | head -40
echo ""

# Extract best model ID
MODEL_ID=$(echo "$TRAIN_RESULT" | python -c "import json,sys; print(json.load(sys.stdin)['bestModel']['modelId'])" 2>/dev/null)

echo ""
echo "✅ Best Model ID: $MODEL_ID"
echo ""
echo ""

# Test 3: Make a Prediction
if [ ! -z "$MODEL_ID" ]; then
    echo "3️⃣ Making Prediction with Best Model..."
    curl -s -X POST http://localhost:8001/api/predict \
      -H "Content-Type: application/json" \
      -d "{
        \"model_id\": \"$MODEL_ID\",
        \"data\": [
          {\"age\": 30, \"income\": 60000, \"credit_score\": 700, \"loan_amount\": 20000},
          {\"age\": 22, \"income\": 35000, \"credit_score\": 600, \"loan_amount\": 5000}
        ]
      }" | python -m json.tool
    echo ""
    echo ""
fi

# Test 4: List All Models
echo "4️⃣ Listing All Models..."
curl -s http://localhost:8001/api/models | python -m json.tool
echo ""
echo ""

echo "✅ All Tests Complete!"
echo ""
echo "🌐 Open http://localhost:3000 to use the web interface"
