import requests
import json
import sys
import time
from datetime import datetime

class AutoMLAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.trained_model_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        return success

    def test_train_classification(self):
        """Test training with classification data"""
        loan_data = """age,income,credit_score,loan_amount,approved
25,45000,650,10000,0
35,75000,720,25000,1
45,95000,780,50000,1
28,52000,680,15000,0
52,120000,800,75000,1
23,38000,620,8000,0
38,82000,740,30000,1
42,88000,760,40000,1
30,62000,700,20000,1
48,105000,790,60000,1"""

        success, response = self.run_test(
            "Train Classification Model",
            "POST",
            "api/train",
            200,
            data={
                "csv_text": loan_data,
                "target_column": "approved",
                "algorithm": "auto",
                "problem_type": "auto"
            },
            timeout=60
        )
        
        if success and response.get("status") == "success":
            self.trained_model_id = response.get("bestModel", {}).get("modelId")
            print(f"   Trained model ID: {self.trained_model_id}")
        
        return success

    def test_train_regression(self):
        """Test training with regression data"""
        house_data = """size,bedrooms,age,location_score,price
1200,2,5,7,250000
1800,3,10,8,380000
2500,4,3,9,520000
1000,1,15,6,180000
2200,3,7,8,450000
1500,2,12,7,280000
3000,5,2,9,680000
1100,2,8,6,220000
2000,3,5,8,420000
2800,4,4,9,610000"""

        success, response = self.run_test(
            "Train Regression Model",
            "POST",
            "api/train",
            200,
            data={
                "csv_text": house_data,
                "target_column": "price",
                "algorithm": "random_forest",
                "problem_type": "regression"
            },
            timeout=60
        )
        return success

    def test_list_models(self):
        """Test listing all models"""
        success, response = self.run_test(
            "List Models",
            "GET",
            "api/models",
            200
        )
        
        if success:
            models_count = response.get("count", 0)
            print(f"   Found {models_count} models")
        
        return success

    def test_predict(self):
        """Test making predictions"""
        if not self.trained_model_id:
            print("❌ Skipping prediction test - no trained model available")
            return False

        prediction_data = [
            {"age": 30, "income": 60000, "credit_score": 720, "loan_amount": 25000}
        ]

        success, response = self.run_test(
            "Make Prediction",
            "POST",
            "api/predict",
            200,
            data={
                "model_id": self.trained_model_id,
                "data": prediction_data
            }
        )
        return success

    def test_delete_model(self):
        """Test deleting a model"""
        if not self.trained_model_id:
            print("❌ Skipping delete test - no trained model available")
            return False

        success, response = self.run_test(
            "Delete Model",
            "DELETE",
            f"api/models/{self.trained_model_id}",
            200
        )
        return success

    def test_error_cases(self):
        """Test various error scenarios"""
        print("\n🔍 Testing Error Cases...")
        
        # Test training without data
        success1, _ = self.run_test(
            "Train Without Data",
            "POST",
            "api/train",
            200,  # Backend returns 200 with error message
            data={"target_column": "test"}
        )
        
        # Test prediction with invalid model ID
        success2, _ = self.run_test(
            "Predict Invalid Model",
            "POST",
            "api/predict",
            404,
            data={
                "model_id": "invalid-model-id",
                "data": [{"test": 1}]
            }
        )
        
        # Test delete invalid model
        success3, _ = self.run_test(
            "Delete Invalid Model",
            "DELETE",
            "api/models/invalid-model-id",
            404
        )
        
        return success1 or success2 or success3  # At least one should work as expected

def main():
    print("🚀 Starting AutoML Backend API Tests")
    print("=" * 50)
    
    tester = AutoMLAPITester()
    
    # Run all tests
    tests = [
        tester.test_health_check,
        tester.test_train_classification,
        tester.test_list_models,
        tester.test_predict,
        tester.test_train_regression,
        tester.test_delete_model,
        tester.test_error_cases
    ]
    
    for test in tests:
        try:
            test()
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())