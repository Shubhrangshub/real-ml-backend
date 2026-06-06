"""
Test suite for Model Deployment and What-If features
Tests: Sample data endpoints, Deploy endpoints, Public predict endpoints
"""
import pytest
import requests
import os
import base64
import pickle
import numpy as np

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

# Test credentials
ADMIN_EMAIL = "shubhrangshub@gmail.com"
ADMIN_PASSWORD = "MyNewPass123!"
TEST_EMAIL = "test@automl.com"
TEST_PASSWORD = "Test1234!"


class TestSampleDataEndpoints:
    """Test sample data CSV endpoints - 1000+ row datasets"""
    
    def test_loan_approval_csv_returns_1201_lines(self):
        """GET /api/sample_data/loan_approval.csv - should return 1201 lines (1200 rows + header)"""
        response = requests.get(f"{BASE_URL}/api/sample_data/loan_approval.csv", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        lines = response.text.strip().split('\n')
        assert len(lines) == 1201, f"Expected 1201 lines, got {len(lines)}"
        # Verify header has expected columns
        header = lines[0].split(',')
        assert len(header) >= 10, f"Expected at least 10 columns, got {len(header)}"
        print(f"PASS: loan_approval.csv has {len(lines)} lines with {len(header)} columns")
    
    def test_customer_churn_csv_returns_1501_lines(self):
        """GET /api/sample_data/customer_churn.csv - should return 1501 lines (1500 rows + header)"""
        response = requests.get(f"{BASE_URL}/api/sample_data/customer_churn.csv", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        lines = response.text.strip().split('\n')
        assert len(lines) == 1501, f"Expected 1501 lines, got {len(lines)}"
        print(f"PASS: customer_churn.csv has {len(lines)} lines")
    
    def test_house_prices_csv_returns_1001_lines(self):
        """GET /api/sample_data/house_prices.csv - should return 1001 lines"""
        response = requests.get(f"{BASE_URL}/api/sample_data/house_prices.csv", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        lines = response.text.strip().split('\n')
        assert len(lines) == 1001, f"Expected 1001 lines, got {len(lines)}"
        print(f"PASS: house_prices.csv has {len(lines)} lines")
    
    def test_insurance_csv_returns_1101_lines(self):
        """GET /api/sample_data/insurance.csv - should return 1101 lines"""
        response = requests.get(f"{BASE_URL}/api/sample_data/insurance.csv", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        lines = response.text.strip().split('\n')
        assert len(lines) == 1101, f"Expected 1101 lines, got {len(lines)}"
        print(f"PASS: insurance.csv has {len(lines)} lines")
    
    def test_customer_segmentation_csv_returns_1001_lines(self):
        """GET /api/sample_data/customer_segmentation.csv - should return 1001 lines"""
        response = requests.get(f"{BASE_URL}/api/sample_data/customer_segmentation.csv", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        lines = response.text.strip().split('\n')
        assert len(lines) == 1001, f"Expected 1001 lines, got {len(lines)}"
        print(f"PASS: customer_segmentation.csv has {len(lines)} lines")


class TestDeployEndpoints:
    """Test model deployment endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and get auth token"""
        self.session = requests.Session()
        login_response = self.session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if login_response.status_code == 200:
            data = login_response.json()
            self.token = data.get("token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            pytest.skip("Login failed - skipping authenticated tests")
    
    def test_deploy_model_creates_deployment(self):
        """POST /api/deploy - deploy a model and get deploy_id"""
        # Create a simple mock model for deployment
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])
        model_bytes = base64.b64encode(pickle.dumps(model)).decode('utf-8')
        
        deploy_data = {
            "model_id": "test_model_123",
            "name": "Test Deployment",
            "description": "Test deployment for API testing",
            "model_data": {
                "algorithm": "logistic_regression",
                "problemType": "classification",
                "features": ["feature1", "feature2"],
                "targetColumn": "target",
                "model_bytes": model_bytes,
                "metrics": {"accuracy": 0.95}
            }
        }
        
        response = self.session.post(f"{BASE_URL}/api/deploy", json=deploy_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "deploy_id" in data, "Response should contain deploy_id"
        assert data["status"] == "success"
        self.deploy_id = data["deploy_id"]
        print(f"PASS: Model deployed with ID: {self.deploy_id}")
        return self.deploy_id
    
    def test_list_deployments_returns_user_deployments(self):
        """GET /api/deploy - list deployments for current user"""
        response = self.session.get(f"{BASE_URL}/api/deploy")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "deployments" in data, "Response should contain deployments array"
        assert isinstance(data["deployments"], list)
        print(f"PASS: Listed {len(data['deployments'])} deployments")
    
    def test_deploy_requires_auth(self):
        """POST /api/deploy - should return 401 without auth"""
        response = requests.post(f"{BASE_URL}/api/deploy", json={
            "model_id": "test",
            "model_data": {}
        })
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("PASS: Deploy endpoint requires authentication")
    
    def test_list_deployments_requires_auth(self):
        """GET /api/deploy - should return 401 without auth"""
        response = requests.get(f"{BASE_URL}/api/deploy")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("PASS: List deployments requires authentication")


class TestPublicPredictEndpoints:
    """Test public prediction endpoints (no auth required)"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and create a deployment for testing"""
        self.session = requests.Session()
        login_response = self.session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if login_response.status_code != 200:
            pytest.skip("Login failed - skipping tests")
        
        data = login_response.json()
        self.token = data.get("token")
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        
        # Create a deployment for testing
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])
        model_bytes = base64.b64encode(pickle.dumps(model)).decode('utf-8')
        
        deploy_data = {
            "model_id": f"test_public_model_{os.urandom(4).hex()}",
            "name": "Public Test Model",
            "description": "Model for public prediction testing",
            "model_data": {
                "algorithm": "logistic_regression",
                "problemType": "classification",
                "features": ["feature1", "feature2"],
                "targetColumn": "target",
                "model_bytes": model_bytes,
                "metrics": {"accuracy": 0.95}
            }
        }
        
        response = self.session.post(f"{BASE_URL}/api/deploy", json=deploy_data)
        if response.status_code == 200:
            self.deploy_id = response.json().get("deploy_id")
        else:
            pytest.skip("Failed to create deployment for testing")
    
    def test_get_public_model_info_no_auth(self):
        """GET /api/public/model/{deploy_id} - public model info without auth"""
        response = requests.get(f"{BASE_URL}/api/public/model/{self.deploy_id}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "deploy_id" in data
        assert "name" in data
        assert "algorithm" in data
        assert "features" in data
        print(f"PASS: Public model info retrieved for {self.deploy_id}")
    
    def test_public_predict_no_auth(self):
        """POST /api/public/predict/{deploy_id} - make prediction without auth"""
        response = requests.post(
            f"{BASE_URL}/api/public/predict/{self.deploy_id}",
            json={"features": {"feature1": 0.5, "feature2": 0.5}}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data
        print(f"PASS: Public prediction returned: {data.get('prediction')}")
    
    def test_public_model_info_returns_404_for_invalid_id(self):
        """GET /api/public/model/{invalid_id} - should return 404"""
        response = requests.get(f"{BASE_URL}/api/public/model/invalid_deploy_id_12345")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("PASS: Invalid deploy_id returns 404")
    
    def test_public_predict_returns_404_for_invalid_id(self):
        """POST /api/public/predict/{invalid_id} - should return 404"""
        response = requests.post(
            f"{BASE_URL}/api/public/predict/invalid_deploy_id_12345",
            json={"features": {}}
        )
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("PASS: Invalid deploy_id for prediction returns 404")


class TestDeploymentManagement:
    """Test deployment toggle and delete operations"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and create a deployment for testing"""
        self.session = requests.Session()
        login_response = self.session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if login_response.status_code != 200:
            pytest.skip("Login failed")
        
        data = login_response.json()
        self.token = data.get("token")
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        
        # Create a deployment
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])
        model_bytes = base64.b64encode(pickle.dumps(model)).decode('utf-8')
        
        deploy_data = {
            "model_id": f"test_mgmt_model_{os.urandom(4).hex()}",
            "name": "Management Test Model",
            "description": "Model for management testing",
            "model_data": {
                "algorithm": "logistic_regression",
                "problemType": "classification",
                "features": ["feature1", "feature2"],
                "targetColumn": "target",
                "model_bytes": model_bytes
            }
        }
        
        response = self.session.post(f"{BASE_URL}/api/deploy", json=deploy_data)
        if response.status_code == 200:
            self.deploy_id = response.json().get("deploy_id")
        else:
            pytest.skip("Failed to create deployment")
    
    def test_disable_deployment(self):
        """PATCH /api/deploy/{deploy_id} - disable a deployment"""
        response = self.session.patch(
            f"{BASE_URL}/api/deploy/{self.deploy_id}",
            json={"enabled": False}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "success"
        print(f"PASS: Deployment {self.deploy_id} disabled")
        
        # Verify disabled model returns 403 on public access
        public_response = requests.get(f"{BASE_URL}/api/public/model/{self.deploy_id}")
        assert public_response.status_code == 403, f"Expected 403 for disabled model, got {public_response.status_code}"
        print("PASS: Disabled model returns 403 on public access")
    
    def test_enable_deployment(self):
        """PATCH /api/deploy/{deploy_id} - enable a deployment"""
        # First disable
        self.session.patch(f"{BASE_URL}/api/deploy/{self.deploy_id}", json={"enabled": False})
        
        # Then enable
        response = self.session.patch(
            f"{BASE_URL}/api/deploy/{self.deploy_id}",
            json={"enabled": True}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print(f"PASS: Deployment {self.deploy_id} re-enabled")
    
    def test_delete_deployment(self):
        """DELETE /api/deploy/{deploy_id} - remove a deployment"""
        response = self.session.delete(f"{BASE_URL}/api/deploy/{self.deploy_id}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "success"
        print(f"PASS: Deployment {self.deploy_id} deleted")
        
        # Verify deleted model returns 404
        public_response = requests.get(f"{BASE_URL}/api/public/model/{self.deploy_id}")
        assert public_response.status_code == 404, f"Expected 404 for deleted model, got {public_response.status_code}"
        print("PASS: Deleted model returns 404")
    
    def test_delete_requires_auth(self):
        """DELETE /api/deploy/{deploy_id} - should require auth"""
        response = requests.delete(f"{BASE_URL}/api/deploy/{self.deploy_id}")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("PASS: Delete deployment requires authentication")
    
    def test_toggle_requires_auth(self):
        """PATCH /api/deploy/{deploy_id} - should require auth"""
        response = requests.patch(
            f"{BASE_URL}/api/deploy/{self.deploy_id}",
            json={"enabled": False}
        )
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("PASS: Toggle deployment requires authentication")


class TestHealthEndpoint:
    """Basic health check"""
    
    def test_health_endpoint(self):
        """GET /api/health - should return healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"PASS: Health check - MongoDB: {data.get('mongodb')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
