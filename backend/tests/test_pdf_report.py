"""
Test PDF Report Generation - Backend API Tests
Tests deploy endpoints needed for PDF report generation
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://automl-validation.preview.emergentagent.com').rstrip('/')

# Test credentials
TEST_EMAIL = "test@automl.com"
TEST_PASSWORD = "Test1234!"


class TestDeployEndpoints:
    """Test deploy endpoints used by PDF report generation"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and get auth token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mongodb"] == "connected"
        print("PASS: Health check - API is healthy")
    
    def test_get_deployments_authenticated(self):
        """Test GET /api/deploy returns deployments list"""
        response = requests.get(f"{BASE_URL}/api/deploy", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "deployments" in data
        assert isinstance(data["deployments"], list)
        print(f"PASS: GET /api/deploy - Found {len(data['deployments'])} deployments")
    
    def test_get_deployments_requires_auth(self):
        """Test GET /api/deploy requires authentication"""
        response = requests.get(f"{BASE_URL}/api/deploy")
        assert response.status_code == 401
        print("PASS: GET /api/deploy requires authentication (401)")
    
    def test_post_deploy_requires_auth(self):
        """Test POST /api/deploy requires authentication"""
        response = requests.post(f"{BASE_URL}/api/deploy", json={
            "model_id": "test-model",
            "model_data": {},
            "name": "Test Model"
        })
        assert response.status_code == 401
        print("PASS: POST /api/deploy requires authentication (401)")
    
    def test_create_and_delete_deployment(self):
        """Test creating and deleting a deployment"""
        # Create deployment
        create_response = requests.post(f"{BASE_URL}/api/deploy", headers=self.headers, json={
            "model_id": "test-pdf-model",
            "model_data": {
                "algorithm": "logistic_regression",
                "problemType": "classification",
                "targetColumn": "Approved",
                "metrics": {"accuracy": 0.85, "f1": 0.82},
                "modelData": {
                    "numericCols": ["Income", "Age"],
                    "categoricalCols": ["Employment"],
                    "encodingMap": {}
                }
            },
            "name": "PDF Test Model",
            "description": "Test deployment for PDF report"
        })
        assert create_response.status_code == 200
        data = create_response.json()
        assert data["status"] == "success"
        assert "deploy_id" in data
        deploy_id = data["deploy_id"]
        print(f"PASS: POST /api/deploy - Created deployment {deploy_id}")
        
        # Verify deployment appears in list
        list_response = requests.get(f"{BASE_URL}/api/deploy", headers=self.headers)
        assert list_response.status_code == 200
        deployments = list_response.json()["deployments"]
        deploy_ids = [d["deploy_id"] for d in deployments]
        assert deploy_id in deploy_ids
        print(f"PASS: Deployment {deploy_id} appears in list")
        
        # Delete deployment
        delete_response = requests.delete(f"{BASE_URL}/api/deploy/{deploy_id}", headers=self.headers)
        assert delete_response.status_code == 200
        print(f"PASS: DELETE /api/deploy/{deploy_id} - Deployment deleted")
    
    def test_sample_data_available(self):
        """Test sample datasets are available for loading"""
        datasets = [
            "/api/sample_data/loan_approval.csv",
            "/api/sample_data/house_prices.csv",
            "/api/sample_data/insurance.csv",
            "/api/sample_data/customer_churn.csv",
            "/api/sample_data/customer_segmentation.csv"
        ]
        for dataset in datasets:
            response = requests.get(f"{BASE_URL}{dataset}")
            assert response.status_code == 200, f"Failed to load {dataset}"
            lines = response.text.strip().split('\n')
            assert len(lines) > 100, f"{dataset} has too few lines"
            print(f"PASS: {dataset} - {len(lines)} lines")


class TestLeaderboardEndpoints:
    """Test leaderboard endpoints used by PDF report"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and get auth token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 200
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def test_get_leaderboard(self):
        """Test GET /api/leaderboard returns entries"""
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)
        print(f"PASS: GET /api/leaderboard - Found {len(data['entries'])} entries")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
