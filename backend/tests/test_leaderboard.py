"""
Test suite for Leaderboard API endpoints
Tests: POST /api/leaderboard, GET /api/leaderboard, DELETE /api/leaderboard/{model_id}, DELETE /api/leaderboard
"""
import pytest
import requests
import os
import uuid

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

# Test credentials from test_credentials.md
TEST_EMAIL = "shubhrangshub@gmail.com"
TEST_PASSWORD = "MyNewPass123!"


class TestLeaderboardAPI:
    """Leaderboard CRUD endpoint tests"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Get auth token before each test"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        self.token = data.get("token")
        assert self.token, "No token returned from login"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
    
    def test_01_health_check(self):
        """Test health endpoint is accessible"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert "mongodb" in data
        print(f"Health check passed: {data}")
    
    def test_02_get_leaderboard_empty_or_existing(self):
        """Test GET /api/leaderboard returns entries list"""
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"
        assert "entries" in data
        assert isinstance(data["entries"], list)
        print(f"GET leaderboard: {len(data['entries'])} entries found")
    
    def test_03_save_leaderboard_entry(self):
        """Test POST /api/leaderboard saves a model entry"""
        model_id = f"TEST_model_{uuid.uuid4().hex[:8]}"
        entry = {
            "model_id": model_id,
            "algorithm": "logistic_regression",
            "problem_type": "classification",
            "dataset_name": "TEST_Loan_Approval",
            "target_column": "approved",
            "metrics": {"accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.92},
            "feature_importance": [{"feature": "income", "importance": 0.5}],
            "duration_sec": 1.5,
            "eval_mode": "cv",
            "num_features": 10,
            "num_samples": 100
        }
        response = requests.post(f"{BASE_URL}/api/leaderboard", json=entry, headers=self.headers)
        assert response.status_code == 200, f"Save failed: {response.text}"
        data = response.json()
        assert data.get("status") == "success"
        assert "Model saved to leaderboard" in data.get("message", "")
        print(f"Saved entry with model_id: {model_id}")
        
        # Verify entry was saved by fetching leaderboard
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        assert response.status_code == 200
        entries = response.json().get("entries", [])
        found = any(e.get("model_id") == model_id for e in entries)
        assert found, f"Entry {model_id} not found in leaderboard after save"
        print(f"Verified entry {model_id} exists in leaderboard")
        
        # Store for cleanup
        self.__class__.test_model_id = model_id
    
    def test_04_save_multiple_entries(self):
        """Test saving multiple leaderboard entries"""
        model_ids = []
        for algo in ["random_forest", "decision_tree"]:
            model_id = f"TEST_model_{uuid.uuid4().hex[:8]}"
            entry = {
                "model_id": model_id,
                "algorithm": algo,
                "problem_type": "classification",
                "dataset_name": "TEST_Loan_Approval",
                "target_column": "approved",
                "metrics": {"accuracy": 0.90 if algo == "random_forest" else 0.85, "f1": 0.89},
                "duration_sec": 2.0,
                "eval_mode": "split",
                "num_features": 10,
                "num_samples": 100
            }
            response = requests.post(f"{BASE_URL}/api/leaderboard", json=entry, headers=self.headers)
            assert response.status_code == 200
            model_ids.append(model_id)
        
        print(f"Saved {len(model_ids)} additional entries")
        self.__class__.additional_model_ids = model_ids
    
    def test_05_delete_single_entry(self):
        """Test DELETE /api/leaderboard/{model_id} removes one entry"""
        # Use one of the additional model IDs
        if hasattr(self.__class__, 'additional_model_ids') and self.__class__.additional_model_ids:
            model_id = self.__class__.additional_model_ids[0]
        else:
            pytest.skip("No test model ID available for deletion")
        
        response = requests.delete(f"{BASE_URL}/api/leaderboard/{model_id}", headers=self.headers)
        assert response.status_code == 200, f"Delete failed: {response.text}"
        data = response.json()
        assert data.get("status") == "success"
        print(f"Deleted entry: {model_id}")
        
        # Verify entry was deleted
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        entries = response.json().get("entries", [])
        found = any(e.get("model_id") == model_id for e in entries)
        assert not found, f"Entry {model_id} still exists after delete"
        print(f"Verified entry {model_id} no longer exists")
    
    def test_06_delete_nonexistent_entry(self):
        """Test DELETE /api/leaderboard/{model_id} returns 404 for nonexistent entry"""
        fake_id = f"nonexistent_{uuid.uuid4().hex[:8]}"
        response = requests.delete(f"{BASE_URL}/api/leaderboard/{fake_id}", headers=self.headers)
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print(f"Correctly returned 404 for nonexistent entry")
    
    def test_07_clear_all_leaderboard(self):
        """Test DELETE /api/leaderboard clears all entries"""
        # First ensure there are entries
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        initial_count = len(response.json().get("entries", []))
        print(f"Initial entry count: {initial_count}")
        
        # Clear all
        response = requests.delete(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        assert response.status_code == 200, f"Clear failed: {response.text}"
        data = response.json()
        assert data.get("status") == "success"
        print("Clear all request successful")
        
        # Verify all entries are gone
        response = requests.get(f"{BASE_URL}/api/leaderboard", headers=self.headers)
        entries = response.json().get("entries", [])
        assert len(entries) == 0, f"Expected 0 entries after clear, got {len(entries)}"
        print("Verified leaderboard is empty after clear")
    
    def test_08_unauthorized_access(self):
        """Test leaderboard endpoints require authentication"""
        # GET without auth
        response = requests.get(f"{BASE_URL}/api/leaderboard")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        
        # POST without auth
        response = requests.post(f"{BASE_URL}/api/leaderboard", json={
            "model_id": "test", "algorithm": "test", "problem_type": "classification", "metrics": {}
        })
        assert response.status_code == 401
        
        # DELETE without auth
        response = requests.delete(f"{BASE_URL}/api/leaderboard/test")
        assert response.status_code == 401
        
        print("All unauthorized requests correctly returned 401")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
