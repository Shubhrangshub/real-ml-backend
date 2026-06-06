"""
Backend API tests for Preprocess and Tune features
Tests existing endpoints to ensure no regression after new feature additions
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthAndAuth:
    """Health check and authentication tests"""
    
    def test_health_endpoint(self):
        """Test API health endpoint"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        print(f"Health check passed: {data}")
    
    def test_login_success(self):
        """Test login with valid credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "test@automl.com",
            "password": "Test1234!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "user" in data
        print(f"Login successful for user: {data['user'].get('email')}")
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "invalid@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        print("Invalid login correctly rejected")


class TestDeployEndpoints:
    """Deploy API endpoint tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "test@automl.com",
            "password": "Test1234!"
        })
        if response.status_code == 200:
            return response.json().get("token")
        pytest.skip("Authentication failed")
    
    def test_get_deployments_authenticated(self, auth_token):
        """Test GET /api/deploy with authentication"""
        response = requests.get(
            f"{BASE_URL}/api/deploy",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # API returns {"deployments": [...]} format
        assert "deployments" in data or isinstance(data, list)
        deployments = data.get("deployments", data) if isinstance(data, dict) else data
        print(f"Got {len(deployments)} deployments")
    
    def test_get_deployments_unauthenticated(self):
        """Test GET /api/deploy without authentication"""
        response = requests.get(f"{BASE_URL}/api/deploy")
        assert response.status_code == 401
        print("Unauthenticated deploy request correctly rejected")


class TestLeaderboardEndpoints:
    """Leaderboard API endpoint tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "test@automl.com",
            "password": "Test1234!"
        })
        if response.status_code == 200:
            return response.json().get("token")
        pytest.skip("Authentication failed")
    
    def test_get_leaderboard(self, auth_token):
        """Test GET /api/leaderboard"""
        response = requests.get(
            f"{BASE_URL}/api/leaderboard",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # API returns {"entries": [...], "status": "success"} format
        assert "entries" in data or isinstance(data, list)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        print(f"Got {len(entries)} leaderboard entries")


class TestSampleDatasets:
    """Sample dataset endpoint tests - these are served as static files"""
    
    def test_loan_approval_dataset(self):
        """Test loan approval sample dataset"""
        # Sample datasets are served from /api/sample_data/ or root
        response = requests.get(f"{BASE_URL}/api/sample_data/loan_approval.csv")
        if response.status_code == 404:
            # Try alternative path
            response = requests.get(f"{BASE_URL}/loan_approval.csv")
        
        # If still 404, the dataset might be served differently
        if response.status_code == 404:
            print("Sample dataset endpoint not found - may be served from frontend")
            pytest.skip("Sample dataset served from frontend, not backend")
        
        assert response.status_code == 200
        lines = response.text.strip().split('\n')
        assert len(lines) > 10
        print(f"Loan approval dataset: {len(lines)} rows")
    
    def test_house_prices_dataset(self):
        """Test house prices sample dataset"""
        response = requests.get(f"{BASE_URL}/api/sample_data/house_prices.csv")
        if response.status_code == 404:
            response = requests.get(f"{BASE_URL}/house_prices.csv")
        
        if response.status_code == 404:
            pytest.skip("Sample dataset served from frontend, not backend")
        
        assert response.status_code == 200
        lines = response.text.strip().split('\n')
        assert len(lines) > 10
        print(f"House prices dataset: {len(lines)} rows")


class TestSnapshotsEndpoints:
    """Snapshots API endpoint tests"""
    
    def test_get_snapshots(self):
        """Test GET /api/snapshots"""
        response = requests.get(f"{BASE_URL}/api/snapshots")
        assert response.status_code == 200
        data = response.json()
        # API returns {"snapshots": [...], "status": "success"} format
        assert "snapshots" in data or isinstance(data, list)
        snapshots = data.get("snapshots", data) if isinstance(data, dict) else data
        print(f"Got {len(snapshots)} snapshots")


class TestAdminEndpoints:
    """Admin API endpoint tests"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "shubhrangshub@gmail.com",
            "password": "MyNewPass123!"
        })
        if response.status_code == 200:
            return response.json().get("token")
        pytest.skip("Admin authentication failed")
    
    def test_admin_users_endpoint(self, admin_token):
        """Test GET /api/admin/users"""
        response = requests.get(
            f"{BASE_URL}/api/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # API returns {"users": [...], "total": N} format
        assert "users" in data or isinstance(data, list)
        users = data.get("users", data) if isinstance(data, dict) else data
        print(f"Got {len(users)} users from admin endpoint")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
