"""
Auth System Tests - Email/Password Authentication and Session Management

Tests cover:
- POST /api/auth/signup - User registration with email/password
- POST /api/auth/login - User login with email/password
- GET /api/auth/me - Get current user from session token
- POST /api/auth/logout - Logout and clear session
- Duplicate email signup returns 400
- Invalid credentials return 401
- Per-user snapshot isolation
"""

import pytest
import requests
import os
import uuid
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthCheck:
    """Basic health check to ensure backend is running"""
    
    def test_health_endpoint(self):
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mongodb"] == "connected"
        print("PASS: Health check - backend is healthy and MongoDB connected")


class TestAuthSignup:
    """User signup tests"""
    
    def test_signup_success(self):
        """Test successful user signup with email/password"""
        unique_email = f"test_signup_{uuid.uuid4().hex[:8]}@example.com"
        payload = {
            "email": unique_email,
            "password": "testpass123",
            "name": "Test Signup User"
        }
        response = requests.post(f"{BASE_URL}/api/auth/signup", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert data["status"] == "success"
        assert "token" in data
        assert len(data["token"]) > 0
        assert "user" in data
        assert data["user"]["email"] == unique_email
        assert data["user"]["name"] == "Test Signup User"
        assert "user_id" in data["user"]
        print(f"PASS: Signup success - created user {unique_email}")
        
        # Store for cleanup
        return data["token"], data["user"]["user_id"]
    
    def test_signup_duplicate_email_returns_400(self):
        """Test that duplicate email signup returns 400 error"""
        unique_email = f"test_dup_{uuid.uuid4().hex[:8]}@example.com"
        payload = {
            "email": unique_email,
            "password": "testpass123",
            "name": "First User"
        }
        
        # First signup should succeed
        response1 = requests.post(f"{BASE_URL}/api/auth/signup", json=payload)
        assert response1.status_code == 200
        
        # Second signup with same email should fail with 400
        payload["name"] = "Second User"
        response2 = requests.post(f"{BASE_URL}/api/auth/signup", json=payload)
        assert response2.status_code == 400, f"Expected 400 for duplicate email, got {response2.status_code}"
        
        data = response2.json()
        assert "detail" in data
        assert "already registered" in data["detail"].lower() or "email" in data["detail"].lower()
        print("PASS: Duplicate email signup correctly returns 400")


class TestAuthLogin:
    """User login tests"""
    
    def test_login_success(self):
        """Test successful login with valid credentials"""
        # First create a user
        unique_email = f"test_login_{uuid.uuid4().hex[:8]}@example.com"
        signup_payload = {
            "email": unique_email,
            "password": "logintest123",
            "name": "Login Test User"
        }
        signup_response = requests.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
        assert signup_response.status_code == 200
        
        # Now login with same credentials
        login_payload = {
            "email": unique_email,
            "password": "logintest123"
        }
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert data["status"] == "success"
        assert "token" in data
        assert len(data["token"]) > 0
        assert "user" in data
        assert data["user"]["email"] == unique_email
        assert data["user"]["name"] == "Login Test User"
        print(f"PASS: Login success - user {unique_email} logged in")
        
        return data["token"]
    
    def test_login_invalid_password_returns_401(self):
        """Test that wrong password returns 401"""
        # First create a user
        unique_email = f"test_wrongpw_{uuid.uuid4().hex[:8]}@example.com"
        signup_payload = {
            "email": unique_email,
            "password": "correctpassword",
            "name": "Wrong Password Test"
        }
        signup_response = requests.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
        assert signup_response.status_code == 200
        
        # Try login with wrong password
        login_payload = {
            "email": unique_email,
            "password": "wrongpassword"
        }
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload)
        assert response.status_code == 401, f"Expected 401 for wrong password, got {response.status_code}"
        
        data = response.json()
        assert "detail" in data
        print("PASS: Invalid password correctly returns 401")
    
    def test_login_nonexistent_email_returns_401(self):
        """Test that non-existent email returns 401"""
        login_payload = {
            "email": f"nonexistent_{uuid.uuid4().hex}@example.com",
            "password": "anypassword"
        }
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload)
        assert response.status_code == 401, f"Expected 401 for non-existent email, got {response.status_code}"
        print("PASS: Non-existent email correctly returns 401")


class TestAuthMe:
    """GET /api/auth/me endpoint tests"""
    
    def test_auth_me_with_valid_token(self):
        """Test that /api/auth/me returns user with valid token"""
        # Create user and get token
        unique_email = f"test_me_{uuid.uuid4().hex[:8]}@example.com"
        signup_payload = {
            "email": unique_email,
            "password": "metest123",
            "name": "Me Test User"
        }
        signup_response = requests.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
        assert signup_response.status_code == 200
        token = signup_response.json()["token"]
        
        # Call /api/auth/me with token
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert data["email"] == unique_email
        assert data["name"] == "Me Test User"
        assert "user_id" in data
        print(f"PASS: /api/auth/me returns correct user data")
    
    def test_auth_me_without_token_returns_401(self):
        """Test that /api/auth/me without token returns 401"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401, f"Expected 401 without token, got {response.status_code}"
        print("PASS: /api/auth/me without token returns 401")
    
    def test_auth_me_with_invalid_token_returns_401(self):
        """Test that /api/auth/me with invalid token returns 401"""
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": "Bearer invalid_token_12345"}
        )
        assert response.status_code == 401, f"Expected 401 with invalid token, got {response.status_code}"
        print("PASS: /api/auth/me with invalid token returns 401")


class TestAuthLogout:
    """POST /api/auth/logout endpoint tests"""
    
    def test_logout_clears_session(self):
        """Test that logout clears the session"""
        # Create user and get token
        unique_email = f"test_logout_{uuid.uuid4().hex[:8]}@example.com"
        signup_payload = {
            "email": unique_email,
            "password": "logouttest123",
            "name": "Logout Test User"
        }
        signup_response = requests.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
        assert signup_response.status_code == 200
        token = signup_response.json()["token"]
        
        # Verify token works
        me_response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == 200
        
        # Logout
        logout_response = requests.post(
            f"{BASE_URL}/api/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert logout_response.status_code == 200
        data = logout_response.json()
        assert data["status"] == "success"
        
        # Verify token no longer works (session cleared)
        me_response2 = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response2.status_code == 401, f"Expected 401 after logout, got {me_response2.status_code}"
        print("PASS: Logout clears session - token no longer valid")


class TestPerUserSnapshots:
    """Test per-user snapshot isolation"""
    
    def test_snapshots_are_per_user(self):
        """Test that each user only sees their own snapshots"""
        # Create User A
        email_a = f"test_user_a_{uuid.uuid4().hex[:8]}@example.com"
        signup_a = requests.post(f"{BASE_URL}/api/auth/signup", json={
            "email": email_a, "password": "userApass", "name": "User A"
        })
        assert signup_a.status_code == 200
        token_a = signup_a.json()["token"]
        
        # Create User B
        email_b = f"test_user_b_{uuid.uuid4().hex[:8]}@example.com"
        signup_b = requests.post(f"{BASE_URL}/api/auth/signup", json={
            "email": email_b, "password": "userBpass", "name": "User B"
        })
        assert signup_b.status_code == 200
        token_b = signup_b.json()["token"]
        
        # User A creates a snapshot
        snapshot_payload = {
            "name": f"TEST_UserA_Snapshot_{uuid.uuid4().hex[:6]}",
            "dataset_name": "Test Dataset",
            "target_column": "target",
            "problem_type": "classification",
            "row_count": 100,
            "col_count": 5,
            "models_summary": [],
            "key_metrics": {},
            "state": {"csvText": "a,b,c\n1,2,3"}
        }
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=snapshot_payload,
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert create_response.status_code == 200
        snapshot_id = create_response.json()["snapshot_id"]
        
        # User A should see the snapshot
        list_a = requests.get(
            f"{BASE_URL}/api/snapshots",
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert list_a.status_code == 200
        snapshots_a = list_a.json()["snapshots"]
        snapshot_ids_a = [s["snapshot_id"] for s in snapshots_a]
        assert snapshot_id in snapshot_ids_a, "User A should see their own snapshot"
        
        # User B should NOT see User A's snapshot
        list_b = requests.get(
            f"{BASE_URL}/api/snapshots",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert list_b.status_code == 200
        snapshots_b = list_b.json()["snapshots"]
        snapshot_ids_b = [s["snapshot_id"] for s in snapshots_b]
        assert snapshot_id not in snapshot_ids_b, "User B should NOT see User A's snapshot"
        
        print("PASS: Snapshots are correctly isolated per user")
        
        # Cleanup - delete the test snapshot
        requests.delete(
            f"{BASE_URL}/api/snapshots/{snapshot_id}",
            headers={"Authorization": f"Bearer {token_a}"}
        )


class TestSharedSnapshotNoAuth:
    """Test that shared snapshots can be viewed without authentication"""
    
    def test_get_snapshot_by_id_no_auth_required(self):
        """Test that GET /api/snapshots/{id} works without auth (for sharing)"""
        # First create a user and snapshot
        unique_email = f"test_share_{uuid.uuid4().hex[:8]}@example.com"
        signup_response = requests.post(f"{BASE_URL}/api/auth/signup", json={
            "email": unique_email, "password": "sharetest123", "name": "Share Test User"
        })
        assert signup_response.status_code == 200
        token = signup_response.json()["token"]
        
        # Create a snapshot
        snapshot_payload = {
            "name": f"TEST_SharedSnapshot_{uuid.uuid4().hex[:6]}",
            "dataset_name": "Shared Dataset",
            "target_column": "target",
            "problem_type": "classification",
            "row_count": 50,
            "col_count": 3,
            "models_summary": [],
            "key_metrics": {"accuracy": 0.95},
            "state": {"csvText": "x,y,target\n1,2,yes\n3,4,no"}
        }
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=snapshot_payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        assert create_response.status_code == 200
        snapshot_id = create_response.json()["snapshot_id"]
        
        # Now try to GET the snapshot WITHOUT any auth header
        get_response = requests.get(f"{BASE_URL}/api/snapshots/{snapshot_id}")
        assert get_response.status_code == 200, f"Expected 200 for shared snapshot, got {get_response.status_code}"
        
        data = get_response.json()
        assert data["status"] == "success"
        assert "snapshot" in data
        assert data["snapshot"]["snapshot_id"] == snapshot_id
        assert data["snapshot"]["name"] == snapshot_payload["name"]
        print(f"PASS: Shared snapshot {snapshot_id} accessible without auth")
        
        # Cleanup
        requests.delete(
            f"{BASE_URL}/api/snapshots/{snapshot_id}",
            headers={"Authorization": f"Bearer {token}"}
        )


class TestGoogleLoginButtonExists:
    """Test that Google login endpoint exists (actual OAuth cannot be tested automatically)"""
    
    def test_google_auth_endpoint_exists(self):
        """Test that POST /api/auth/google endpoint exists"""
        # Send a request with invalid session_id - should get 401, not 404
        response = requests.post(
            f"{BASE_URL}/api/auth/google",
            json={"session_id": "invalid_test_session"}
        )
        # Should be 401 (invalid session) not 404 (endpoint not found)
        assert response.status_code in [401, 500], f"Expected 401 or 500, got {response.status_code}"
        print("PASS: Google auth endpoint exists (returns 401 for invalid session)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
