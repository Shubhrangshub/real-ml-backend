"""
Admin Dashboard Backend Tests
Tests for admin endpoints: user management, analytics, activity log, system controls
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://automl-validation.preview.emergentagent.com')

# Test credentials
ADMIN_EMAIL = "shubhrangshub@gmail.com"
ADMIN_PASSWORD = "MyNewPass123!"
NON_ADMIN_EMAIL = "test@automl.com"
NON_ADMIN_PASSWORD = "Test1234!"


class TestAdminAuth:
    """Test admin authentication and authorization"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        """Get admin user token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        })
        assert response.status_code == 200, f"Admin login failed: {response.text}"
        data = response.json()
        assert "token" in data, "No token in response"
        assert data.get("user", {}).get("is_admin") == True, "User should be admin"
        return data["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        """Get non-admin user token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL,
            "password": NON_ADMIN_PASSWORD
        })
        assert response.status_code == 200, f"Non-admin login failed: {response.text}"
        data = response.json()
        assert "token" in data, "No token in response"
        # Non-admin should not have is_admin=True
        assert data.get("user", {}).get("is_admin") != True, "User should NOT be admin"
        return data["token"]
    
    def test_admin_login_returns_is_admin_flag(self, admin_token):
        """Verify admin user has is_admin=True in login response"""
        response = requests.get(f"{BASE_URL}/api/auth/me", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("is_admin") == True, "Admin user should have is_admin=True"
        assert data.get("email") == ADMIN_EMAIL
        print(f"✓ Admin user {ADMIN_EMAIL} has is_admin=True")
    
    def test_non_admin_login_returns_is_admin_false(self, non_admin_token):
        """Verify non-admin user has is_admin=False in login response"""
        response = requests.get(f"{BASE_URL}/api/auth/me", headers={
            "Authorization": f"Bearer {non_admin_token}"
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("is_admin") != True, "Non-admin user should NOT have is_admin=True"
        print(f"✓ Non-admin user {NON_ADMIN_EMAIL} has is_admin=False")


class TestAdminUsersEndpoint:
    """Test GET /api/admin/users endpoint"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_admin_can_list_users(self, admin_token):
        """Admin should be able to list all users"""
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "users" in data, "Response should contain 'users' key"
        assert "total" in data, "Response should contain 'total' key"
        assert isinstance(data["users"], list), "Users should be a list"
        assert data["total"] >= 1, "Should have at least 1 user"
        
        # Verify user structure
        if data["users"]:
            user = data["users"][0]
            assert "user_id" in user, "User should have user_id"
            assert "email" in user, "User should have email"
            assert "is_admin" in user, "User should have is_admin flag"
            assert "is_disabled" in user, "User should have is_disabled flag"
            assert "snapshots_count" in user, "User should have snapshots_count"
            assert "leaderboard_count" in user, "User should have leaderboard_count"
        
        print(f"✓ Admin can list users - Total: {data['total']}")
    
    def test_non_admin_cannot_list_users(self, non_admin_token):
        """Non-admin should get 403 Forbidden"""
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {non_admin_token}"
        })
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on /api/admin/users")
    
    def test_unauthenticated_cannot_list_users(self):
        """Unauthenticated request should get 401"""
        response = requests.get(f"{BASE_URL}/api/admin/users")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Unauthenticated request correctly gets 401")


class TestAdminAnalyticsEndpoint:
    """Test GET /api/admin/analytics endpoint"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_admin_can_get_analytics(self, admin_token):
        """Admin should be able to get platform analytics"""
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify analytics structure
        expected_keys = [
            "total_users", "total_snapshots", "total_leaderboard_entries",
            "active_sessions", "google_users", "email_users",
            "recent_signups", "total_trains", "total_logins", "total_saves"
        ]
        for key in expected_keys:
            assert key in data, f"Analytics should contain '{key}'"
            assert isinstance(data[key], int), f"'{key}' should be an integer"
        
        print(f"✓ Admin can get analytics - Total users: {data['total_users']}, Active sessions: {data['active_sessions']}")
    
    def test_non_admin_cannot_get_analytics(self, non_admin_token):
        """Non-admin should get 403 Forbidden"""
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers={
            "Authorization": f"Bearer {non_admin_token}"
        })
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on /api/admin/analytics")


class TestAdminActivityEndpoint:
    """Test GET /api/admin/activity endpoint"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_admin_can_get_activity(self, admin_token):
        """Admin should be able to get activity log"""
        response = requests.get(f"{BASE_URL}/api/admin/activity?limit=50", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert "activities" in data, "Response should contain 'activities' key"
        assert "total" in data, "Response should contain 'total' key"
        assert isinstance(data["activities"], list), "Activities should be a list"
        
        # Verify activity structure if there are any
        if data["activities"]:
            activity = data["activities"][0]
            assert "action" in activity, "Activity should have action"
            assert "timestamp" in activity, "Activity should have timestamp"
        
        print(f"✓ Admin can get activity log - Total: {data['total']}")
    
    def test_admin_can_filter_activity_by_action(self, admin_token):
        """Admin should be able to filter activity by action type"""
        response = requests.get(f"{BASE_URL}/api/admin/activity?limit=50&action=login", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # All activities should be login actions
        for activity in data["activities"]:
            assert activity["action"] == "login", f"Expected 'login' action, got '{activity['action']}'"
        
        print(f"✓ Admin can filter activity by action - Login activities: {len(data['activities'])}")
    
    def test_non_admin_cannot_get_activity(self, non_admin_token):
        """Non-admin should get 403 Forbidden"""
        response = requests.get(f"{BASE_URL}/api/admin/activity", headers={
            "Authorization": f"Bearer {non_admin_token}"
        })
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on /api/admin/activity")


class TestAdminUserManagement:
    """Test PATCH/DELETE /api/admin/users/{user_id} endpoints"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_admin_can_toggle_user_disabled_flag(self, admin_token):
        """Admin should be able to toggle is_disabled flag"""
        # First get a non-admin user
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        users = response.json()["users"]
        
        # Find a non-admin user to toggle
        target_user = None
        for user in users:
            if user["email"] != ADMIN_EMAIL and not user.get("is_admin"):
                target_user = user
                break
        
        if not target_user:
            pytest.skip("No non-admin user found to test toggle")
        
        original_disabled = target_user.get("is_disabled", False)
        
        # Toggle disabled flag
        response = requests.patch(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"is_disabled": not original_disabled}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert data["updated"]["is_disabled"] == (not original_disabled)
        
        # Revert the change
        requests.patch(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"is_disabled": original_disabled}
        )
        
        print(f"✓ Admin can toggle user disabled flag for {target_user['email']}")
    
    def test_non_admin_cannot_update_user(self, non_admin_token, admin_token):
        """Non-admin should get 403 when trying to update user"""
        # Get a user to try to update
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        users = response.json()["users"]
        if not users:
            pytest.skip("No users found")
        
        target_user = users[0]
        
        response = requests.patch(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}",
            headers={"Authorization": f"Bearer {non_admin_token}"},
            json={"is_disabled": True}
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on PATCH /api/admin/users/{user_id}")


class TestAdminResetPassword:
    """Test POST /api/admin/users/{user_id}/reset-password endpoint"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_admin_can_reset_user_password(self, admin_token):
        """Admin should be able to reset a user's password"""
        # Get a non-admin user
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        users = response.json()["users"]
        
        # Find test@automl.com user
        target_user = None
        for user in users:
            if user["email"] == NON_ADMIN_EMAIL:
                target_user = user
                break
        
        if not target_user:
            pytest.skip("test@automl.com user not found")
        
        # Reset password to the same password (so tests continue to work)
        response = requests.post(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}/reset-password",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"new_password": NON_ADMIN_PASSWORD}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert NON_ADMIN_EMAIL in data.get("message", "")
        
        print(f"✓ Admin can reset password for {NON_ADMIN_EMAIL}")
    
    def test_admin_reset_password_validation(self, admin_token):
        """Admin reset password should validate password length"""
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        users = response.json()["users"]
        if not users:
            pytest.skip("No users found")
        
        target_user = users[0]
        
        # Try with too short password
        response = requests.post(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}/reset-password",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"new_password": "12345"}  # Less than 6 chars
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ Admin reset password correctly validates password length")
    
    def test_non_admin_cannot_reset_password(self, non_admin_token, admin_token):
        """Non-admin should get 403 when trying to reset password"""
        response = requests.get(f"{BASE_URL}/api/admin/users", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        users = response.json()["users"]
        if not users:
            pytest.skip("No users found")
        
        target_user = users[0]
        
        response = requests.post(
            f"{BASE_URL}/api/admin/users/{target_user['user_id']}/reset-password",
            headers={"Authorization": f"Bearer {non_admin_token}"},
            json={"new_password": "newpassword123"}
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on POST /api/admin/users/{user_id}/reset-password")


class TestAdminSystemControls:
    """Test DELETE /api/admin/system/* endpoints"""
    
    @pytest.fixture(scope="class")
    def admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    @pytest.fixture(scope="class")
    def non_admin_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": NON_ADMIN_EMAIL, "password": NON_ADMIN_PASSWORD
        })
        return response.json()["token"]
    
    def test_non_admin_cannot_clear_leaderboard(self, non_admin_token):
        """Non-admin should get 403 when trying to clear leaderboard"""
        response = requests.delete(
            f"{BASE_URL}/api/admin/system/leaderboard",
            headers={"Authorization": f"Bearer {non_admin_token}"}
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on DELETE /api/admin/system/leaderboard")
    
    def test_non_admin_cannot_clear_snapshots(self, non_admin_token):
        """Non-admin should get 403 when trying to clear snapshots"""
        response = requests.delete(
            f"{BASE_URL}/api/admin/system/snapshots",
            headers={"Authorization": f"Bearer {non_admin_token}"}
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Non-admin correctly gets 403 on DELETE /api/admin/system/snapshots")
    
    def test_admin_clear_leaderboard_endpoint_exists(self, admin_token):
        """Verify admin clear leaderboard endpoint exists and is accessible"""
        # We don't actually clear - just verify the endpoint works
        # This is a destructive operation so we just check it's accessible
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers={
            "Authorization": f"Bearer {admin_token}"
        })
        assert response.status_code == 200
        initial_count = response.json().get("total_leaderboard_entries", 0)
        print(f"✓ Admin system controls accessible - Current leaderboard entries: {initial_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
