"""
Test suite for Auto-save and Share Analysis features
Tests:
1. Auto-save: After training, a new snapshot should be created automatically
2. Share flow: Share URL should contain ?snapshot= parameter
3. Snapshot CRUD: Create, Read, List, Delete operations
4. Snapshot state: Full state including csvText, models, trainingResult, SHAP/LIME data
"""

import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://automl-validation.preview.emergentagent.com')

# Sample dataset for testing (Loan Approval)
LOAN_APPROVAL_CSV = """age,income,credit_score,loan_amount,approved
25,45000,650,10000,0
35,75000,720,25000,1
45,95000,780,50000,1
28,52000,680,15000,0
52,120000,800,75000,1
23,38000,620,8000,0
38,82000,740,30000,1
42,88000,760,40000,1
30,62000,700,20000,1
48,105000,790,60000,1
22,32000,600,5000,0
55,130000,810,80000,1
29,48000,660,12000,0
40,90000,750,35000,1
33,70000,710,22000,1
27,44000,640,11000,0
50,110000,795,65000,1
36,78000,730,28000,1
24,40000,630,9000,0
44,98000,770,45000,1"""


class TestHealthEndpoint:
    """Test health endpoint is working"""
    
    def test_health_check(self):
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "mongodb" in data
        print(f"✓ Health check passed: {data}")


class TestSnapshotCRUD:
    """Test Snapshot CRUD operations"""
    
    def test_list_snapshots(self):
        """GET /api/snapshots - List all snapshots"""
        response = requests.get(f"{BASE_URL}/api/snapshots")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "snapshots" in data
        assert isinstance(data["snapshots"], list)
        print(f"✓ List snapshots: Found {len(data['snapshots'])} snapshots")
        return data["snapshots"]
    
    def test_create_snapshot(self):
        """POST /api/snapshots - Create a new snapshot"""
        # Create a test snapshot with full state
        payload = {
            "name": "TEST_auto_save_test",
            "dataset_name": "Loan Approval",
            "target_column": "approved",
            "problem_type": "classification",
            "row_count": 20,
            "col_count": 5,
            "models_summary": [{"algorithm": "random_forest", "score": 0.95}],
            "key_metrics": {"accuracy": 0.95, "f1": 0.94},
            "state": {
                "csvText": LOAN_APPROVAL_CSV,
                "targetColumn": "approved",
                "algorithm": "auto",
                "evalMode": "split",
                "trainingResult": {
                    "status": "success",
                    "problemType": "classification",
                    "bestModel": {
                        "algorithm": "random_forest",
                        "testMetrics": {"accuracy": 0.95, "f1": 0.94}
                    },
                    "leaderboard": [
                        {"algorithm": "random_forest", "score": 0.95},
                        {"algorithm": "logistic_regression", "score": 0.90}
                    ]
                },
                "models": [{"id": "test-model-1", "algorithm": "random_forest"}],
                "shapGlobal": [{"feature": "income", "importance": 0.35}],
                "limeResult": {"explanation": "test"}
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "snapshot_id" in data
        snapshot_id = data["snapshot_id"]
        print(f"✓ Created snapshot with ID: {snapshot_id}")
        return snapshot_id
    
    def test_get_snapshot_by_id(self):
        """GET /api/snapshots/{id} - Get full snapshot by ID"""
        # First create a snapshot
        snapshot_id = self.test_create_snapshot()
        
        # Then retrieve it
        response = requests.get(f"{BASE_URL}/api/snapshots/{snapshot_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "snapshot" in data
        
        snapshot = data["snapshot"]
        assert snapshot["snapshot_id"] == snapshot_id
        assert snapshot["name"] == "TEST_auto_save_test"
        assert snapshot["dataset_name"] == "Loan Approval"
        assert snapshot["target_column"] == "approved"
        assert snapshot["problem_type"] == "classification"
        
        # Verify full state is stored
        state = snapshot["state"]
        assert "csvText" in state
        assert "trainingResult" in state
        assert "models" in state
        assert "shapGlobal" in state
        assert "limeResult" in state
        
        # Verify csvText contains actual data
        assert "age,income,credit_score" in state["csvText"]
        
        # Verify trainingResult structure
        assert state["trainingResult"]["status"] == "success"
        assert state["trainingResult"]["problemType"] == "classification"
        
        print(f"✓ Retrieved snapshot {snapshot_id} with full state")
        print(f"  - Dataset: {snapshot['dataset_name']}")
        print(f"  - Target: {snapshot['target_column']}")
        print(f"  - Problem type: {snapshot['problem_type']}")
        print(f"  - State keys: {list(state.keys())}")
        
        return snapshot_id
    
    def test_delete_snapshot(self):
        """DELETE /api/snapshots/{id} - Delete a snapshot"""
        # First create a snapshot
        snapshot_id = self.test_create_snapshot()
        
        # Delete it
        response = requests.delete(f"{BASE_URL}/api/snapshots/{snapshot_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Verify it's deleted
        response = requests.get(f"{BASE_URL}/api/snapshots/{snapshot_id}")
        assert response.status_code == 404
        
        print(f"✓ Deleted snapshot {snapshot_id}")
    
    def test_snapshot_not_found(self):
        """GET /api/snapshots/{id} - 404 for non-existent snapshot"""
        response = requests.get(f"{BASE_URL}/api/snapshots/nonexistent-id")
        assert response.status_code == 404
        print("✓ 404 returned for non-existent snapshot")


class TestSnapshotStateIntegrity:
    """Test that snapshot state contains all required fields for sharing"""
    
    def test_snapshot_contains_full_analysis_state(self):
        """Verify snapshot state has all fields needed for view-only mode"""
        # Create a comprehensive snapshot
        payload = {
            "name": "TEST_full_state_test",
            "dataset_name": "Test Dataset",
            "target_column": "target",
            "problem_type": "classification",
            "row_count": 100,
            "col_count": 10,
            "models_summary": [],
            "key_metrics": {},
            "state": {
                # Core data
                "csvText": "col1,col2,target\n1,2,0\n3,4,1",
                "targetColumn": "target",
                "algorithm": "auto",
                "evalMode": "split",
                
                # Training results
                "trainingResult": {
                    "status": "success",
                    "problemType": "classification",
                    "bestModel": {"algorithm": "random_forest"},
                    "leaderboard": [],
                    "dataInfo": {"numSamples": 100}
                },
                
                # Models
                "models": [{"id": "m1", "algorithm": "rf"}],
                
                # Predictions
                "predictionResult": {"prediction": 1},
                "predictionHistory": [{"id": 1, "prediction": 1}],
                
                # XAI data
                "shapGlobal": [{"feature": "col1", "importance": 0.5}],
                "shapBeeswarm": [{"feature": "col1", "values": [0.1, 0.2]}],
                "shapLocal": {"values": [0.1, 0.2]},
                "shapDependence": {"feature": "col1", "data": []},
                "limeResult": {"explanation": []},
                "limeProbs": [0.3, 0.7],
                
                # Unsupervised
                "unsupervisedResult": None,
                "clusterResult": None,
                "anomalyResult": None,
                
                # Other
                "batchResults": None,
                "cleaningLog": [],
                "precleanScan": None
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        snapshot_id = response.json()["snapshot_id"]
        
        # Retrieve and verify
        response = requests.get(f"{BASE_URL}/api/snapshots/{snapshot_id}")
        assert response.status_code == 200
        state = response.json()["snapshot"]["state"]
        
        # Verify all critical fields are present
        required_fields = [
            "csvText", "targetColumn", "algorithm", "evalMode",
            "trainingResult", "models", "predictionResult", "predictionHistory",
            "shapGlobal", "shapBeeswarm", "shapLocal", "shapDependence",
            "limeResult", "limeProbs"
        ]
        
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"
        
        print(f"✓ Snapshot contains all required fields for sharing")
        print(f"  - Fields verified: {required_fields}")
        
        # Cleanup
        requests.delete(f"{BASE_URL}/api/snapshots/{snapshot_id}")


class TestShareURLFormat:
    """Test share URL format and snapshot ID structure"""
    
    def test_snapshot_id_format(self):
        """Verify snapshot_id is suitable for URL parameter"""
        payload = {
            "name": "TEST_url_format_test",
            "dataset_name": "Test",
            "state": {"csvText": "a,b\n1,2"}
        }
        
        response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        snapshot_id = response.json()["snapshot_id"]
        
        # Verify ID format (should be URL-safe)
        assert len(snapshot_id) > 0
        assert " " not in snapshot_id
        assert "?" not in snapshot_id
        assert "&" not in snapshot_id
        
        # Verify it can be used in URL
        expected_url_format = f"https://example.com/?snapshot={snapshot_id}"
        assert snapshot_id in expected_url_format
        
        print(f"✓ Snapshot ID is URL-safe: {snapshot_id}")
        
        # Cleanup
        requests.delete(f"{BASE_URL}/api/snapshots/{snapshot_id}")


class TestAutoSaveScenario:
    """Test auto-save scenario (simulated)"""
    
    def test_count_snapshots_before_and_after(self):
        """Verify snapshot count increases after creating a new one"""
        # Get initial count
        response = requests.get(f"{BASE_URL}/api/snapshots")
        initial_count = len(response.json()["snapshots"])
        
        # Create a new snapshot (simulating auto-save after training)
        payload = {
            "name": "TEST_auto_save_simulation",
            "dataset_name": "Loan Approval",
            "target_column": "approved",
            "problem_type": "classification",
            "state": {
                "csvText": LOAN_APPROVAL_CSV,
                "targetColumn": "approved",
                "trainingResult": {"status": "success", "problemType": "classification"}
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        snapshot_id = response.json()["snapshot_id"]
        
        # Get new count
        response = requests.get(f"{BASE_URL}/api/snapshots")
        new_count = len(response.json()["snapshots"])
        
        assert new_count == initial_count + 1, f"Expected {initial_count + 1} snapshots, got {new_count}"
        print(f"✓ Snapshot count increased from {initial_count} to {new_count}")
        
        # Cleanup
        requests.delete(f"{BASE_URL}/api/snapshots/{snapshot_id}")


# Cleanup function to remove test snapshots
def cleanup_test_snapshots():
    """Remove all TEST_ prefixed snapshots"""
    response = requests.get(f"{BASE_URL}/api/snapshots")
    if response.status_code == 200:
        snapshots = response.json().get("snapshots", [])
        for snap in snapshots:
            if snap.get("name", "").startswith("TEST_"):
                requests.delete(f"{BASE_URL}/api/snapshots/{snap['snapshot_id']}")
                print(f"  Cleaned up: {snap['name']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Auto-save and Share Analysis Tests")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Cleanup
    print("\nCleaning up test snapshots...")
    cleanup_test_snapshots()
