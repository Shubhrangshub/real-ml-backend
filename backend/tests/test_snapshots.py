"""
Test suite for History & Sharing (Snapshot) API endpoints.
Tests: POST/GET/DELETE /api/snapshots
"""

import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://browser-ml-fast.preview.emergentagent.com').rstrip('/')


class TestHealthCheck:
    """Basic health check to ensure API is running"""
    
    def test_health_endpoint(self):
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "mongodb" in data
        print(f"Health check passed: {data}")


class TestSnapshotCRUD:
    """Test CRUD operations for snapshots (History & Sharing feature)"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Store created snapshot IDs for cleanup"""
        self.created_snapshot_ids = []
        yield
        # Cleanup: Delete all test snapshots
        for sid in self.created_snapshot_ids:
            try:
                requests.delete(f"{BASE_URL}/api/snapshots/{sid}")
            except:
                pass
    
    def test_create_snapshot_success(self):
        """TEST 16: POST /api/snapshots -> verify returns snapshot_id"""
        payload = {
            "name": "TEST_Loan_Approval_Analysis",
            "dataset_name": "Loan Approval Dataset",
            "target_column": "approved",
            "problem_type": "classification",
            "row_count": 20,
            "col_count": 5,
            "models_summary": [
                {"algorithm": "random_forest", "score": 0.85},
                {"algorithm": "logistic_regression", "score": 0.80}
            ],
            "key_metrics": {
                "accuracy": 0.85,
                "f1": 0.82,
                "precision": 0.84
            },
            "state": {
                "csvText": "col1,col2,approved\n1,2,yes\n3,4,no",
                "targetColumn": "approved",
                "algorithm": "auto",
                "trainingResult": {
                    "problemType": "classification",
                    "leaderboard": [
                        {"algorithm": "random_forest", "score": 0.85}
                    ]
                }
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "snapshot_id" in data, f"Response missing snapshot_id: {data}"
        assert data["status"] == "success"
        assert isinstance(data["snapshot_id"], str)
        assert len(data["snapshot_id"]) > 0
        
        self.created_snapshot_ids.append(data["snapshot_id"])
        print(f"Created snapshot with ID: {data['snapshot_id']}")
        
        return data["snapshot_id"]
    
    def test_list_snapshots(self):
        """TEST 17: GET /api/snapshots -> verify returns list of snapshots"""
        # First create a snapshot
        payload = {
            "name": "TEST_List_Snapshot",
            "dataset_name": "Test Dataset",
            "target_column": "target",
            "problem_type": "regression",
            "row_count": 100,
            "col_count": 10,
            "models_summary": [],
            "key_metrics": {},
            "state": {"csvText": "a,b\n1,2"}
        }
        
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert create_response.status_code == 200
        created_id = create_response.json()["snapshot_id"]
        self.created_snapshot_ids.append(created_id)
        
        # Now list snapshots
        response = requests.get(f"{BASE_URL}/api/snapshots")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # Verify response structure
        assert "snapshots" in data, f"Response missing snapshots: {data}"
        assert isinstance(data["snapshots"], list)
        
        # Verify our created snapshot is in the list
        snapshot_ids = [s.get("snapshot_id") for s in data["snapshots"]]
        assert created_id in snapshot_ids, f"Created snapshot {created_id} not found in list"
        
        # Verify snapshot structure (should NOT include full state for performance)
        for snap in data["snapshots"]:
            assert "snapshot_id" in snap
            assert "name" in snap
            assert "created_at" in snap
            # State should NOT be included in list response
            assert "state" not in snap, "List response should not include full state"
        
        print(f"Listed {len(data['snapshots'])} snapshots")
    
    def test_get_snapshot_by_id(self):
        """TEST 18: GET /api/snapshots/{id} -> verify returns full snapshot with state"""
        # First create a snapshot with specific state
        test_state = {
            "csvText": "feature1,feature2,target\n1,2,yes\n3,4,no\n5,6,yes",
            "targetColumn": "target",
            "algorithm": "random_forest",
            "trainingResult": {
                "problemType": "classification",
                "bestModel": {"algorithm": "random_forest", "score": 0.90}
            },
            "shapGlobal": {"feature1": 0.5, "feature2": 0.3},
            "limeResult": {"explanation": "test"}
        }
        
        payload = {
            "name": "TEST_Get_Snapshot",
            "dataset_name": "Test Dataset",
            "target_column": "target",
            "problem_type": "classification",
            "row_count": 3,
            "col_count": 3,
            "models_summary": [{"algorithm": "random_forest", "score": 0.90}],
            "key_metrics": {"accuracy": 0.90},
            "state": test_state
        }
        
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert create_response.status_code == 200
        created_id = create_response.json()["snapshot_id"]
        self.created_snapshot_ids.append(created_id)
        
        # Now get the snapshot by ID
        response = requests.get(f"{BASE_URL}/api/snapshots/{created_id}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # Verify response structure
        assert "snapshot" in data, f"Response missing snapshot: {data}"
        snapshot = data["snapshot"]
        
        # Verify all fields are present
        assert snapshot["snapshot_id"] == created_id
        assert snapshot["name"] == "TEST_Get_Snapshot"
        assert snapshot["dataset_name"] == "Test Dataset"
        assert snapshot["target_column"] == "target"
        assert snapshot["problem_type"] == "classification"
        assert snapshot["row_count"] == 3
        assert snapshot["col_count"] == 3
        
        # Verify state is included (full snapshot)
        assert "state" in snapshot, "Full snapshot should include state"
        assert snapshot["state"]["csvText"] == test_state["csvText"]
        assert snapshot["state"]["targetColumn"] == test_state["targetColumn"]
        assert snapshot["state"]["algorithm"] == test_state["algorithm"]
        
        print(f"Retrieved snapshot {created_id} with full state")
    
    def test_delete_snapshot(self):
        """TEST 19: DELETE /api/snapshots/{id} -> verify deletes successfully"""
        # First create a snapshot
        payload = {
            "name": "TEST_Delete_Snapshot",
            "dataset_name": "To Be Deleted",
            "target_column": "target",
            "problem_type": "regression",
            "row_count": 10,
            "col_count": 5,
            "models_summary": [],
            "key_metrics": {},
            "state": {"csvText": "a,b\n1,2"}
        }
        
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert create_response.status_code == 200
        created_id = create_response.json()["snapshot_id"]
        
        # Verify it exists
        get_response = requests.get(f"{BASE_URL}/api/snapshots/{created_id}")
        assert get_response.status_code == 200
        
        # Delete the snapshot
        delete_response = requests.delete(f"{BASE_URL}/api/snapshots/{created_id}")
        
        assert delete_response.status_code == 200, f"Expected 200, got {delete_response.status_code}"
        data = delete_response.json()
        assert data["status"] == "success"
        
        # Verify it no longer exists
        get_after_delete = requests.get(f"{BASE_URL}/api/snapshots/{created_id}")
        assert get_after_delete.status_code == 404, "Deleted snapshot should return 404"
        
        print(f"Successfully deleted snapshot {created_id}")
    
    def test_get_nonexistent_snapshot(self):
        """Test GET for non-existent snapshot returns 404"""
        response = requests.get(f"{BASE_URL}/api/snapshots/nonexistent-id-12345")
        assert response.status_code == 404
        print("Correctly returned 404 for non-existent snapshot")
    
    def test_delete_nonexistent_snapshot(self):
        """Test DELETE for non-existent snapshot returns 404"""
        response = requests.delete(f"{BASE_URL}/api/snapshots/nonexistent-id-12345")
        assert response.status_code == 404
        print("Correctly returned 404 for deleting non-existent snapshot")
    
    def test_snapshot_with_full_analysis_state(self):
        """Test saving a complete analysis state with all XAI results"""
        full_state = {
            "csvText": "age,income,approved\n25,50000,yes\n35,75000,yes\n45,30000,no",
            "targetColumn": "approved",
            "algorithm": "auto",
            "evalMode": "split",
            "cleaningLog": ["Removed 2 duplicates", "Filled 5 missing values"],
            "precleanScan": {"score": 85, "warnings": []},
            "trainingResult": {
                "problemType": "classification",
                "leaderboard": [
                    {"algorithm": "random_forest", "score": 0.92, "metrics": {"accuracy": 0.92}},
                    {"algorithm": "logistic_regression", "score": 0.88, "metrics": {"accuracy": 0.88}}
                ],
                "bestModel": {"algorithm": "random_forest", "score": 0.92}
            },
            "predictionResult": {"prediction": "yes", "confidence": 0.95},
            "predictionHistory": [
                {"id": 1, "type": "supervised", "prediction": "yes", "timestamp": 1234567890}
            ],
            "selectedModelIdx": 0,
            "unsupervisedResult": None,
            "clusterResult": None,
            "anomalyResult": None,
            "batchResults": None,
            "shapGlobal": [{"feature": "income", "importance": 0.6}, {"feature": "age", "importance": 0.4}],
            "shapBeeswarm": [{"feature": "income", "values": [0.1, 0.2, 0.3]}],
            "shapLocal": {"values": [0.3, 0.2], "baseValue": 0.5},
            "shapDependence": {"feature": "income", "data": []},
            "limeResult": {"explanation": [{"feature": "income", "weight": 0.5}]},
            "limeProbs": [0.95, 0.05],
            "clusterShap": None,
            "clusterBeeswarm": None,
            "shapSummary": "Income is the most important feature",
            "featureVsPred": None,
            "clusterComparison": None,
            "models": [{"id": "model-1", "algorithm": "random_forest"}]
        }
        
        payload = {
            "name": "TEST_Full_Analysis_State",
            "dataset_name": "Loan Approval",
            "target_column": "approved",
            "problem_type": "classification",
            "row_count": 3,
            "col_count": 3,
            "models_summary": [
                {"algorithm": "random_forest", "score": 0.92},
                {"algorithm": "logistic_regression", "score": 0.88}
            ],
            "key_metrics": {"accuracy": 0.92, "f1": 0.90, "precision": 0.91, "recall": 0.89},
            "state": full_state
        }
        
        # Create snapshot
        create_response = requests.post(
            f"{BASE_URL}/api/snapshots",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert create_response.status_code == 200
        created_id = create_response.json()["snapshot_id"]
        self.created_snapshot_ids.append(created_id)
        
        # Retrieve and verify full state is preserved
        get_response = requests.get(f"{BASE_URL}/api/snapshots/{created_id}")
        assert get_response.status_code == 200
        
        snapshot = get_response.json()["snapshot"]
        state = snapshot["state"]
        
        # Verify all state fields are preserved
        assert state["csvText"] == full_state["csvText"]
        assert state["targetColumn"] == full_state["targetColumn"]
        assert state["trainingResult"]["problemType"] == "classification"
        assert len(state["trainingResult"]["leaderboard"]) == 2
        assert state["shapGlobal"][0]["feature"] == "income"
        assert state["limeResult"]["explanation"][0]["feature"] == "income"
        assert state["cleaningLog"] == full_state["cleaningLog"]
        
        print(f"Full analysis state preserved correctly in snapshot {created_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
