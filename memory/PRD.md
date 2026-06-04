# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), and predictions — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & sample datasets (Loan Approval, House Prices, Insurance Costs, Customer Churn, Customer Segmentation)
- [x] Dataset profiling & auto-summary
- [x] Dataset scanner (quality score, missing values, outliers, duplicates)
- [x] Data cleaning (fill missing, remove duplicates, remove outliers, drop constants, normalize)
- [x] Auto algorithm selection
- [x] 10+ ML algorithms (Linear/Ridge/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, Baseline)
- [x] K-fold cross-validation
- [x] Single & batch predictions
- [x] Unsupervised learning (K-Means, Hierarchical, DBSCAN, GMM)
- [x] Anomaly detection (Z-Score, IQR)

### Explainability
- [x] SHAP (Global, Local, Beeswarm, Dependence plots)
- [x] LIME explanations
- [x] Feature importance
- [x] Business interpretation text
- [x] Decision Tree inline visualization

### Auth & UX
- [x] JWT session-based auth (localStorage tokens)
- [x] Google OAuth integration
- [x] Forgot Password / Reset Password flow
- [x] Compare Models view (radar chart, metric table, feature importance, confusion matrix, winner recommendation)
- [x] Dark/Light mode
- [x] Analysis history (save/load/share/delete snapshots)
- [x] Model import/export (download/upload JSON)
- [x] Secure model serialization (Pickle HMAC)

### Model Leaderboard (Completed Feb 2026)
- [x] Auto-save trained models to leaderboard after training
- [x] Dedicated sidebar tab with LeaderboardView
- [x] Stats row, timeline chart, algo trend chart, ranked table with sorting/filtering
- [x] Dashboard compact widget (top 5 models)
- [x] Backend CRUD APIs: GET/POST/DELETE /api/leaderboard

### Dataset Context & Analysis Management (Fixed Feb 2026)
- [x] Full state reset when loading a new dataset
- [x] Auto-save current analysis to History before switching datasets (with toast notification)
- [x] Dataset name badge in header showing current dataset (sample name or filename)
- [x] Dataset name displayed in History entries with Database icon
- [x] Better auto-save naming: "Dataset — Target — Date"
- [x] Dataset name persists in session and restores from History snapshots
- [x] Cleaning actions don't reset analysis state

## File Structure
```
src/
├── App.js                          (~1800 lines - state + routing)
├── AuthPage.js                     (auth + forgot password)
├── constants.js                    (ALGO_NAMES, GUIDE_STEPS, etc.)
├── context/AppContext.js           (React Context for shared state)
├── utils/
│   ├── helpers.js, mlEngine.js, datasetUtils.js
├── components/
│   ├── SmartTooltip.js
│   └── views/
│       ├── DashboardView.js, AnalysisView.js, PredictView.js
│       ├── ExplainabilityView.js, DataExplorerView.js
│       ├── CompareModelsView.js, LeaderboardView.js
│       ├── HistoryView.js, SmallViews.js
```

## API Endpoints
- POST /api/auth/signup, /api/auth/login, /api/auth/logout, /api/auth/google
- POST /api/auth/forgot-password, /api/auth/reset-password
- GET /api/auth/me
- POST /api/train, /api/predict
- GET/POST/DELETE /api/snapshots, POST /api/snapshots/{id}/share
- GET /api/download-model/{model_id}
- GET/POST/DELETE /api/leaderboard

## DB Schema
- users: {email, password_hash, name, picture, auth_provider}
- user_sessions: {session_token, user_id, expires_at}
- analysis_snapshots: {snapshot_id, user_id, data, fingerprint, createdAt, dataset_name}
- password_reset_tokens: {email, token, expires_at, used, created_at}
- leaderboard_entries: {user_id, model_id, algorithm, problem_type, dataset_name, metrics, ...}

## Test Credentials
- Email: test@automl.com / Password: Test1234!
- See /app/memory/test_credentials.md for full list

## Known Issues
- Token stored in localStorage (httpOnly cookies cause Kubernetes proxy fetch hangs)
- Some unused variable warnings in App.js (cosmetic, from refactor)

## Backlog
- [ ] P1: React Hook dependency issues & expensive JSX optimization
- [ ] P1: Array index as key in React lists (58 instances)
- [ ] P1: Real-time Collaborative Sessions
- [ ] P1: Model Deployment API
- [ ] P1: Automated Report Generation (PDF/HTML)
- [ ] P2: Backend server.py refactoring (high complexity functions)
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: "What-If" Analyzer
- [ ] P2: Interactive Tutorial Mode
- [ ] P3: Dataset preprocessing pipeline UI
