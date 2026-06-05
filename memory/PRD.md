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
- [x] Dataset profiling, scanner, cleaning
- [x] Auto algorithm selection, 10+ ML algorithms, K-fold cross-validation
- [x] Single & batch predictions
- [x] Unsupervised learning, anomaly detection, clustering

### Explainability
- [x] SHAP (Global, Local, Beeswarm, Dependence), LIME, feature importance
- [x] Business interpretation text, Decision Tree inline visualization

### Auth & UX
- [x] JWT session auth, Google OAuth, Forgot/Reset Password
- [x] Compare Models (radar chart, metrics, confusion matrix, winner)
- [x] Dark/Light mode, History (save/load/share), Model import/export
- [x] Secure model serialization (Pickle HMAC)

### Model Leaderboard (Feb 2026)
- [x] Auto-save to leaderboard, sidebar tab, dashboard widget
- [x] Stats, charts, ranked table, sorting/filtering, CRUD APIs

### Dataset Context & Analysis Management (Feb 2026)
- [x] Full state reset on new dataset, auto-save to History with toast
- [x] Dataset name badge in header, dataset name in History entries
- [x] Better naming ("Dataset — Target — Date"), session/restore persistence

### UI Polish (Feb 2026)
- [x] Compact header: single-line title + badge + compact ghost buttons
- [x] Colorful sample dataset cards (unique colors/icons per dataset)
- [x] Modern Model Library: card-based with gradient algo icons
- [x] Gradient "N Models" badge, polished user avatar with ring
- [x] Removed subtitle text overflow/wrapping issues

## File Structure
```
src/
├── App.js (~1800 lines), AuthPage.js, constants.js
├── context/AppContext.js
├── utils/helpers.js, mlEngine.js, datasetUtils.js
├── components/SmartTooltip.js
├── components/views/
│   ├── DashboardView.js, AnalysisView.js, PredictView.js
│   ├── ExplainabilityView.js, DataExplorerView.js
│   ├── CompareModelsView.js, LeaderboardView.js
│   ├── HistoryView.js, SmallViews.js
```

## API Endpoints
- Auth: signup, login, logout, google, forgot-password, reset-password, me
- ML: train, predict, snapshots CRUD, download-model
- Leaderboard: GET/POST/DELETE /api/leaderboard

## Test Credentials
- Email: test@automl.com / Password: Test1234!

## Known Issues
- Token in localStorage (httpOnly cookies cause proxy hangs)
- Some unused variable warnings in App.js (cosmetic)

## Backlog
- [ ] P1: React Hook dependency issues & expensive JSX optimization
- [ ] P1: Array index as key in React lists (58 instances)
- [ ] P1: Real-time Collaborative Sessions
- [ ] P1: Model Deployment API
- [ ] P1: Automated Report Generation (PDF/HTML)
- [ ] P2: Backend server.py refactoring
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: "What-If" Analyzer
- [ ] P2: Interactive Tutorial Mode
- [ ] P3: Dataset preprocessing pipeline UI
