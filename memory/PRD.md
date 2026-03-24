# AutoML Master - Product Requirements Document

## Original Problem Statement
Full-stack AutoML application (React + FastAPI + MongoDB) that enables users to upload CSV datasets, train machine learning models, visualize results, and generate explanations. The app provides automated model training, comparison, and explainability tools.

## Architecture
- **Frontend:** React SPA (`/app/frontend/src/App.js`) with Shadcn UI components, Framer Motion animations, Recharts for data visualization, Sonner for toast notifications
- **Backend:** FastAPI (`/app/backend/server.py`) with MongoDB for persistence
- **Database:** MongoDB (users, snapshots collections)
- **ML:** 100% client-side JavaScript (scikit-learn style algorithms in App.js + explainableAI.js)
- **Auth:** Email/password JWT + Google OAuth

## Core Features (Implemented)
- [x] User auth (signup/login/Google OAuth)
- [x] CSV file upload and data profiling
- [x] Automated model training (Auto mode trains all compatible algorithms)
- [x] Model leaderboard with ranking
- [x] Feature importance charts
- [x] Regression visualizations (Actual vs Predicted, Residual Analysis)
- [x] Cross-validation support
- [x] Single prediction with form inputs
- [x] Batch prediction from CSV
- [x] Model download (pickle format)
- [x] Unsupervised learning (K-Means, DBSCAN)
- [x] Dark/Light mode
- [x] Shareable analysis snapshots
- [x] History with auto-save

## 7-Point Validation Fixes (Completed - March 24, 2026)
- [x] **P0: Export System** - Download CSV, Share Analysis, Export to Google Sheets all working with toast feedback
- [x] **P1: Decision Tree Visualization** - View Tree button on leaderboard, modal with interactive flowchart
- [x] **P1: UI Floating/Clipping** - SmartTooltip component with viewport-aware positioning
- [x] **P1: Visualization Quality** - Improved chart heights, margins, Y-axis widths, label spacing
- [x] **P1: SHAP + LIME Explanations** - Plain-English panels explaining "What does this mean?" and "Why this prediction?"
- [x] **P1: History Deduplication** - Fingerprint-based dedup (dsName|target|models|evalMode), debounced auto-save, saveInProgressRef
- [x] **P1: Overall Stability** - Try/catch in all handlers, null guards, loading states on export buttons

## Key API Endpoints
- POST /api/auth/signup, /api/auth/login, /api/auth/logout
- GET /api/auth/me
- POST /api/train, /api/predict
- POST /api/snapshots (create/update with fingerprint dedup)
- GET /api/snapshots (list), GET /api/snapshots/{id} (view)
- DELETE /api/snapshots/{id}
- POST /api/export/prepare, GET /api/export/download/{token}
- GET /api/models, DELETE /api/models/{id}, GET /api/models/{id}/download

## Test Credentials
- Email: test@automl.com / Password: Test1234!

## Backlog

### P1 - Upcoming
- [ ] Real-time Collaborative Sessions
- [ ] Model Deployment API
- [ ] Automated Report Generation (PDF/HTML)

### P2 - Future
- [ ] Advanced hyperparameter tuning UI
- [ ] Interactive Tutorial Mode
- [ ] Metric Comparison Radar Chart
- [ ] Performance Benchmark Mode
- [ ] "What-If" Analyzer
- [ ] Counterfactual Explanations

### P3 - Low Priority
- [ ] Dataset preprocessing pipeline UI
- [ ] Real-time model comparison dashboard

### Refactoring
- [ ] Break App.js (~4500 lines) into modular components
- [ ] Split server.py (~1150 lines) into routes/models/logic modules
