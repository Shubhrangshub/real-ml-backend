# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard (AutoML Master) in React. All ML analysis runs directly in the browser using JavaScript libraries — no Python backend or server-side ML needed.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Custom client-side JS (supervised in App.js, unsupervised in unsupervisedML.js)
- **XAI Engine**: explainableAI.js (SHAP approximation + LIME)
- **Backend**: FastAPI + MongoDB (snapshots API for history & sharing)
- **All ML**: Runs in-browser — no server calls for training/prediction/explanation

## Core Files
- `/app/frontend/src/App.js` — Main UI + supervised ML + all features (~3550 lines)
- `/app/frontend/src/unsupervisedML.js` — Unsupervised ML engine
- `/app/frontend/src/explainableAI.js` — SHAP & LIME computation engine
- `/app/backend/server.py` — FastAPI backend with snapshot CRUD endpoints

## Completed Features

### Phase 1-8 (Previous)
- [x] Supervised learning (11 algorithms), Unsupervised learning, Dataset Scanner, K-Fold CV
- [x] UI/UX overhaul, Algorithm leaderboard, Predictions, Cluster predictions

### Phase 9 (Feb 2026)
- [x] Batch Predictions, Model Export/Import, PDF Reports, Data Explorer, Dark Mode

### Phase 10 (Feb 2026)
- [x] Metric Hover Tooltips, Explainable AI (SHAP + LIME + Cluster Explanations)

### Phase 11 (Feb 2026)
- [x] Performance Optimization (arrayMin/arrayMax, KNN, RF, DT optimizations)

### Phase 12 (Feb 2026)
- [x] Session Persistence (full state → localStorage, Clear Session button)

### Phase 13 (Feb 2026)
- [x] Enhanced XAI Dashboard (7 new charts, vibrant colors, descriptions)

### Phase 14 (Feb 2026)
- [x] Smart Guided Help System (HelpTip, Guide Panel, Target Suggestion, Smart Row Suggestions)

### Phase 15 (Feb 2026)
- [x] **History & Sharing System**:
  - **Analysis History Tab**: Sidebar nav item, list view with dataset name, date, target, problem type, model scores. Actions: Restore, Share, Delete
  - **Backend API**: `POST/GET/DELETE /api/snapshots` — MongoDB-backed CRUD for analysis snapshots. Each snapshot stores full app state (dataset, models, metrics, SHAP/LIME results)
  - **Save Analysis**: Button in header saves current state to MongoDB with auto-generated name
  - **Share**: Generates unique URL (`?snapshot={id}`), copies to clipboard, shows toast with share link
  - **View-Only Mode**: Opening shared URL shows read-only dashboard with amber banner. Editing buttons (Save, Share, Clear Session) hidden. "Exit View-Only" button to unlock
  - **Export Buttons**: PDF (existing), CSV (download), JSON (download), Google Sheets (TSV copied to clipboard or downloaded)
  - **Restore**: Click any history item to fully restore dataset, training results, models, XAI results without retraining

## API Endpoints
- `GET /api/health` — Health check
- `POST /api/snapshots` — Save analysis snapshot
- `GET /api/snapshots` — List snapshots (summary only, no full state)
- `GET /api/snapshots/{id}` — Get full snapshot by ID
- `DELETE /api/snapshots/{id}` — Delete snapshot

## Testing Status
- Iteration 17: 21/21 UI + 8/8 backend tests passed (100% — History & Sharing)
- Iteration 16: 24/24 (100% — Help System)
- Iteration 15: 14/14 (100% — XAI)
- Iteration 14: 22/22 (100% — Session Persistence)
- Iteration 13: 27/27 (100% — Performance)

## Backlog
- [ ] P1: Counterfactual Explanations ("what would need to change" for different prediction)
- [ ] P2: Refactor App.js into modular components (~3550+ lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
