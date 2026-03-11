# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries (simple-statistics, custom implementations). No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (Normal equation linear regression, Logistic regression, Baseline models)
- **Storage**: React state (in-memory, no persistence across page refresh)
- **Backend**: DEPRECATED (FastAPI + MongoDB no longer used)

## Core Features
- CSV upload (drag-drop, paste, file picker) + 4 sample datasets
- Auto problem type detection (classification vs regression)
- Client-side model training with leaderboard (main model + baseline)
- Data leakage prevention (auto-removes ID, date, year columns)
- Feature importance visualization
- Predictions using latest trained model
- Model library (list, delete, download as JSON)
- Dashboard with real-time stats (total models, avg metric, trends)

## What's Been Implemented (Feb 2026)
- [x] Full client-side ML engine (linear regression, logistic regression, baseline)
- [x] FIX #1: Predictions tab navigation (opens own view, shows warning if no model)
- [x] FIX #2: Model persistence with setModels(prev => [...prev, newModel])
- [x] FIX #3: Total Models dashboard metric uses models.length
- [x] FIX #4: Trend indicators show N/A when value is 0
- [x] FIX #5: Predictions auto-use latest trained model
- [x] Regression visualizations (scatter plot, residual plot)
- [x] Feature importance chart
- [x] Model download (JSON export)
- [x] All 21 tests passing (100% frontend)

## Upcoming Tasks (User Requested)
- [ ] Universal automatic analysis engine
- [ ] Clustering dashboard (K-Means) - ml-kmeans already installed
- [ ] Anomaly detection module

## Backlog
- [ ] Model persistence across page refresh (localStorage or IndexedDB)
- [ ] More ML algorithms (Decision Tree, Random Forest, etc.)
- [ ] Text processing with TF-IDF for TV Shows dataset
- [ ] Batch predictions from CSV upload
- [ ] Export training reports

## Tech Stack
- React 18, Tailwind CSS, Shadcn UI, Recharts, Framer Motion
- simple-statistics, ml-kmeans, ml-regression, danfojs (installed but not yet used)

## Key File
- `/app/frontend/src/App.js` - All application logic (ML engine + UI)
