# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), and predictions — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & 5 sample datasets
- [x] Dataset profiling, scanner, cleaning
- [x] 10+ ML algorithms, auto selection, K-fold cross-validation
- [x] Single & batch predictions, unsupervised learning, anomaly detection

### Explainability
- [x] SHAP, LIME, feature importance, business interpretation, Decision Tree viz

### Auth & UX
- [x] JWT session auth, Google OAuth, Forgot/Reset Password
- [x] Compare Models, Dark/Light mode, History, Model import/export
- [x] Secure serialization (Pickle HMAC)

### Model Leaderboard (Feb 2026)
- [x] Auto-save to leaderboard, sidebar tab, dashboard widget, CRUD APIs

### Dataset Context Management (Feb 2026)
- [x] Full state reset on new dataset, auto-save with toast, dataset name badge

### UI Polish (Feb 2026)
- [x] Compact header, colorful sample cards, modern Model Library

### Admin Dashboard (Feb 2026)
- [x] Admin tab in sidebar (conditional on is_admin flag)
- [x] **Analytics**: Total users, active sessions, saved analyses, leaderboard models, logins, trains, recent signups, auth provider breakdown
- [x] **User Management**: List all users with search, toggle admin, disable/enable, reset password, delete user (with all data), confirmation modals
- [x] **Activity Log**: Tracks login, signup, train, save_analysis, admin actions with filters
- [x] **System Controls**: Clear all leaderboard entries, clear all snapshots (with confirmation)
- [x] Admin seeded: shubhrangshub@gmail.com
- [x] Non-admin users cannot see or access admin features (403 protection)

## DB Collections
- users, user_sessions, analysis_snapshots, password_reset_tokens
- leaderboard_entries, activity_log

## Admin Access
- Default admin: shubhrangshub@gmail.com
- Admin can designate other admins via Users tab

## Backlog
- [ ] P1: React Hook dependency issues & array key optimization
- [ ] P1: Real-time Collaborative Sessions
- [ ] P1: Model Deployment API
- [ ] P1: Automated Report Generation (PDF/HTML)
- [ ] P2: Backend server.py refactoring
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: "What-If" Analyzer, Interactive Tutorial Mode
- [ ] P3: Dataset preprocessing pipeline UI
