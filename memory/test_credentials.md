# Test Credentials

## Admin Account
- Email: `shubhrangshub@gmail.com`
- Password: `MyNewPass123!`
- Role: Admin (is_admin: true)

## Standard Test Account
- Email: `test@automl.com`
- Password: `Test1234!`
- Role: Regular user

## Auth Endpoints
- `POST /api/auth/signup`, `/api/auth/login`, `/api/auth/google`
- `POST /api/auth/forgot-password`, `/api/auth/reset-password`

## Admin Endpoints (require is_admin=true)
- `GET /api/admin/users`, `PATCH /api/admin/users/{id}`, `DELETE /api/admin/users/{id}`
- `POST /api/admin/users/{id}/reset-password`
- `GET /api/admin/analytics`, `GET /api/admin/activity`
- `DELETE /api/admin/system/leaderboard`, `DELETE /api/admin/system/snapshots`

## Deploy Endpoints
- `POST /api/deploy`, `GET /api/deploy`, `PATCH /api/deploy/{id}`, `DELETE /api/deploy/{id}`
- Public (no auth): `GET /api/public/model/{id}`, `POST /api/public/predict/{id}`
