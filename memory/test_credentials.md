# Test Credentials

## Admin Account
- Email: `shubhrangshub@gmail.com`
- Password: `MyNewPass123!`
- Role: Admin (is_admin: true)

## Standard Test Account
- Email: `test@automl.com`
- Password: `Test1234!`
- Role: Regular user

## Other Accounts
- Email: `shubhampol3006@gmail.com` (signed up with own password)
- Email: `radhika677phadnis@gmail.com` (signed up with own password)

## Auth Endpoints
- `POST /api/auth/signup` — Create new account
- `POST /api/auth/login` — Login with email/password
- `POST /api/auth/google` — Google OAuth login
- `POST /api/auth/forgot-password` — Generate reset token
- `POST /api/auth/reset-password` — Reset password with token

## Admin Endpoints (require is_admin=true)
- `GET /api/admin/users` — List all users with stats
- `PATCH /api/admin/users/{user_id}` — Toggle admin/disable flags
- `DELETE /api/admin/users/{user_id}` — Delete user and all data
- `POST /api/admin/users/{user_id}/reset-password` — Admin reset password
- `GET /api/admin/analytics` — Platform usage analytics
- `GET /api/admin/activity` — Activity log
- `DELETE /api/admin/system/leaderboard` — Clear all leaderboard entries
- `DELETE /api/admin/system/snapshots` — Clear all snapshots
