import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { AlertCircle, Mail, Lock, User, Zap, ArrowLeft, CheckCircle2, KeyRound } from 'lucide-react';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState('login'); // 'login' | 'signup' | 'forgot' | 'reset'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); setSuccess(''); setLoading(true);
    try {
      const endpoint = mode === 'signup' ? '/api/auth/signup' : '/api/auth/login';
      const body = mode === 'signup' ? { email, password, name } : { email, password };
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail;
        if (Array.isArray(detail)) {
          setError(detail.map(d => d.msg || d.message || JSON.stringify(d)).join('. '));
        } else {
          setError(typeof detail === 'string' ? detail : 'Authentication failed. Please check your details.');
        }
        setLoading(false); return;
      }
      if (data.token) localStorage.setItem('automl_token', data.token);
      if (data.user) onAuth(data.user);
    } catch (err) {
      console.error('Auth error:', err);
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Unable to connect to server. Please check your internet connection and try again.');
      } else {
        setError(err.message || 'Something went wrong. Please try again.');
      }
    }
    setLoading(false);
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    setError(''); setSuccess(''); setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/auth/forgot-password`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(typeof data.detail === 'string' ? data.detail : 'Something went wrong.');
        setLoading(false); return;
      }
      if (data.status === 'error') {
        setError(data.message);
        setLoading(false); return;
      }
      if (data.token) {
        setResetToken(data.token);
        setMode('reset');
        setSuccess('Reset token generated. Enter your new password below.');
      } else {
        setSuccess(data.message || 'If that email is registered, check your email for reset instructions.');
      }
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Unable to connect to server.');
      } else {
        setError(err.message || 'Something went wrong.');
      }
    }
    setLoading(false);
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError(''); setSuccess(''); setLoading(true);
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      setLoading(false); return;
    }
    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters.');
      setLoading(false); return;
    }
    try {
      const res = await fetch(`${API_URL}/api/auth/reset-password`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: resetToken, new_password: newPassword }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(typeof data.detail === 'string' ? data.detail : 'Reset failed.');
        setLoading(false); return;
      }
      setSuccess(data.message || 'Password reset successfully!');
      setPassword('');
      setTimeout(() => {
        setMode('login');
        setSuccess('Password reset! You can now sign in with your new password.');
        setResetToken('');
        setNewPassword('');
        setConfirmPassword('');
      }, 1500);
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Unable to connect to server.');
      } else {
        setError(err.message || 'Something went wrong.');
      }
    }
    setLoading(false);
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    setError('');
    setSuccess('');
    setResetToken('');
    setNewPassword('');
    setConfirmPassword('');
  };

  const handleGoogleLogin = () => {
    // REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
    const redirectUrl = window.location.origin + '/';
    window.location.href = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`;
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-zinc-950 dark:to-zinc-900 p-4" data-testid="auth-page">
      <Card className="w-full max-w-md shadow-xl border-border/60" data-testid="auth-card">
        <CardHeader className="text-center pb-2">
          <div className="mx-auto mb-3 w-12 h-12 rounded-xl bg-gradient-to-br from-violet-600 to-pink-500 flex items-center justify-center">
            <Zap className="h-6 w-6 text-white" />
          </div>
          <CardTitle className="text-2xl font-bold">AutoML Master</CardTitle>
          <CardDescription>
            {mode === 'login' && 'Sign in to your account'}
            {mode === 'signup' && 'Create your account'}
            {mode === 'forgot' && 'Reset your password'}
            {mode === 'reset' && 'Set a new password'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">

          {/* ===== LOGIN / SIGNUP MODE ===== */}
          {(mode === 'login' || mode === 'signup') && (
            <>
              <Button variant="outline" className="w-full h-11 text-sm font-medium" onClick={handleGoogleLogin} data-testid="google-login-btn">
                <svg className="h-5 w-5 mr-2" viewBox="0 0 24 24"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
                Continue with Google
              </Button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center"><span className="w-full border-t" /></div>
                <div className="relative flex justify-center text-xs uppercase"><span className="bg-background px-2 text-muted-foreground">or</span></div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-3">
                {mode === 'signup' && (
                  <div className="space-y-1.5">
                    <Label htmlFor="name" className="text-sm">Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input id="name" placeholder="Your name" value={name} onChange={e => setName(e.target.value)} className="pl-9" required data-testid="auth-name-input" />
                    </div>
                  </div>
                )}
                <div className="space-y-1.5">
                  <Label htmlFor="email" className="text-sm">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input id="email" type="email" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} className="pl-9" required data-testid="auth-email-input" />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password" className="text-sm">Password</Label>
                    {mode === 'login' && (
                      <button type="button" onClick={() => switchMode('forgot')} className="text-xs text-violet-600 dark:text-violet-400 hover:underline" data-testid="forgot-password-link">
                        Forgot password?
                      </button>
                    )}
                  </div>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input id="password" type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} className="pl-9" required minLength={6} data-testid="auth-password-input" />
                  </div>
                </div>

                {error && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800" data-testid="auth-error">
                    <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
                    <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
                  </div>
                )}

                {success && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800" data-testid="auth-success">
                    <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                    <span className="text-sm text-green-700 dark:text-green-300">{success}</span>
                  </div>
                )}

                <Button type="submit" className="w-full h-11 bg-gradient-to-r from-violet-600 to-pink-500 hover:from-violet-700 hover:to-pink-600 text-white font-medium" disabled={loading} data-testid="auth-submit-btn">
                  {loading ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
                </Button>
              </form>

              <p className="text-center text-sm text-muted-foreground">
                {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
                <button onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')} className="text-violet-600 dark:text-violet-400 font-medium hover:underline" data-testid="auth-toggle-mode">
                  {mode === 'login' ? 'Sign up' : 'Sign in'}
                </button>
              </p>
            </>
          )}

          {/* ===== FORGOT PASSWORD MODE ===== */}
          {mode === 'forgot' && (
            <>
              <form onSubmit={handleForgotPassword} className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Enter the email address you used to sign up and we'll generate a reset link for you.
                </p>
                <div className="space-y-1.5">
                  <Label htmlFor="reset-email" className="text-sm">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input id="reset-email" type="email" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} className="pl-9" required data-testid="forgot-email-input" />
                  </div>
                </div>

                {error && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800" data-testid="auth-error">
                    <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
                    <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
                  </div>
                )}

                {success && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800" data-testid="auth-success">
                    <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                    <span className="text-sm text-green-700 dark:text-green-300">{success}</span>
                  </div>
                )}

                <Button type="submit" className="w-full h-11 bg-gradient-to-r from-violet-600 to-pink-500 hover:from-violet-700 hover:to-pink-600 text-white font-medium" disabled={loading} data-testid="forgot-submit-btn">
                  {loading ? 'Please wait...' : 'Send Reset Link'}
                </Button>
              </form>

              <button onClick={() => switchMode('login')} className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mx-auto" data-testid="back-to-login">
                <ArrowLeft className="h-3.5 w-3.5" /> Back to sign in
              </button>
            </>
          )}

          {/* ===== RESET PASSWORD MODE ===== */}
          {mode === 'reset' && (
            <>
              <form onSubmit={handleResetPassword} className="space-y-3">
                <div className="space-y-1.5">
                  <Label htmlFor="new-password" className="text-sm">New Password</Label>
                  <div className="relative">
                    <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input id="new-password" type="password" placeholder="Enter new password" value={newPassword} onChange={e => setNewPassword(e.target.value)} className="pl-9" required minLength={6} data-testid="reset-new-password-input" />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="confirm-password" className="text-sm">Confirm Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input id="confirm-password" type="password" placeholder="Confirm new password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} className="pl-9" required minLength={6} data-testid="reset-confirm-password-input" />
                  </div>
                </div>

                {error && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800" data-testid="auth-error">
                    <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
                    <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
                  </div>
                )}

                {success && (
                  <div className="flex items-center gap-2 p-2.5 rounded-md bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800" data-testid="auth-success">
                    <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                    <span className="text-sm text-green-700 dark:text-green-300">{success}</span>
                  </div>
                )}

                <Button type="submit" className="w-full h-11 bg-gradient-to-r from-violet-600 to-pink-500 hover:from-violet-700 hover:to-pink-600 text-white font-medium" disabled={loading} data-testid="reset-submit-btn">
                  {loading ? 'Resetting...' : 'Reset Password'}
                </Button>
              </form>

              <button onClick={() => switchMode('login')} className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mx-auto" data-testid="back-to-login">
                <ArrowLeft className="h-3.5 w-3.5" /> Back to sign in
              </button>
            </>
          )}

        </CardContent>
      </Card>
    </div>
  );
}
