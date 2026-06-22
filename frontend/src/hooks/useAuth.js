import { useState, useEffect, useCallback, useRef } from 'react';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

export function useAuth() {
  const [authUser, setAuthUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const authProcessedRef = useRef(false);

  useEffect(() => {
    if (authProcessedRef.current) return;
    const hash = window.location.hash;
    if (hash.includes('session_id=')) {
      authProcessedRef.current = true;
      const sessionId = hash.split('session_id=')[1]?.split('&')[0];
      if (sessionId) {
        fetch(`${API_URL}/api/auth/google`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId }),
        }).then(r => r.json()).then(data => {
          if (data.token) localStorage.setItem('automl_token', data.token);
          if (data.user) { setAuthUser(data.user); setAuthChecked(true); }
          else { setAuthUser(false); setAuthChecked(true); }
          window.history.replaceState({}, '', window.location.pathname);
        }).catch(() => { setAuthUser(false); setAuthChecked(true); window.history.replaceState({}, '', window.location.pathname); });
        return;
      }
    }
    const token = localStorage.getItem('automl_token');
    if (!token) { setAuthUser(false); setAuthChecked(true); return; }
    fetch(`${API_URL}/api/auth/me`, { headers: { 'Authorization': `Bearer ${token}` } })
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(user => { setAuthUser(user); setAuthChecked(true); })
      .catch(() => { localStorage.removeItem('automl_token'); setAuthUser(false); setAuthChecked(true); });
  }, []);

  const handleLogout = useCallback(async () => {
    const token = localStorage.getItem('automl_token');
    await fetch(`${API_URL}/api/auth/logout`, { method: 'POST', headers: token ? { 'Authorization': `Bearer ${token}` } : {} });
    localStorage.removeItem('automl_token');
    setAuthUser(false);
  }, []);

  return { authUser, setAuthUser, authChecked, handleLogout };
}
