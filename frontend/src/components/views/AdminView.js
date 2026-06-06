import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Users, BarChart3, Activity, Shield, Trash2, Key, UserX, UserCheck,
  Crown, Clock, Database, Mail, Search, RefreshCw, AlertTriangle, ChevronDown
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { fadeInUp } from '../../constants';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

function getToken() {
  return localStorage.getItem('automl_token') || '';
}

async function adminFetch(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { 'Authorization': `Bearer ${getToken()}`, 'Content-Type': 'application/json', ...options.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ========================= ANALYTICS TAB =========================
function AnalyticsTab({ analytics, loading }) {
  if (loading) return <div className="flex justify-center py-12"><RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" /></div>;
  if (!analytics) return null;

  const stats = [
    { label: 'Total Users', value: analytics.total_users, icon: Users, color: 'from-blue-500 to-cyan-500' },
    { label: 'Active Sessions', value: analytics.active_sessions, icon: Activity, color: 'from-emerald-500 to-teal-500' },
    { label: 'Saved Analyses', value: analytics.total_snapshots, icon: Database, color: 'from-violet-500 to-purple-500' },
    { label: 'Leaderboard Models', value: analytics.total_leaderboard_entries, icon: BarChart3, color: 'from-amber-500 to-orange-500' },
    { label: 'Total Logins', value: analytics.total_logins, icon: Key, color: 'from-rose-500 to-pink-500' },
    { label: 'Models Trained', value: analytics.total_trains, icon: Activity, color: 'from-indigo-500 to-blue-500' },
    { label: 'Recent Signups (7d)', value: analytics.recent_signups, icon: UserCheck, color: 'from-green-500 to-emerald-500' },
    { label: 'Analyses Saved', value: analytics.total_saves, icon: Database, color: 'from-fuchsia-500 to-violet-500' },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map(s => (
          <Card key={s.label} className="overflow-hidden">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`h-10 w-10 rounded-lg bg-gradient-to-br ${s.color} flex items-center justify-center shrink-0`}>
                  <s.icon className="h-5 w-5 text-white" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{s.value}</p>
                  <p className="text-xs text-muted-foreground">{s.label}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader><CardTitle className="text-sm">Auth Provider Breakdown</CardTitle></CardHeader>
        <CardContent>
          <div className="flex gap-6">
            <div className="flex items-center gap-2"><Mail className="h-4 w-4 text-blue-500" /><span className="text-sm">Email: <strong>{analytics.email_users}</strong></span></div>
            <div className="flex items-center gap-2"><Shield className="h-4 w-4 text-emerald-500" /><span className="text-sm">Google: <strong>{analytics.google_users}</strong></span></div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ========================= USERS TAB =========================
function UsersTab({ users, loading, onRefresh }) {
  const [search, setSearch] = useState('');
  const [confirmAction, setConfirmAction] = useState(null);
  const [resetPwUser, setResetPwUser] = useState(null);
  const [newPassword, setNewPassword] = useState('');

  const filtered = users.filter(u =>
    u.email?.toLowerCase().includes(search.toLowerCase()) ||
    u.name?.toLowerCase().includes(search.toLowerCase())
  );

  const handleToggleAdmin = async (user) => {
    try {
      await adminFetch(`/api/admin/users/${user.user_id}`, {
        method: 'PATCH', body: JSON.stringify({ is_admin: !user.is_admin })
      });
      toast.success(`${user.email} ${user.is_admin ? 'removed from' : 'promoted to'} admin`);
      onRefresh();
    } catch (e) { toast.error(e.message); }
  };

  const handleToggleDisable = async (user) => {
    try {
      await adminFetch(`/api/admin/users/${user.user_id}`, {
        method: 'PATCH', body: JSON.stringify({ is_disabled: !user.is_disabled })
      });
      toast.success(`${user.email} ${user.is_disabled ? 'enabled' : 'disabled'}`);
      onRefresh();
    } catch (e) { toast.error(e.message); }
  };

  const handleDelete = async (user) => {
    try {
      await adminFetch(`/api/admin/users/${user.user_id}`, { method: 'DELETE' });
      toast.success(`Deleted ${user.email} and all their data`);
      onRefresh();
      setConfirmAction(null);
    } catch (e) { toast.error(e.message); }
  };

  const handleResetPassword = async () => {
    if (!resetPwUser || newPassword.length < 6) { toast.error('Password must be at least 6 characters'); return; }
    try {
      await adminFetch(`/api/admin/users/${resetPwUser.user_id}/reset-password`, {
        method: 'POST', body: JSON.stringify({ new_password: newPassword })
      });
      toast.success(`Password reset for ${resetPwUser.email}`);
      setResetPwUser(null);
      setNewPassword('');
    } catch (e) { toast.error(e.message); }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input type="text" placeholder="Search by email or name..." value={search} onChange={e => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30" data-testid="admin-user-search" />
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh} data-testid="admin-refresh-users"><RefreshCw className="h-3.5 w-3.5 mr-1.5" />Refresh</Button>
        <Badge variant="secondary">{filtered.length} users</Badge>
      </div>

      {loading ? <div className="flex justify-center py-12"><RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" /></div> : (
        <div className="space-y-2">
          {filtered.map((user, idx) => (
            <div key={user.user_id} className="group flex items-center gap-4 p-3 rounded-xl border hover:shadow-sm transition-all hover:bg-accent/20" data-testid={`admin-user-row-${idx}`}>
              <div className={`h-10 w-10 rounded-full flex items-center justify-center shrink-0 text-white text-sm font-bold ${user.is_admin ? 'bg-gradient-to-br from-amber-400 to-orange-500' : user.is_disabled ? 'bg-gray-400' : 'bg-gradient-to-br from-violet-500 to-fuchsia-500'}`}>
                {user.picture ? <img src={user.picture} alt="" className="h-10 w-10 rounded-full" referrerPolicy="no-referrer" /> : (user.name || user.email)?.[0]?.toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm truncate">{user.name || user.email}</span>
                  {user.is_admin && <Badge className="text-[10px] px-1.5 py-0 bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 border-0">Admin</Badge>}
                  {user.is_disabled && <Badge className="text-[10px] px-1.5 py-0 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 border-0">Disabled</Badge>}
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0">{user.auth_provider || 'email'}</Badge>
                </div>
                <div className="flex items-center gap-4 mt-0.5 text-xs text-muted-foreground">
                  <span>{user.email}</span>
                  <span className="flex items-center gap-1"><Clock className="h-3 w-3" />Joined {user.created_at ? new Date(user.created_at).toLocaleDateString() : '?'}</span>
                  <span>{user.snapshots_count} analyses</span>
                  <span>{user.leaderboard_count} models</span>
                </div>
              </div>
              <div className="flex gap-1 opacity-40 group-hover:opacity-100 transition-opacity">
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleToggleAdmin(user)} title={user.is_admin ? 'Remove admin' : 'Make admin'} data-testid={`admin-toggle-admin-${idx}`}>
                  <Crown className={`h-4 w-4 ${user.is_admin ? 'text-amber-500' : 'text-muted-foreground'}`} />
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleToggleDisable(user)} title={user.is_disabled ? 'Enable user' : 'Disable user'} data-testid={`admin-toggle-disable-${idx}`}>
                  {user.is_disabled ? <UserCheck className="h-4 w-4 text-green-500" /> : <UserX className="h-4 w-4 text-muted-foreground" />}
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setResetPwUser(user)} title="Reset password" data-testid={`admin-reset-pw-${idx}`}>
                  <Key className="h-4 w-4 text-blue-500" />
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setConfirmAction(user)} title="Delete user" data-testid={`admin-delete-user-${idx}`}>
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Delete confirmation modal */}
      {confirmAction && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setConfirmAction(null)}>
          <Card className="w-full max-w-md mx-4" onClick={e => e.stopPropagation()} data-testid="admin-delete-confirm-modal">
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-3"><AlertTriangle className="h-8 w-8 text-destructive" /><div><h3 className="font-bold">Delete User</h3><p className="text-sm text-muted-foreground">This will permanently delete <strong>{confirmAction.email}</strong> and all their data (analyses, models, sessions).</p></div></div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setConfirmAction(null)}>Cancel</Button>
                <Button variant="destructive" onClick={() => handleDelete(confirmAction)} data-testid="admin-confirm-delete">Delete User</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Reset password modal */}
      {resetPwUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => { setResetPwUser(null); setNewPassword(''); }}>
          <Card className="w-full max-w-md mx-4" onClick={e => e.stopPropagation()} data-testid="admin-reset-pw-modal">
            <CardContent className="p-6 space-y-4">
              <div><h3 className="font-bold">Reset Password</h3><p className="text-sm text-muted-foreground">Set a new password for <strong>{resetPwUser.email}</strong></p></div>
              <input type="text" placeholder="New password (min 6 chars)" value={newPassword} onChange={e => setNewPassword(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30" data-testid="admin-new-password-input" />
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => { setResetPwUser(null); setNewPassword(''); }}>Cancel</Button>
                <Button onClick={handleResetPassword} disabled={newPassword.length < 6} data-testid="admin-confirm-reset-pw">Reset Password</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

// ========================= ACTIVITY TAB =========================
function ActivityTab({ activities, loading, onRefresh }) {
  const [filter, setFilter] = useState('');

  const actionColors = {
    login: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    signup: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    train: 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400',
    save_analysis: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
    admin_update_user: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
    admin_delete_user: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    admin_reset_password: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
    admin_clear_leaderboard: 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400',
  };

  const actionTypes = ['', 'login', 'signup', 'train', 'save_analysis', 'admin_update_user', 'admin_delete_user'];

  const filtered = filter ? activities.filter(a => a.action === filter) : activities;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative">
          <select value={filter} onChange={e => setFilter(e.target.value)}
            className="appearance-none pl-3 pr-8 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30" data-testid="admin-activity-filter">
            <option value="">All Actions</option>
            {actionTypes.filter(Boolean).map(a => <option key={a} value={a}>{a.replace(/_/g, ' ')}</option>)}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh}><RefreshCw className="h-3.5 w-3.5 mr-1.5" />Refresh</Button>
        <Badge variant="secondary">{filtered.length} events</Badge>
      </div>

      {loading ? <div className="flex justify-center py-12"><RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" /></div> : (
        <div className="space-y-1.5">
          {filtered.length === 0 && <div className="text-center py-12 text-muted-foreground text-sm">No activity recorded yet. Actions will appear as users interact with the platform.</div>}
          {filtered.map((act, idx) => (
            <div key={idx} className="flex items-center gap-3 p-2.5 rounded-lg border hover:bg-accent/20 transition-colors text-sm" data-testid={`admin-activity-${idx}`}>
              <Badge className={`text-[10px] px-2 py-0.5 border-0 whitespace-nowrap ${actionColors[act.action] || 'bg-gray-100 text-gray-700'}`}>{act.action?.replace(/_/g, ' ')}</Badge>
              <span className="font-medium truncate max-w-[200px]">{act.email || act.user_id}</span>
              <span className="text-muted-foreground truncate flex-1">{act.details}</span>
              <span className="text-xs text-muted-foreground whitespace-nowrap flex items-center gap-1"><Clock className="h-3 w-3" />{act.timestamp ? new Date(act.timestamp).toLocaleString() : '?'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ========================= SYSTEM TAB =========================
function SystemTab() {
  const [confirmClear, setConfirmClear] = useState(null);

  const handleClear = async (type) => {
    try {
      const res = await adminFetch(`/api/admin/system/${type}`, { method: 'DELETE' });
      toast.success(`Cleared ${res.deleted} ${type} entries`);
      setConfirmClear(null);
    } catch (e) { toast.error(e.message); }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader><CardTitle className="text-sm flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-destructive" />Danger Zone</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-lg border border-destructive/20 bg-destructive/5">
            <div><p className="font-medium text-sm">Clear All Leaderboard Entries</p><p className="text-xs text-muted-foreground">Permanently deletes ALL leaderboard entries for ALL users.</p></div>
            <Button variant="destructive" size="sm" onClick={() => setConfirmClear('leaderboard')} data-testid="admin-clear-leaderboard">Clear All</Button>
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border border-destructive/20 bg-destructive/5">
            <div><p className="font-medium text-sm">Clear All Saved Analyses</p><p className="text-xs text-muted-foreground">Permanently deletes ALL saved analysis snapshots for ALL users.</p></div>
            <Button variant="destructive" size="sm" onClick={() => setConfirmClear('snapshots')} data-testid="admin-clear-snapshots">Clear All</Button>
          </div>
        </CardContent>
      </Card>

      {confirmClear && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setConfirmClear(null)}>
          <Card className="w-full max-w-md mx-4" onClick={e => e.stopPropagation()}>
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-3"><AlertTriangle className="h-8 w-8 text-destructive" /><div><h3 className="font-bold">Confirm Clear All</h3><p className="text-sm text-muted-foreground">This will permanently delete ALL {confirmClear} entries for ALL users. This cannot be undone.</p></div></div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setConfirmClear(null)}>Cancel</Button>
                <Button variant="destructive" onClick={() => handleClear(confirmClear)} data-testid="admin-confirm-clear">Delete Everything</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

// ========================= MAIN ADMIN VIEW =========================
export default function AdminView() {
  const [tab, setTab] = useState('analytics');
  const [analytics, setAnalytics] = useState(null);
  const [users, setUsers] = useState([]);
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    try { const d = await adminFetch('/api/admin/analytics'); setAnalytics(d); } catch (e) { toast.error(e.message); }
    setLoading(false);
  }, []);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try { const d = await adminFetch('/api/admin/users'); setUsers(d.users || []); } catch (e) { toast.error(e.message); }
    setLoading(false);
  }, []);

  const fetchActivity = useCallback(async () => {
    setLoading(true);
    try { const d = await adminFetch('/api/admin/activity?limit=100'); setActivities(d.activities || []); } catch (e) { toast.error(e.message); }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAnalytics();
    fetchUsers();
    fetchActivity();
  }, [fetchAnalytics, fetchUsers, fetchActivity]);

  const tabs = [
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'activity', label: 'Activity', icon: Activity },
    { id: 'system', label: 'System', icon: Shield },
  ];

  return (
    <motion.div variants={fadeInUp} initial="initial" animate="animate" className="space-y-6" data-testid="admin-view">
      <div className="flex items-center gap-2 border-b pb-3">
        {tabs.map(t => (
          <Button key={t.id} variant={tab === t.id ? 'default' : 'ghost'} size="sm"
            className={`gap-2 ${tab === t.id ? 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600' : ''}`}
            onClick={() => setTab(t.id)} data-testid={`admin-tab-${t.id}`}>
            <t.icon className="h-3.5 w-3.5" />{t.label}
          </Button>
        ))}
      </div>

      {tab === 'analytics' && <AnalyticsTab analytics={analytics} loading={loading} />}
      {tab === 'users' && <UsersTab users={users} loading={loading} onRefresh={fetchUsers} />}
      {tab === 'activity' && <ActivityTab activities={activities} loading={loading} onRefresh={fetchActivity} />}
      {tab === 'system' && <SystemTab />}
    </motion.div>
  );
}
