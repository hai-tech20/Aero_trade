import { useState, useEffect, useMemo } from 'react'

const API = ''

function Logo({ size = 48, src = '/aero_trade_logo.png' }) {
  return (
    <img src={src} alt="AERO TRADE"
      style={{ height: size, width: size, objectFit: 'contain', borderRadius: 10 }} />
  )
}

function Splash() {
  return (
    <div className="splash">
      <Logo size={120} />
      <h1>AERO TRADE</h1>
      <p>Loading...</p>
    </div>
  )
}

function BrokerSelect({ onSelect }) {
  const [brokers, setBrokers] = useState([])
  const [query, setQuery] = useState('')

  useEffect(() => {
    fetch(`${API}/brokers`).then(r => r.json()).then(setBrokers).catch(() => {})
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const sorted = [...brokers].sort((a, b) => {
      if (a.id === 'custom') return 1
      if (b.id === 'custom') return -1
      return a.display_name.localeCompare(b.display_name)
    })
    if (!q) return sorted
    return sorted.filter(b => b.display_name.toLowerCase().includes(q))
  }, [brokers, query])

  return (
    <div className="screen">
      <header className="header">
        <Logo size={40} />
        <h1>AERO TRADE</h1>
        <p>Select your broker</p>
      </header>

      <div className="form" style={{ marginBottom: '1rem' }}>
        <label>
          Search brokers
          <input
            placeholder="Type broker name..."
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </label>
      </div>

      <div className="broker-grid">
        {filtered.map(b => (
          <button key={b.id} className="broker-card" onClick={() => onSelect(b)}>
            <div className="broker-card-top">
              <Logo size={36} src={b.logo_path || '/aero_trade_logo.png'} />
              <div>
                <strong>{b.display_name}</strong>
                <span>{b.servers?.length ? b.servers.join(' / ') : 'Custom server entry'}</span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function CredentialEntry({ broker, onSuccess, onBack }) {
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [server, setServer] = useState(broker?.servers?.[0] || '')
  const [customServer, setCustomServer] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [needsAccountTypeConfirm, setNeedsAccountTypeConfirm] = useState(false)
  const [confirmedAccountType, setConfirmedAccountType] = useState('demo')

  useEffect(() => {
    setServer(broker?.servers?.[0] || '')
    setCustomServer('')
    setNeedsAccountTypeConfirm(false)
    setConfirmedAccountType('demo')
    setError('')
  }, [broker])

  const selectedServer = broker.id === 'custom' ? customServer.trim() : server

  async function submitAccount(isRetry = false) {
    const payload = {
      broker_id: broker.id,
      login: parseInt(login, 10),
      password,
      server: selectedServer,
      custom_server: broker.id === 'custom' ? customServer.trim() : '',
      is_demo_confirmed: needsAccountTypeConfirm || isRetry ? confirmedAccountType === 'demo' : null,
    }

    const res = await fetch(`${API}/accounts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      const detail = data.detail
      if (detail?.code === 'ambiguous_server_type') {
        setNeedsAccountTypeConfirm(true)
        setError(detail.message || 'Please confirm whether this server is demo or real.')
        return null
      }
      throw new Error((typeof detail === 'string' && detail) || data.message || 'Verification failed')
    }

    return data
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await submitAccount(false)
      if (data) onSuccess(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleConfirmedSubmit() {
    setLoading(true)
    setError('')
    try {
      const data = await submitAccount(true)
      if (data) onSuccess(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="screen">
      <header className="header">
        <Logo size={36} src={broker.logo_path || '/aero_trade_logo.png'} />
        <div>
          <h2>{broker.display_name}</h2>
          <p>Enter your MT5 login, password, and server.</p>
        </div>
        <button className="back-btn" onClick={onBack}>← Back</button>
      </header>

      <form className="form" onSubmit={handleSubmit}>
        <label>
          Account Number
          <input value={login} onChange={e => setLogin(e.target.value)} required inputMode="numeric" />
        </label>

        <label>
          Password
          <input
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
        </label>

        <label className="toggle">
          <input type="checkbox" checked={showPassword} onChange={e => setShowPassword(e.target.checked)} />
          Show Password
        </label>

        {broker.id === 'custom' ? (
          <label>
            Server
            <input
              value={customServer}
              onChange={e => setCustomServer(e.target.value)}
              placeholder="Enter exact MT5 server name"
              required
            />
          </label>
        ) : (
          <label>
            Server
            <select value={server} onChange={e => setServer(e.target.value)} required>
              {broker.servers.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
        )}

        {needsAccountTypeConfirm && (
          <div className="confirm-panel">
            <h3>Is this a Demo or Real account?</h3>
            <p>
              The custom server name <strong>{selectedServer}</strong> is ambiguous. We will not guess.
            </p>
            <div className="server-choice-group">
              <label className="server-option">
                <input
                  type="radio"
                  name="accountType"
                  checked={confirmedAccountType === 'demo'}
                  onChange={() => setConfirmedAccountType('demo')}
                />
                Demo
              </label>
              <label className="server-option">
                <input
                  type="radio"
                  name="accountType"
                  checked={confirmedAccountType === 'real'}
                  onChange={() => setConfirmedAccountType('real')}
                />
                Real
              </label>
            </div>
            <button type="button" onClick={handleConfirmedSubmit} disabled={loading}>
              {loading ? 'Verifying...' : 'Confirm & Verify'}
            </button>
          </div>
        )}

        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={loading || !selectedServer}>
          {loading ? 'Verifying...' : 'Connect & Verify'}
        </button>
      </form>
    </div>
  )
}

function ModelStatusPanel() {
  const [models, setModels] = useState([])

  useEffect(() => {
    let mounted = true
    const load = async () => {
      const res = await fetch(`${API}/models/status`).then(r => r.json()).catch(() => ({ models: [] }))
      if (mounted) setModels(res.models || [])
    }
    load()
    const iv = setInterval(load, 10000)
    return () => { mounted = false; clearInterval(iv) }
  }, [])

  return (
    <section>
      <details>
        <summary style={{cursor: 'pointer', fontWeight: 'bold', fontSize: '1.1em'}}>Model Status (click to expand)</summary>
      <div className="model-status-table-wrap">
        <table className="model-status-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>TF</th>
              <th>History</th>
              <th>Rows</th>
              <th>Date Range</th>
              <th>Model Version</th>
              <th>Accuracy</th>
              <th>Last Retrain</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {models.map(row => {
              const weak = row.weak_signal
              const missing = !row.history_available
              return (
                <tr key={`${row.symbol}-${row.timeframe}`} className={missing ? 'status-missing' : weak ? 'status-weak' : ''}>
                  <td>{row.symbol}</td>
                  <td>{row.timeframe}</td>
                  <td>{row.history_available ? 'Yes' : 'No'}</td>
                  <td>{row.history_rows || 0}</td>
                  <td>{row.history_date_range?.[0] && row.history_date_range?.[1] ? `${row.history_date_range[0]} → ${row.history_date_range[1]}` : '—'}</td>
                  <td>{row.active_model_version || '—'}</td>
                  <td>{row.active_model_accuracy != null ? Number(row.active_model_accuracy).toFixed(4) : '—'}</td>
                  <td>{row.last_retrain_at || '—'}</td>
                  <td>{row.last_retrain_outcome || '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      </details>
    </section>
  )
}

function Dashboard({ account, onSettings }) {
  const [info, setInfo] = useState(account)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState({ trades: [], summary: {} })
  const [running, setRunning] = useState(false)
  const [activity, setActivity] = useState([])
  const [toggling, setToggling] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const id = account.account_id
    let mounted = true
    const load = async () => {
      try {
        const [accRes, posRes, tradesRes, actRes] = await Promise.all([
          fetch(`${API}/accounts/${id}`).then(r => { if (!r.ok) throw new Error(); return r.json(); }).catch(() => null),
          fetch(`${API}/accounts/${id}/positions`).then(r => { if (!r.ok) throw new Error(); return r.json(); }).catch(() => null),
          fetch(`${API}/accounts/${id}/trades`).then(r => { if (!r.ok) throw new Error(); return r.json(); }).catch(() => null),
          fetch(`${API}/accounts/${id}/activity?limit=50`).then(r => { if (!r.ok) throw new Error(); return r.json(); }).catch(() => ({ activity: [] })),
        ])
        if (!mounted) return
        if (accRes) setInfo(prev => ({ ...prev, ...accRes }))
        if (posRes) setPositions(Array.isArray(posRes) ? posRes : [])
        if (tradesRes) setTrades(tradesRes)
        if (actRes) setActivity(actRes.activity || [])

        if (!toggling) {
          const isRunning = actRes && actRes.activity && actRes.activity.length > 0 && actRes.activity.some(a => a.message && a.message.includes('cycle'))
          setRunning(isRunning)
        }
      } catch {
      }
    }
    load()
    const iv = setInterval(load, 5000)
    return () => { mounted = false; clearInterval(iv) }
  }, [account.account_id, toggling])

  async function toggleBot() {
    const id = account.account_id
    setToggling(true)
    setError('')
    try {
      const endpoint = running ? 'stop' : 'start'
      const res = await fetch(`${API}/accounts/${id}/${endpoint}`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Request failed with ${res.status}`)
      }
      await new Promise(r => setTimeout(r, 800))
      const actRes = await fetch(`${API}/accounts/${id}/activity?limit=3`).then(r => r.json()).catch(() => ({ activity: [] }))
      const isRunning = actRes.activity && actRes.activity.some(a => a.message && a.message.includes('cycle'))
      setRunning(isRunning)
    } catch (err) {
      setError(err.message)
    } finally {
      setToggling(false)
    }
  }

  const s = trades.summary || {}
  return (
    <div className="screen dashboard">
      <header className="header">
        <Logo size={36} />
        <div>
          <h2>AERO TRADE</h2>
          <span className={`badge ${info.is_demo ? 'demo' : 'live'}`}>
            {info.is_demo ? 'DEMO' : 'LIVE'}
          </span>
          <p>{info.server || 'No server reported'}</p>
        </div>
        <button className="settings-btn" onClick={onSettings}>⚙ Settings</button>
      </header>

      <div className="stats-grid">
        <div className="stat-card"><span>Balance</span><strong>${(info.balance || 0).toLocaleString()}</strong></div>
        <div className="stat-card"><span>Equity</span><strong>${(info.equity || 0).toLocaleString()}</strong></div>
        <div className="stat-card"><span>Win Rate</span><strong>{(s.win_rate || 0).toFixed(1)}%</strong></div>
        <div className="stat-card"><span>Total P&L</span><strong className={s.total_pnl >= 0 ? 'profit' : 'loss'}>${(s.total_pnl || 0).toFixed(2)}</strong></div>
      </div>

      <button className={`bot-toggle ${running ? 'stop' : 'start'}`} onClick={toggleBot} disabled={toggling}>
        {toggling ? '...' : (running ? '■ Stop Trading' : '▶ Start Trading')}
      </button>
      {error && <div className="error">{error}</div>}

      <ModelStatusPanel />

      <section>
        <h3>Open Positions ({positions.length})</h3>
        {positions.map(p => (
          <div key={p.order_id} className="trade-row">
            <span>{p.symbol}</span>
            <span className={p.type === 'BUY' ? 'buy' : 'sell'}>{p.type}</span>
            <span>{p.volume} lots</span>
            <span className={p.profit >= 0 ? 'profit' : 'loss'}>${(p.profit || 0).toFixed(2)}</span>
          </div>
        ))}
      </section>

      <section>
        <h3>Trade History</h3>
        {(trades.trades || []).map(t => (
          <div key={t.order_id} className="trade-row">
            <span>{t.symbol}</span>
            <span>{t.direction}</span>
            <span className={t.outcome === 'win' ? 'profit' : 'loss'}>
              ${(t.pnl_usd || 0).toFixed(2)} ({t.outcome})
            </span>
          </div>
        ))}
      </section>

      <section>
        <h3>Bot Activity</h3>
        <div className="activity-feed">
          {activity.map((a, i) => (
            <div key={i} className="activity-row">
              <span className="activity-time">{new Date(a.timestamp).toLocaleTimeString()}</span>
              <span className="activity-symbol">{a.symbol}</span>
              <span className="activity-msg">{a.message}</span>
            </div>
          ))}
          {activity.length === 0 && <div className="muted">No activity yet. Start trading to see live updates.</div>}
        </div>
      </section>
    </div>
  )
}

function Settings({ accountId, onBack }) {
  const [settings, setSettings] = useState({})
  const [livePassword, setLivePassword] = useState('')

  useEffect(() => {
    fetch(`${API}/accounts/${accountId}/settings`).then(r => r.json()).then(setSettings)
  }, [accountId])

  async function save() {
    await fetch(`${API}/accounts/${accountId}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    onBack()
  }

  async function enableLive() {
    await fetch(`${API}/accounts/${accountId}/live-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enable: true, password: livePassword }),
    })
    alert('Live trading enabled. Real money is at risk.')
    onBack()
  }

  return (
    <div className="screen">
      <header className="header">
        <h2>Settings</h2>
        <button className="back-btn" onClick={onBack}>← Back</button>
      </header>
      <div className="form">
        <label>Risk Per Trade
          <input type="number" step="0.01" value={settings.risk_per_trade || 0.02}
            onChange={e => setSettings({ ...settings, risk_per_trade: parseFloat(e.target.value) })} />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.aggressive_mode || false}
            disabled={!settings.is_demo}
            onChange={e => setSettings({ ...settings, aggressive_mode: e.target.checked })} />
          Aggressive Mode {settings.is_demo ? '(Demo only)' : '(Disabled on real)'}
        </label>
        {!settings.is_demo && (
          <div className="live-gate">
            <h3>Enable Live Trading</h3>
            <p>Real money is at risk. Re-enter your password to confirm.</p>
            <input type="password" value={livePassword} onChange={e => setLivePassword(e.target.value)} />
            <button onClick={enableLive} className="danger">Enable Live Trading</button>
          </div>
        )}
        <button onClick={save}>Save Settings</button>
      </div>
    </div>
  )
}

export default function App() {
  const [screen, setScreen] = useState('splash')
  const [broker, setBroker] = useState(null)
  const [account, setAccount] = useState(null)

  useEffect(() => {
    const t = setTimeout(() => setScreen('brokers'), 1500)
    return () => clearTimeout(t)
  }, [])

  if (screen === 'splash') return <Splash />
  if (screen === 'brokers') return <BrokerSelect onSelect={b => { setBroker(b); setScreen('credentials') }} />
  if (screen === 'credentials') return <CredentialEntry broker={broker} onBack={() => setScreen('brokers')}
    onSuccess={a => { setAccount(a); setScreen('dashboard') }} />
  if (screen === 'settings') return <Settings accountId={account.account_id} onBack={() => setScreen('dashboard')} />
  if (screen === 'dashboard') return <Dashboard account={account} onSettings={() => setScreen('settings')} />
  return null
}
