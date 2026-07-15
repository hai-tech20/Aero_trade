import { useState, useEffect } from 'react'

const API = ''

function Logo({ size = 48 }) {
  return (
    <img src="/aero_trade_logo.png" alt="AERO TRADE"
      style={{ height: size, objectFit: 'contain' }} />
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
  useEffect(() => {
    fetch(`${API}/brokers`).then(r => r.json()).then(setBrokers).catch(() => {})
  }, [])
  return (
    <div className="screen">
      <header className="header">
        <Logo size={40} />
        <h1>AERO TRADE</h1>
        <p>Select your broker</p>
      </header>
      <div className="broker-grid">
        {brokers.map(b => (
          <button key={b.id} className="broker-card" onClick={() => onSelect(b)}>
            <strong>{b.display_name}</strong>
            <span>{b.demo_server || 'Custom server'}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function CredentialEntry({ broker, onSuccess, onBack }) {
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [isDemo, setIsDemo] = useState(true)
  const [customServer, setCustomServer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showPassword, setShowPassword] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/accounts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          broker_id: broker.id,
          login: parseInt(login),
          password,
          is_demo: isDemo,
          custom_server: broker.id === 'custom' ? customServer : '',
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Verification failed')
      onSuccess(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="screen">
      <header className="header">
        <Logo size={36} />
        <h2>{broker.display_name}</h2>
        <button className="back-btn" onClick={onBack}>← Back</button>
      </header>
      <form className="form" onSubmit={handleSubmit}>
        <label>Account Number<input value={login} onChange={e => setLogin(e.target.value)} required /></label>
        <label>Password
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input 
              type={showPassword ? "text" : "password"} 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
              required 
              style={{ flex: 1 }}
            />
          </div>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showPassword} onChange={e => setShowPassword(e.target.checked)} />
          Show Password
        </label>
        {broker.id === 'custom' && (
          <label>Server Name<input value={customServer} onChange={e => setCustomServer(e.target.value)} required /></label>
        )}
        <label className="toggle">
          <input type="checkbox" checked={isDemo} onChange={e => setIsDemo(e.target.checked)} />
          Demo Account
        </label>
        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={loading}>{loading ? 'Verifying...' : 'Connect & Verify'}</button>
      </form>
    </div>
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
      } catch (e) {
        // silent fail for polling
      }
    }
    load()
    const iv = setInterval(load, 5000)
    return () => { mounted = false; clearInterval(iv) }
  }, [account.account_id])

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
            onChange={e => setSettings({...settings, risk_per_trade: parseFloat(e.target.value)})} />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.aggressive_mode || false}
            disabled={!settings.is_demo}
            onChange={e => setSettings({...settings, aggressive_mode: e.target.checked})} />
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
