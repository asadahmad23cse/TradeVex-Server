from pathlib import Path
import re

p = Path(r'c:\Users\KRISH\Desktop\Trading\src\dashboard\static\index.html')
html = p.read_text('utf8')
new_style = r'''<style>
:root {
  --bg-primary: #000000;
  --bg-secondary: #0a0a0a;
  --bg-card: rgba(18, 18, 20, 0.45);
  --bg-glass: rgba(18, 18, 20, 0.75);
  --border: rgba(255, 255, 255, 0.08);
  --border-glow: rgba(59, 130, 246, 0.3);
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  --accent-blue: #3b82f6;
  --accent-blue-hover: #2563eb;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-amber: #f59e0b;
  --accent-purple: #8b5cf6;
  
  --gradient-blue: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
  --gradient-green: linear-gradient(135deg, #34d399 0%, #10b981 100%);
  --gradient-red: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
  --gradient-brand: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
  
  --shadow-glow: 0 0 40px rgba(59, 130, 246, 0.15);
  --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.5);
  --shadow-inner: inset 0 1px 0 rgba(255, 255, 255, 0.05);
  
  --radius: 20px;
  --radius-sm: 12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
  position: relative;
}

body::before, body::after {
  content: '';
  position: fixed;
  border-radius: 50%;
  filter: blur(120px);
  z-index: 0;
  pointer-events: none;
  opacity: 0.4;
  animation: float 20s infinite ease-in-out alternate;
}
body::before {
  top: -10%; left: -10%; width: 500px; height: 500px;
  background: rgba(59, 130, 246, 0.15);
}
body::after {
  bottom: -10%; right: -10%; width: 600px; height: 600px;
  background: rgba(139, 92, 246, 0.15);
  animation-delay: -10s;
}

@keyframes float {
  0% { transform: translate(0, 0) scale(1); }
  100% { transform: translate(50px, 50px) scale(1.1); }
}

.topbar {
  position: sticky; top: 0; z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 32px;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}
.topbar-left { display: flex; align-items: center; gap: 20px; }
.logo {
  font-size: 22px; font-weight: 900; letter-spacing: -0.5px;
  background: var(--gradient-brand);
  background-clip: text; /* STANDARD PROP */
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent;
  display: flex; align-items: center; gap: 8px;
}
.logo span { font-weight: 400; color: var(--text-secondary); -webkit-text-fill-color: initial; background: none; }
.live-indicator {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 14px; border-radius: 30px;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  font-size: 11px; font-weight: 700; color: var(--accent-green);
  text-transform: uppercase; letter-spacing: 1.5px;
  box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
}
.live-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent-green);
  animation: livePulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes livePulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  50% { opacity: 0.5; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
}
.topbar-right { 
  display: flex; align-items: center; gap: 16px; 
  font-size: 14px; font-weight: 500; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;
}
#clock { background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; }

.tab-bar {
  display: flex; gap: 6px; padding: 12px 32px;
  background: rgba(10, 10, 10, 0.8); border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  overflow-x: auto; scrollbar-width: none;
}
.tab-btn {
  padding: 10px 20px; border: 1px solid transparent; border-radius: 10px;
  background: transparent; color: var(--text-secondary);
  font-size: 14px; font-weight: 600; cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap; display: flex; align-items: center; gap: 8px;
}
.tab-btn:hover { 
  background: rgba(255, 255, 255, 0.05); color: var(--text-primary); 
  transform: translateY(-1px);
}
.tab-btn.active {
  background: rgba(59, 130, 246, 0.15); color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.3);
  box-shadow: inset 0 0 20px rgba(59, 130, 246, 0.1);
}

.content { position: relative; z-index: 1; padding: 24px 32px; }
.tab-page { display: none; animation: fadeIn 0.4s ease-out; }
.tab-page.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.market-strip { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.market-card {
  flex: 1; min-width: 200px; padding: 20px 24px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card), var(--shadow-inner);
  position: relative; overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
}
.market-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-glow), var(--shadow-card); }
.market-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}
.market-card .label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 8px; }
.market-card .price { font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--text-primary); letter-spacing: -1px; }
.market-card .change { font-size: 14px; font-weight: 600; margin-top: 6px; display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 6px; }
.market-card .change.up { background: rgba(16, 185, 129, 0.1); color: var(--accent-green); }
.market-card .change.down { background: rgba(239, 68, 68, 0.1); color: var(--accent-red); }

.main-grid { display: grid; grid-template-columns: 360px 1fr; gap: 24px; height: calc(100vh - 240px); }
@media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; height: auto; } }

.stock-list-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-card); backdrop-filter: blur(20px);
}
.stock-search { padding: 16px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.2); }
.stock-search input {
  width: 100%; padding: 12px 16px; border-radius: 12px;
  border: 1px solid var(--border); background: rgba(255, 255, 255, 0.03);
  color: var(--text-primary); font-size: 14px; outline: none;
  transition: all 0.3s; font-family: 'Inter', sans-serif;
}
.stock-search input:focus { border-color: var(--accent-blue); background: rgba(59, 130, 246, 0.05); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15); }
.stock-filter-bar { display: flex; gap: 6px; padding: 10px 16px; border-bottom: 1px solid var(--border); overflow-x: auto; }
.filter-btn {
  padding: 6px 14px; border: 1px solid transparent; border-radius: 20px;
  background: rgba(255, 255, 255, 0.03); color: var(--text-secondary); font-size: 12px;
  cursor: pointer; font-weight: 600; transition: all 0.2s; white-space: nowrap;
}
.filter-btn:hover { background: rgba(255, 255, 255, 0.08); color: var(--text-primary); }
.filter-btn.active { background: var(--accent-blue); color: #fff; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
.stock-items { flex: 1; overflow-y: auto; padding: 0; scrollbar-width: thin; }
.stock-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  cursor: pointer; transition: all 0.2s; position: relative;
}
.stock-item:hover { background: rgba(255, 255, 255, 0.03); padding-left: 20px; }
.stock-item.active { background: rgba(59, 130, 246, 0.08); border-left: 4px solid var(--accent-blue); }
.stock-item .info { display: flex; flex-direction: column; }
.stock-item .symbol { font-size: 15px; font-weight: 700; color: var(--text-primary); }
.stock-item .name { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.stock-item .signal-badge {
  padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 800;
  text-transform: uppercase; letter-spacing: 1px;
}
.signal-BUY { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
.signal-SELL { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }
.signal-HOLD { background: rgba(255, 255, 255, 0.08); color: var(--text-secondary); border: 1px solid rgba(255, 255, 255, 0.1); }

.chart-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-card); backdrop-filter: blur(20px);
}
.chart-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 24px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.2);
}
.chart-title { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; }
.chart-subtitle { font-size: 13px; color: var(--text-muted); margin-left: 10px; font-weight: 500; }
.interval-btns { display: flex; gap: 6px; background: rgba(0,0,0,0.4); padding: 4px; border-radius: 10px; border: 1px solid var(--border); }
.interval-btn {
  padding: 6px 12px; border: none; border-radius: 6px;
  background: transparent; color: var(--text-secondary); font-size: 12px; font-weight: 700;
  cursor: pointer; transition: all 0.2s;
}
.interval-btn:hover { color: var(--text-primary); }
.interval-btn.active { background: rgba(255,255,255,0.1); color: var(--text-primary); box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.chart-container { flex: 1; min-height: 400px; position: relative; }

.ai-signal-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; padding: 20px 24px; border-top: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%);
}
.signal-metric {
  padding: 16px; border-radius: var(--radius-sm);
  background: rgba(0, 0, 0, 0.4); border: 1px solid var(--border);
  position: relative; overflow: hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.signal-metric::after {
  content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
  pointer-events: none;
}
.signal-metric .label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); }
.signal-metric .value { font-size: 22px; font-weight: 800; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
.signal-metric .value.buy { color: var(--accent-green); text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
.signal-metric .value.sell { color: var(--accent-red); text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
.signal-metric .value.hold { color: var(--accent-amber); text-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }

.signals-grid, .stat-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}
.signal-card, .stat-card {
  padding: 24px; border-radius: var(--radius);
  background: var(--bg-card); border: 1px solid var(--border);
  backdrop-filter: blur(20px); box-shadow: var(--shadow-card), var(--shadow-inner);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative; overflow: hidden;
}
.signal-card:hover, .stat-card:hover { 
  transform: translateY(-4px); box-shadow: var(--shadow-glow), var(--shadow-card); 
  border-color: rgba(59, 130, 246, 0.4);
}
.signal-card::before {
  content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%;
  background: var(--gradient-blue); opacity: 0; transition: opacity 0.3s;
}
.signal-card:hover::before { opacity: 1; }
.signal-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.signal-card-ticker { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; }
.signal-card-time { font-size: 12px; color: var(--text-muted); font-weight: 500; }
.signal-card-body { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.signal-card-body .metric { display: flex; flex-direction: column; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; border: 1px solid var(--border); }
.signal-card-body .metric .label { font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }
.signal-card-body .metric .val { font-size: 16px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 4px; color: var(--text-primary); }

.stat-card .label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 10px; }
.stat-card .value { font-size: 32px; font-weight: 800; font-family: 'JetBrains Mono', monospace; background: var(--gradient-blue); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.tab-page h2 { font-size: 24px; font-weight: 800; margin-bottom: 24px; letter-spacing: -0.5px; display: inline-flex; align-items: center; gap: 10px; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

.spinner {
  width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.1);
  border-top-color: var(--accent-blue); border-radius: 50%;
  animation: spin 0.8s cubic-bezier(0.4, 0, 0.2, 1) infinite; margin: 60px auto;
}
@keyframes spin { to { transform: rotate(360deg); } }
.empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); font-size: 15px; font-weight: 500; }
</style>'''
new_html = re.sub(r'<style>.*?</style>', new_style, html, flags=re.DOTALL)
p.write_text(new_html, 'utf8')

import os
for f in ['rp.py', 'update_css.py', 'new_style.css', 'cleanup.py']:
    f_path = Path(f)
    if f_path.exists():
        try:
            f_path.unlink()
        except:
            pass
