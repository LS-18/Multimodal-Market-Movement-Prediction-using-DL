/* ═══════════════════════════════════════════════════════════
   MarketMind — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const API = 'http://localhost:5000/api';

// ── State ────────────────────────────────────────────────────
let currentPrices = new Array(10).fill('');
let predictionLog = [];

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildPriceGrid();
  startClock();
  checkModelStatus();
  fetchLivePrices();
  setupHeadlineCounter();
  drawIdleCanvas();
  loadComparison();
});

// ══════════════════════════════════════════════════════════════
// CLOCK
// ══════════════════════════════════════════════════════════════
function startClock() {
  const el = document.getElementById('clock');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', {
      hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };
  tick();
  setInterval(tick, 1000);
}

// ══════════════════════════════════════════════════════════════
// PRICE GRID
// ══════════════════════════════════════════════════════════════
function buildPriceGrid() {
  const grid = document.getElementById('priceGrid');
  grid.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'price-input-wrap';
    wrap.innerHTML = `
      <label class="price-label">DAY ${i + 1}</label>
      <input class="price-input" id="p${i}" type="number" step="0.01"
             placeholder="0.00" oninput="onPriceChange(${i}, this.value)" />
    `;
    grid.appendChild(wrap);
  }
}

function onPriceChange(idx, val) {
  currentPrices[idx] = val;
  const inp = document.getElementById(`p${idx}`);
  inp.classList.toggle('filled', val !== '');
  drawPriceCanvas();
}

function clearPrices() {
  currentPrices = new Array(10).fill('');
  for (let i = 0; i < 10; i++) {
    const inp = document.getElementById(`p${i}`);
    inp.value = '';
    inp.classList.remove('filled');
  }
  drawIdleCanvas();
}

function fillDemo() {
  const demo = [4200, 4215, 4198, 4230, 4245, 4260, 4238, 4275, 4290, 4310];
  demo.forEach((v, i) => {
    currentPrices[i] = v;
    const inp = document.getElementById(`p${i}`);
    inp.value = v;
    inp.classList.add('filled');
  });
  drawPriceCanvas();
}

function getPricesArray() {
  return currentPrices.map(v => parseFloat(v)).filter(v => !isNaN(v));
}

// ══════════════════════════════════════════════════════════════
// HEADLINE COUNTER
// ══════════════════════════════════════════════════════════════
function setupHeadlineCounter() {
  const ta = document.getElementById('headlineInput');
  const counter = document.getElementById('charCount');
  ta.addEventListener('input', () => {
    counter.textContent = ta.value.length;
    counter.style.color = ta.value.length > 450 ? 'var(--amber)' : '';
  });
}

// ══════════════════════════════════════════════════════════════
// PRICE CANVAS (mini visualiser)
// ══════════════════════════════════════════════════════════════
function drawIdleCanvas() {
  const canvas = document.getElementById('priceCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1e2d3d';
  ctx.lineWidth = 1;
  for (let y = 20; y < H; y += 30) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let x = 60; x < W; x += 60) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  ctx.fillStyle = '#3d5166';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('ENTER PRICES TO VISUALISE', W / 2, H / 2 + 4);
}

function drawPriceCanvas() {
  const prices = getPricesArray();
  if (prices.length < 2) { drawIdleCanvas(); return; }

  const canvas = document.getElementById('priceCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 12, right: 16, bottom: 24, left: 52 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const range = maxP - minP || 1;

  const toX = i => PAD.left + (i / (prices.length - 1)) * IW;
  const toY = v => PAD.top + IH - ((v - minP) / range) * IH;

  // Grid
  ctx.strokeStyle = '#1e2d3d';
  ctx.lineWidth = 1;
  const steps = 4;
  for (let s = 0; s <= steps; s++) {
    const y = PAD.top + (s / steps) * IH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    const val = maxP - (s / steps) * range;
    ctx.fillStyle = '#3d5166';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(0), PAD.left - 4, y + 3);
  }

  // Day labels
  ctx.fillStyle = '#3d5166';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  prices.forEach((_, i) => {
    ctx.fillText(`D${i + 1}`, toX(i), H - 6);
  });

  // Fill area
  const isUp = prices[prices.length - 1] >= prices[0];
  const fillColor = isUp ? 'rgba(0,255,136,0.08)' : 'rgba(255,68,68,0.08)';
  const lineColor = isUp ? '#00ff88' : '#ff4444';

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(prices[0]));
  prices.forEach((p, i) => { if (i > 0) ctx.lineTo(toX(i), toY(p)); });
  ctx.lineTo(toX(prices.length - 1), PAD.top + IH);
  ctx.lineTo(toX(0), PAD.top + IH);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(prices[0]));
  prices.forEach((p, i) => { if (i > 0) ctx.lineTo(toX(i), toY(p)); });
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Dots
  prices.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(toX(i), toY(p), 3, 0, Math.PI * 2);
    ctx.fillStyle = lineColor;
    ctx.fill();
  });

  // Latest price label
  const lastX = toX(prices.length - 1);
  const lastY = toY(prices[prices.length - 1]);
  ctx.fillStyle = lineColor;
  ctx.font = 'bold 10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`$${prices[prices.length - 1].toFixed(2)}`, lastX + 6, lastY + 4);
}

// ══════════════════════════════════════════════════════════════
// FETCH LIVE PRICES
// ══════════════════════════════════════════════════════════════
async function fetchLivePrices() {
  const btn = document.getElementById('fetchBtn');
  btn.textContent = '⟳ ...';
  btn.disabled = true;

  try {
    const res  = await fetch(`${API}/live-prices`);
    const data = await res.json();

    if (data.status === 'success') {
      data.prices.forEach((p, i) => {
        currentPrices[i] = p;
        const inp = document.getElementById(`p${i}`);
        if (inp) { inp.value = p; inp.classList.add('filled'); }
      });
      drawPriceCanvas();

      // Update topbar
      const chgEl = document.getElementById('spxChg');
      const valEl = document.getElementById('spxVal');
      valEl.textContent = `$${data.latest_close.toLocaleString()}`;
      const up = data.change_pct >= 0;
      chgEl.textContent = `${up ? '+' : ''}${data.change_pct}%`;
      chgEl.style.color = up ? 'var(--green)' : 'var(--red)';
    }
  } catch (e) {
    console.warn('Live price fetch failed:', e.message);
  } finally {
    btn.innerHTML = '<span class="btn-icon">⟳</span> LIVE';
    btn.disabled = false;
  }
}

// ══════════════════════════════════════════════════════════════
// MODEL STATUS
// ══════════════════════════════════════════════════════════════
async function checkModelStatus() {
  try {
    const res  = await fetch(`${API}/model-status`);
    const data = await res.json();

    document.getElementById('msStatus').textContent  = data.model_loaded ? 'READY' : 'DEMO';
    document.getElementById('msStatus').className    = `tag ${data.model_loaded ? 'tag-ok' : 'tag-warn'}`;
    document.getElementById('msDevice').textContent  = data.device.toUpperCase();
    document.getElementById('msWeights').textContent = data.model_loaded ? `${data.size_mb} MB` : 'NOT FOUND';
    document.getElementById('msMode').textContent    = data.mode.toUpperCase();

    document.getElementById('modelMode').textContent = data.model_loaded ? 'READY' : 'DEMO';
    document.getElementById('deviceVal').textContent = data.device.toUpperCase();
    document.getElementById('footerModel').textContent =
      `Model: ${data.model_loaded ? `Loaded (${data.size_mb}MB)` : 'Demo mode — run minor.py to train'}`;

    // Status dot
    document.getElementById('statusDot').className   = 'status-dot online';
    document.getElementById('statusLabel').textContent = 'ONLINE';
  } catch (e) {
    document.getElementById('statusDot').className    = 'status-dot offline';
    document.getElementById('statusLabel').textContent = 'OFFLINE';
    document.getElementById('footerModel').textContent = 'API: offline — run py app.py';
  }
}

// ══════════════════════════════════════════════════════════════
// TRAINING HISTORY
// ══════════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const res  = await fetch(`${API}/history`);
    const data = await res.json();
    if (data.status !== 'ok') {
      document.getElementById('historyStats').textContent = 'No history found. Train the model first.';
      return;
    }
    drawHistoryChart(data);
    const bestVal = Math.min(...data.val_loss).toFixed(4);
    const bestAcc = (Math.max(...data.val_acc) * 100).toFixed(1);
    document.getElementById('historyStats').innerHTML =
      `Epochs: ${data.epochs} &nbsp;|&nbsp; Best Val Loss: ${bestVal} &nbsp;|&nbsp; Best Val Acc: ${bestAcc}%`;
  } catch (e) {
    document.getElementById('historyStats').textContent = 'Could not load history.';
  }
}

function drawHistoryChart(data) {
  const canvas = document.getElementById('historyCanvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 10, right: 10, bottom: 20, left: 36 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  const epochs = data.epochs;
  const toX = i => PAD.left + (i / (epochs - 1)) * IW;

  function drawLine(vals, color, minV, maxV) {
    const range = maxV - minV || 1;
    const toY = v => PAD.top + IH - ((v - minV) / range) * IH;
    ctx.beginPath();
    vals.forEach((v, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  // Grid
  ctx.strokeStyle = '#1e2d3d';
  ctx.lineWidth = 1;
  for (let s = 0; s <= 3; s++) {
    const y = PAD.top + (s / 3) * IH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
  }

  const allLoss = [...data.train_loss, ...data.val_loss];
  const minL = Math.min(...allLoss), maxL = Math.max(...allLoss);
  drawLine(data.train_loss, '#00ff88', minL, maxL);
  drawLine(data.val_loss,   '#ff9900', minL, maxL);
  drawLine(data.val_acc,    '#00aaff', 0, 1);

  // Epoch labels
  ctx.fillStyle = '#3d5166';
  ctx.font = '8px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  [0, Math.floor(epochs / 2), epochs - 1].forEach(i => {
    ctx.fillText(i + 1, toX(i), H - 4);
  });
}

// ══════════════════════════════════════════════════════════════
// MODEL COMPARISON
// ══════════════════════════════════════════════════════════════
async function loadComparison() {
  try {
    const res  = await fetch(`${API}/comparison`);
    const data = await res.json();
    if (data.status !== 'ok') return;

    const list = document.getElementById('comparisonList');
    if (!list) return;
    list.innerHTML = '';

    // Sort by accuracy descending
    const sorted = data.results.sort((a, b) => b.Accuracy - a.Accuracy);
    const maxAcc = sorted[0].Accuracy;

    sorted.forEach(row => {
      const isOurs = row.Model.includes('Multimodal') || row.Model.includes('Ours');
      const bar = Math.round((row.Accuracy / maxAcc) * 100);
      const el = document.createElement('div');
      el.className = `comp-row ${isOurs ? 'comp-ours' : ''}`;
      el.innerHTML = `
        <div class="comp-name">${row.Model}</div>
        <div class="comp-bar-wrap">
          <div class="comp-bar" style="width:${bar}%;background:${isOurs ? 'var(--green)' : 'var(--border2)'}"></div>
        </div>
        <div class="comp-acc ${isOurs ? 'bullish' : ''}">${row.Accuracy.toFixed(1)}%</div>
      `;
      list.appendChild(el);
    });
  } catch (e) {
    console.warn('Comparison load failed:', e.message);
  }
}
async function runPrediction() {
  const prices   = getPricesArray();
  const headline = document.getElementById('headlineInput').value.trim();

  // Validate
  if (prices.length !== 10) {
    flashError(`Need exactly 10 prices. Got ${prices.length}.`); return;
  }
  if (!headline) {
    flashError('Please enter a news headline.'); return;
  }

  // Show loading
  showLoading(true);
  const btn = document.getElementById('predictBtn');
  btn.classList.add('loading');

  const steps = [
    'Encoding price chart via ResNet18 (224×224)...',
    'Tokenising headline via FinBERT (128 tokens)...',
    'Running multimodal fusion (1280-D joint embedding)...',
    'Computing probability via MLP + BatchNorm...',
  ];
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    document.getElementById('loaderSub').textContent = steps[stepIdx % steps.length];
    stepIdx++;
  }, 900);

  try {
    const res  = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prices, headline }),
    });
    const data = await res.json();

    if (data.error) { flashError(data.error); return; }

    clearInterval(stepInterval);
    showLoading(false);
    displayResult(data, headline);
    addToLog(data, headline);

  } catch (e) {
    clearInterval(stepInterval);
    flashError(`API error: ${e.message}`);
  } finally {
    btn.classList.remove('loading');
    showLoading(false);
  }
}

function showLoading(show) {
  document.getElementById('resultIdle').classList.add('hidden');
  document.getElementById('resultActive').classList.add('hidden');
  document.getElementById('resultLoading').classList.toggle('hidden', !show);
}

function displayResult(data, headline) {
  const isBull = data.label === 'Bullish';
  const cls    = isBull ? 'bullish' : 'bearish';

  // Card border glow
  const card = document.getElementById('resultCard');
  card.className = `result-card ${cls}`;

  // Show active panel
  document.getElementById('resultActive').classList.remove('hidden');

  // ── Hero prediction label ──
  const predLabel = document.getElementById('predLabel');
  predLabel.textContent = data.label.toUpperCase();
  predLabel.className   = `pred-label ${cls}`;

  const predHeroBg = document.getElementById('predHeroBg');
  predHeroBg.className  = `pred-hero-bg ${cls}`;

  document.getElementById('predUpPct').textContent   = `${data.up_pct.toFixed(1)}%`;
  document.getElementById('predDownPct').textContent = `${data.down_pct.toFixed(1)}%`;
  const confEl = document.getElementById('predConf');
  confEl.textContent = data.confidence.toUpperCase();
  confEl.style.color =
    data.confidence === 'High'   ? 'var(--green)' :
    data.confidence === 'Medium' ? 'var(--amber)'  : 'var(--text-dim)';

  // Time
  document.getElementById('verdictTime').textContent =
    new Date().toLocaleTimeString('en-US', { hour12: false });

  // Gauge
  const pct = data.up_pct;
  document.getElementById('gaugeFill').style.width  = `${pct}%`;
  document.getElementById('gaugeNeedle').style.left = `${pct}%`;
  document.getElementById('gaugeProb').textContent  = `${pct.toFixed(1)}% UP`;
  document.getElementById('downPct').textContent    = `↓ ${data.down_pct}%`;
  document.getElementById('upPct').textContent      = `↑ ${data.up_pct}%`;

  // Stats
  document.getElementById('statProb').textContent   = `${(data.probability * 100).toFixed(1)}%`;
  document.getElementById('statProb').style.color   = isBull ? 'var(--green)' : 'var(--red)';
  document.getElementById('statSignal').textContent = data.label.toUpperCase();
  document.getElementById('statSignal').style.color = isBull ? 'var(--green)' : 'var(--red)';
  document.getElementById('statConf').textContent   = data.confidence.toUpperCase();
  document.getElementById('statConf').style.color   =
    data.confidence === 'High'   ? 'var(--green)' :
    data.confidence === 'Medium' ? 'var(--amber)'  : 'var(--text-dim)';

  // Headline echo
  document.getElementById('echoText').textContent =
    headline.length > 120 ? headline.slice(0, 120) + '…' : headline;

  // ── Brand header signal (below logo, always visible) ──
  const brandSignal = document.getElementById('brandSignal');
  const brandWord   = document.getElementById('brandSignalWord');
  const brandProb   = document.getElementById('brandSignalProb');

  brandWord.textContent = data.label.toUpperCase();
  brandWord.className   = `brand-signal-word ${cls}`;
  brandProb.textContent = `${data.up_pct.toFixed(1)}% UP  |  ${data.down_pct.toFixed(1)}% DOWN  |  ${data.confidence.toUpperCase()} CONFIDENCE`;

  // Re-trigger pop animation
  brandWord.style.animation = 'none';
  brandWord.offsetHeight;
  brandWord.style.animation = '';

  brandSignal.classList.remove('hidden');
}

// ══════════════════════════════════════════════════════════════
// SESSION LOG
// ══════════════════════════════════════════════════════════════
function addToLog(data, headline) {
  const isBull = data.label === 'Bullish';
  const entry  = {
    label:    data.label,
    prob:     data.up_pct,
    conf:     data.confidence,
    headline: headline,
    time:     new Date().toLocaleTimeString('en-US', { hour12: false }),
  };
  predictionLog.unshift(entry);

  const list = document.getElementById('logList');
  const empty = list.querySelector('.log-empty');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = `log-entry ${isBull ? 'bullish' : 'bearish'}`;
  el.innerHTML = `
    <div class="log-top">
      <span class="log-verdict ${isBull ? 'bullish' : 'bearish'}">${data.label.toUpperCase()}</span>
      <span class="log-time">${entry.time}</span>
    </div>
    <div class="log-prob">UP ${data.up_pct}% &nbsp;|&nbsp; ${data.confidence} confidence</div>
    <div class="log-headline">${headline}</div>
  `;
  list.insertBefore(el, list.firstChild);

  // Keep max 20 entries
  while (list.children.length > 20) list.removeChild(list.lastChild);
}

function clearLog() {
  predictionLog = [];
  document.getElementById('logList').innerHTML =
    '<div class="log-empty">No predictions yet</div>';
}

// ══════════════════════════════════════════════════════════════
// ERROR FLASH
// ══════════════════════════════════════════════════════════════
function flashError(msg) {
  showLoading(false);
  document.getElementById('resultIdle').classList.remove('hidden');
  document.getElementById('resultActive').classList.add('hidden');
  document.getElementById('brandSignal').classList.add('hidden');

  const idle = document.getElementById('resultIdle');
  const orig = idle.innerHTML;
  idle.innerHTML = `
    <div class="idle-center-text">
      <div class="idle-center-label" style="color:var(--red);animation:none">ERROR</div>
      <div class="idle-center-sub" style="color:var(--text-dim)">${msg}</div>
    </div>
  `;
  setTimeout(() => { idle.innerHTML = orig; }, 4000);
}
