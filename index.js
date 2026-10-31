const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const SOURCE_API = 'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';
const POLL_INTERVAL = 15000;

// ============ 1. FILE ============
function initDataFile() {
  fs.writeFileSync(DATA_FILE, '[]', 'utf8');
  console.log('[INIT] Đã xoá & tạo mới lichsulc79md5.json');
}
function readHistory() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8') || '[]'); }
  catch { return []; }
}
function writeHistory(arr) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(arr, null, 2), 'utf8');
}

// ============ 2. SYNC ============
async function fetchSource() {
  const res = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(10000)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

// ⚠️ SỬA THEO RESPONSE THẬT CỦA TELE68 NẾU CẦN
function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;
  const dices = raw.dices ?? raw.dice ?? raw.result ?? [];
  const sum = Array.isArray(dices)
    ? dices.reduce((a, b) => a + Number(b), 0)
    : Number(raw.sum ?? 0);
  const result = sum >= 11 ? 1 : 0;
  return { id: Number(id), dices: dices.map(Number), sum, result };
}

async function syncData() {
  try {
    const raw = await fetchSource();
    const list = Array.isArray(raw) ? raw
      : raw.data ?? raw.sessions ?? raw.list ?? [];
    if (!list.length) return;

    const history = readHistory();
    const knownIds = new Set(history.map(h => h.id));
    let added = 0;

    for (const item of list) {
      const s = normalizeSession(item);
      if (!Number.isFinite(s.id) || knownIds.has(s.id)) continue;
      history.push(s);
      knownIds.add(s.id);
      added++;
    }
    if (added > 0) {
      history.sort((a, b) => a.id - b.id);
      writeHistory(history.slice(-500));
      console.log(`[SYNC] +${added} phiên, tổng ${Math.min(500, history.length)}`);
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  }
}

// ============ 3. SÁU THUẬT TOÁN ============
function algoMarkov(seq) {
  if (seq.length < 5) return { pick: 1, confidence: 0.5 };
  const c = { '00': [0,0], '01': [0,0], '10': [0,0], '11': [0,0] };
  for (let i = 2; i < seq.length; i++) {
    c[`${seq[i-2]}${seq[i-1]}`][seq[i]]++;
  }
  const [c0, c1] = c[`${seq[seq.length-2]}${seq[seq.length-1]}`];
  const t = c0 + c1;
  if (!t) return { pick: 1, confidence: 0.5 };
  return { pick: c1 >= c0 ? 1 : 0, confidence: Math.max(c0, c1) / t };
}

function algoPattern(seq) {
  const n = seq.length;
  if (n < 6) return { pick: 1, confidence: 0.5 };
  for (let len = 5; len >= 2; len--) {
    const tail = seq.slice(-len).join('');
    for (let i = 0; i <= n - len - 1; i++) {
      if (seq.slice(i, i + len).join('') === tail) {
        const next = seq[i + len];
        if (next !== undefined) return { pick: next, confidence: 0.65 };
      }
    }
  }
  return { pick: 1, confidence: 0.5 };
}

function algoMomentum(seq) {
  if (seq.length < 5) return { pick: 1, confidence: 0.5 };
  const s = seq.slice(-5), l = seq.slice(-20);
  const aS = s.reduce((a,b)=>a+b,0) / s.length;
  const aL = l.reduce((a,b)=>a+b,0) / l.length;
  const m = aS - aL;
  if (Math.abs(m) < 0.05) return { pick: 1, confidence: 0.5 };
  return { pick: m > 0 ? 1 : 0, confidence: Math.min(0.75, 0.5 + Math.abs(m)) };
}

function algoStreak(seq) {
  const n = seq.length;
  if (n < 3) return { pick: 1, confidence: 0.5 };
  const last = seq[n-1];
  let s = 1;
  for (let i = n-2; i >= 0 && seq[i] === last; i--) s++;
  if (s >= 4) return { pick: 1 - last, confidence: Math.min(0.7, 0.5 + s*0.03) };
  return { pick: last, confidence: 0.52 };
}

function algoFrequency(seq) {
  if (seq.length < 10) return { pick: 1, confidence: 0.5 };
  const w = seq.slice(-30);
  const c1 = w.filter(x => x === 1).length;
  const c0 = w.length - c1;
  if (Math.abs(c1 - c0) < 3) return { pick: 1, confidence: 0.5 };
  return {
    pick: c1 > c0 ? 0 : 1,
    confidence: Math.min(0.7, 0.5 + Math.abs(c1 - c0) / w.length)
  };
}

const lrW = { w: [0,0,0,0,0], b: 0, lr: 0.05 };
function feats(seq) {
  const n = seq.length;
  const last = seq[n-1] ?? 0;
  let streak = 0;
  for (let i = n-1; i >= 0 && seq[i] === last; i--) streak++;
  const freq = seq.slice(-20).reduce((a,b)=>a+b,0) / Math.max(1, Math.min(20, n));
  return [last, seq[n-2]??0, seq[n-3]??0, streak/5, freq];
}
function lrPredict(f) {
  const z = f.reduce((s, v, i) => s + v*lrW.w[i], lrW.b);
  const p = 1 / (1 + Math.exp(-z));
  return { pick: p >= 0.5 ? 1 : 0, confidence: Math.max(p, 1-p) };
}
function lrTrain(seq) {
  if (seq.length < 10) return;
  const x = feats(seq.slice(0, -1));
  const y = seq[seq.length-1];
  const z = x.reduce((s, v, i) => s + v*lrW.w[i], lrW.b);
  const p = 1 / (1 + Math.exp(-z));
  const e = p - y;
  for (let i = 0; i < lrW.w.length; i++) lrW.w[i] -= lrW.lr * e * x[i];
  lrW.b -= lrW.lr * e;
}
function algoLogistic(seq) {
  if (seq.length < 6) return { pick: 1, confidence: 0.5 };
  return lrPredict(feats(seq));
}

// ============ 4. ENSEMBLE ============
const ALGOS = [
  { name: 'markov', fn: algoMarkov },
  { name: 'pattern', fn: algoPattern },
  { name: 'momentum', fn: algoMomentum },
  { name: 'streak', fn: algoStreak },
  { name: 'frequency', fn: algoFrequency },
  { name: 'logistic', fn: algoLogistic }
];
const algoScore = Object.fromEntries(
  ALGOS.map(a => [a.name, { correct: 0, total: 0 }])
);

function updateScores(history) {
  for (const algo of ALGOS) {
    const s = algoScore[algo.name];
    s.correct = 0; s.total = 0;
    const start = Math.max(6, history.length - 60);
    for (let i = start; i < history.length; i++) {
      const past = history.slice(0, i).map(h => h.result);
      const ok = algo.fn(past).pick === history[i].result ? 1 : 0;
      s.correct += ok; s.total++;
    }
  }
}

function ensemblePredict(seq) {
  let sumTai = 0, sumXiu = 0, totalW = 0;
  for (const algo of ALGOS) {
    const p = algo.fn(seq);
    const s = algoScore[algo.name];
    const acc = s.total >= 10 ? s.correct / s.total : 0.5;
    const w = acc * p.confidence;
    totalW += w;
    if (p.pick === 1) sumTai += w; else sumXiu += w;
  }
  if (!totalW) return { pick: 1, confidence: 51 };
  const pTai = sumTai / totalW;
  const pick = pTai >= 0.5 ? 1 : 0;
  const conf = Math.round((pick === 1 ? pTai : 1 - pTai) * 100);
  return { pick, confidence: Math.min(99, Math.max(51, conf)) };
}

// ============ 5. TIME VN ============
function formatTimeVN() {
  const now = new Date(Date.now() + 7 * 3600 * 1000);
  return now.toISOString().replace('Z', '+07:00');
}

// ============ 6. RESPONSE ============
function buildResponse() {
  const history = readHistory();
  if (history.length === 0) {
    return { "id": "noname09", "error": "Chưa có dữ liệu, chờ sync", "uptime": formatTimeVN() };
  }

  const current = history[history.length - 1];
  const seq = history.map(h => h.result);

  updateScores(history);
  const ens = ensemblePredict(seq);
  lrTrain(seq);

  return {
    "id": "noname09",
    "phienhientai": current.id,
    "ketqua": current.result === 1 ? "tai" : "xiu",
    "xucxac": current.dices,
    "tong": current.sum,
    "dudoanphien": current.id + 1,
    "dudoan": ens.pick === 1 ? "tai" : "xiu",
    "dotincay": `${ens.confidence}%`,
    "uptime": formatTimeVN()
  };
}

// ============ 7. ROUTES ============
app.get('/', (req, res) => res.json(buildResponse()));
app.get('/predict', (req, res) => res.json(buildResponse()));
app.get('/ping', (req, res) => res.send('pong'));
app.get('/history', (req, res) => {
  const h = readHistory();
  res.json({ count: h.length, last20: h.slice(-20) });
});

// ============ 8. BOOT ============
initDataFile();
syncData();
setInterval(syncData, POLL_INTERVAL);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`));