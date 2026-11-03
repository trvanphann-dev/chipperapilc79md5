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
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8') || '[]');
  } catch {
    return [];
  }
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

function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;

  const rawDices = raw.dices ?? raw.dice ??
    (Array.isArray(raw.result) ? raw.result : []);

  const dices = Array.isArray(rawDices)
    ? rawDices.map(Number)
    : [];

  const sum = dices.length
    ? dices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? 0);

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    raw.result ??
    ''
  ).trim().toUpperCase();

  let result;

  if (
    sourceResult === 'TAI' ||
    sourceResult === 'TÀI' ||
    sourceResult === 'BIG'
  ) {
    result = 1;
  } else if (
    sourceResult === 'XIU' ||
    sourceResult === 'XỈU' ||
    sourceResult === 'SMALL'
  ) {
    result = 0;
  } else {
    result = sum >= 11 ? 1 : 0;
  }

  return {
    id: Number(id),
    dices,
    sum,
    result
  };
}

async function syncData() {
  try {
    const raw = await fetchSource();

    const list = Array.isArray(raw)
      ? raw
      : raw.data ?? raw.sessions ?? raw.list ?? [];

    if (!list.length) return;

    const history = readHistory();
    const knownIds = new Set(history.map(h => h.id));

    let added = 0;

    for (const item of list) {
      const s = normalizeSession(item);

      if (
        !Number.isFinite(s.id) ||
        !Number.isFinite(s.sum) ||
        ![0, 1].includes(s.result) ||
        knownIds.has(s.id)
      ) continue;

      history.push(s);
      knownIds.add(s.id);
      added++;
    }

    if (added > 0) {
      history.sort((a, b) => a.id - b.id);

      writeHistory(history.slice(-500));

      console.log(
        `[SYNC] +${added} phiên, tổng ${Math.min(500, history.length)}`
      );
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  }
}

// ============ 3. STATISTICAL FUNCTIONS ============

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getRate(seq) {
  if (!seq.length) return 0.5;

  const valid = seq.filter(v => v === 0 || v === 1);

  if (!valid.length) return 0.5;

  const tai = valid.reduce((sum, value) => sum + value, 0);

  return tai / valid.length;
}

function makePrediction(pTai, strength = 0) {
  const p = clamp(pTai, 0, 1);

  return {
    pick: p > 0.5 ? 1 : 0,
    pTai: p,
    strength: clamp(strength, 0, 1)
  };
}

// ============ 4. ALGORITHM 1: BAYESIAN BASE RATE ============

function algoBayesian(seq) {
  const n = seq.length;

  if (n === 0) return makePrediction(0.5);

  const tai = seq.reduce((sum, value) => sum + value, 0);

  // Prior đối xứng, không thiên vị Tài/Xỉu.
  const alpha = 2;
  const beta = 2;

  const pTai = (tai + alpha) / (n + alpha + beta);

  const strength = Math.abs(pTai - 0.5) * 2;

  return makePrediction(pTai, strength);
}

// ============ 5. ALGORITHM 2: MARKOV TRANSITION ============

function algoMarkov(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const current = seq[n - 1];

  const globalRate = getRate(seq);

  let tai = 0;
  let total = 0;

  // Đếm xác suất kết quả tiếp theo sau trạng thái hiện tại.
  for (let i = 1; i < n; i++) {
    if (seq[i - 1] !== current) continue;

    total++;

    if (seq[i] === 1) tai++;
  }

  if (total === 0) {
    return makePrediction(globalRate);
  }

  // Co xác suất về tỷ lệ tổng thể để tránh mẫu nhỏ gây lệch.
  const priorWeight = 8;

  const pTai = (
    tai + globalRate * priorWeight
  ) / (
    total + priorWeight
  );

  const strength = (
    Math.abs(pTai - globalRate) * 2
  ) * (
    total / (total + priorWeight)
  );

  return makePrediction(pTai, strength);
}

// ============ 6. ALGORITHM 3: RECENT WINDOW ============

function algoRecent(seq) {
  const n = seq.length;

  if (n < 5) return makePrediction(0.5);

  const windowSize = Math.min(30, n);

  const recent = seq.slice(-windowSize);

  const globalRate = getRate(seq);

  const tai = recent.reduce((sum, value) => sum + value, 0);

  // Làm trơn dữ liệu ngắn hạn bằng tỷ lệ tổng thể.
  const priorWeight = 10;

  const pTai = (
    tai + globalRate * priorWeight
  ) / (
    recent.length + priorWeight
  );

  const strength = (
    Math.abs(pTai - globalRate) * 2
  ) * (
    recent.length / (recent.length + priorWeight)
  );

  return makePrediction(pTai, strength);
}

// ============ 7. ALGORITHM 4: EWMA ============

function algoEWMA(seq) {
  const n = seq.length;

  if (n < 5) return makePrediction(0.5);

  const globalRate = getRate(seq);

  const alpha = 0.15;

  let average = globalRate;

  for (let i = 0; i < n; i++) {
    average =
      alpha * seq[i] +
      (1 - alpha) * average;
  }

  // Không để EWMA ngắn hạn lấn át tỷ lệ tổng thể.
  const blend = 0.30;

  const pTai = (
    globalRate * (1 - blend) +
    average * blend
  );

  const strength = Math.abs(
    pTai - globalRate
  ) * 2;

  return makePrediction(pTai, strength);
}

// ============ 8. ALGORITHM 5: MOMENTUM ============

function algoMomentum(seq) {
  const n = seq.length;

  if (n < 20) return makePrediction(0.5);

  const globalRate = getRate(seq);

  const recentSize = Math.min(10, n);
  const previousSize = Math.min(30, n - recentSize);

  const recent = seq.slice(-recentSize);

  const previous = seq.slice(
    -(recentSize + previousSize),
    -recentSize
  );

  if (!previous.length) {
    return makePrediction(globalRate);
  }

  const recentRate = getRate(recent);
  const previousRate = getRate(previous);

  // Chỉ sử dụng một phần rất nhỏ chênh lệch ngắn hạn.
  // Không giả định xu hướng sẽ tiếp tục vô hạn.
  const adjustment = (
    recentRate - previousRate
  ) * 0.10;

  const pTai = clamp(
    globalRate + adjustment,
    0.05,
    0.95
  );

  const strength = Math.abs(
    adjustment
  ) * 2;

  return makePrediction(pTai, strength);
}

// ============ 9. FIVE ALGORITHMS ============

const ALGOS = [
  { name: 'bayesian', fn: algoBayesian },
  { name: 'markov_transition', fn: algoMarkov },
  { name: 'recent_window', fn: algoRecent },
  { name: 'ewma', fn: algoEWMA },
  { name: 'momentum', fn: algoMomentum }
];

// ============ 10. ENSEMBLE ============

function ensemblePredict(seq) {
  if (seq.length < 10) {
    return {
      pick: 0,
      confidence: 50,
      pTai: 0.5
    };
  }

  const predictions = ALGOS.map(algo => {
    return algo.fn(seq);
  });

  // Trọng số giảm ảnh hưởng của các tín hiệu ngắn hạn.
  const weights = [
    0.30,
    0.25,
    0.20,
    0.15,
    0.10
  ];

  let weightedTai = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    weightedTai += predictions[i].pTai * weights[i];
    totalWeight += weights[i];
  }

  let pTai = weightedTai / totalWeight;

  // Giới hạn độ lệch để tránh dự đoán quá tự tin.
  pTai = clamp(pTai, 0.20, 0.80);

  // Nếu xác suất gần 50/50 thì xem là tín hiệu yếu.
  const distance = Math.abs(pTai - 0.5);

  // Điểm tin cậy thận trọng, không phải xác suất thắng.
  const confidence = 50 + Math.min(
    5,
    distance * 20
  );

  const pick = pTai > 0.5 ? 1 : 0;

  return {
    pick,

    confidence: Math.round(
      confidence * 100
    ) / 100,

    pTai: Math.round(
      pTai * 10000
    ) / 10000,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,

      pick: predictions[index].pick,

      probabilityTai: Math.round(
        predictions[index].pTai * 10000
      ) / 10000,

      strength: Math.round(
        predictions[index].strength * 10000
      ) / 10000
    }))
  };
}

// ============ 11. TIME VN ============

function formatTimeVN() {
  const now = new Date(Date.now() + 7 * 3600 * 1000);

  return now.toISOString().replace('Z', '+07:00');
}

// ============ 12. RESPONSE ============

function buildResponse() {
  const history = readHistory();

  if (history.length === 0) {
    return {
      "id": "noname09",
      "error": "Chưa có dữ liệu, chờ sync",
      "uptime": formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  const seq = history.map(h => h.result);

  const ens = ensemblePredict(seq);

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

// ============ 13. ROUTES ============

app.get('/', (req, res) => res.json(buildResponse()));

app.get('/predict', (req, res) => res.json(buildResponse()));

app.get('/ping', (req, res) => res.send('pong'));

app.get('/history', (req, res) => {
  const h = readHistory();

  res.json({
    count: h.length,
    last20: h.slice(-20)
  });
});

// ============ 14. BOOT ============

initDataFile();

syncData();

setInterval(syncData, POLL_INTERVAL);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`);
});
