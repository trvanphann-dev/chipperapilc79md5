
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

  const tai = seq.reduce((sum, value) => sum + value, 0);

  return tai / seq.length;
}

function makePrediction(pTai, strength = 0) {
  const p = clamp(pTai, 0, 1);

  return {
    pick: p > 0.5 ? 1 : 0,
    pTai: p,
    strength: clamp(strength, 0, 1)
  };
}

// ============ 4. ALGORITHM 1: BAYESIAN PROBABILITY ============

function algoBayesian(seq) {
  const n = seq.length;

  if (n === 0) return makePrediction(0.5);

  const tai = seq.reduce((sum, value) => sum + value, 0);
  const xiu = n - tai;

  // Prior đối xứng, không thiên vị Tài hoặc Xỉu.
  const alpha = 2;
  const beta = 2;

  const pTai = (tai + alpha) / (n + alpha + beta);

  const strength = Math.abs(
    pTai - 0.5
  ) * 2;

  return makePrediction(pTai, strength);
}

// ============ 5. ALGORITHM 2: MARKOV TRANSITION ============

function algoMarkov(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const current = seq[n - 1];

  let tai = 0;
  let total = 0;

  for (let i = 1; i < n; i++) {
    if (seq[i - 1] !== current) continue;

    total++;

    if (seq[i] === 1) tai++;
  }

  if (total === 0) return makePrediction(0.5);

  // Làm trơn xác suất chuyển trạng thái.
  const pTai = (tai + 2) / (total + 4);

  const strength = (
    Math.abs(pTai - 0.5) * 2
  ) * (total / (total + 5));

  return makePrediction(pTai, strength);
}

// ============ 6. ALGORITHM 3: RUN-LENGTH STATISTICS ============

function algoRunLength(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const current = seq[n - 1];

  let runLength = 1;

  for (let i = n - 2; i >= 0; i--) {
    if (seq[i] !== current) break;
    runLength++;
  }

  // Phân tích tỷ lệ chuyển cửa của toàn bộ chuỗi.
  let transitions = 0;
  let switches = 0;

  for (let i = 1; i < n; i++) {
    if (seq[i - 1] !== current) continue;

    transitions++;

    if (seq[i] !== current) switches++;
  }

  if (transitions === 0) {
    return makePrediction(0.5);
  }

  const switchRate = (switches + 1) / (transitions + 2);

  // Phân tích xu hướng bệt hiện tại.
  const persistence = 1 - switchRate;

  // Không mặc định cầu bệt sẽ tiếp tục.
  // Chỉ sử dụng mức chuyển đổi thống kê quan sát được.
  const pSame = clamp(
    0.5 + (persistence - 0.5) *
    (runLength / (runLength + 3)),
    0.05,
    0.95
  );

  const pTai = current === 1
    ? pSame
    : 1 - pSame;

  const strength = (
    Math.abs(pSame - 0.5) * 2
  ) * (transitions / (transitions + 5));

  return makePrediction(pTai, strength);
}

// ============ 7. ALGORITHM 4: EWMA ============

function algoEWMA(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const alpha = 0.20;

  let average = seq[0];

  let variance = 0;

  for (let i = 1; i < n; i++) {
    const previous = average;

    average = alpha * seq[i] +
      (1 - alpha) * average;

    const difference = seq[i] - previous;

    variance = alpha * difference * difference +
      (1 - alpha) * variance;
  }

  const pTai = clamp(average, 0, 1);

  const strength = Math.abs(pTai - 0.5) * 2;

  return makePrediction(
    pTai,
    strength
  );
}

// ============ 8. ALGORITHM 5: SEQUENTIAL MOMENTUM ============

function algoMomentum(seq) {
  const n = seq.length;

  if (n < 5) return makePrediction(0.5);

  const windowSize = Math.min(30, n);

  const recent = seq.slice(-windowSize);

  let weightedTai = 0;
  let totalWeight = 0;

  // Trọng số giảm dần theo thời gian.
  for (let i = 0; i < recent.length; i++) {
    const weight = i + 1;

    weightedTai += recent[i] * weight;
    totalWeight += weight;
  }

  const weightedRate = weightedTai / totalWeight;

  const normalRate = getRate(recent);

  // Kết hợp xu hướng có trọng số và tỷ lệ thực tế.
  const pTai = (
    weightedRate * 0.65 +
    normalRate * 0.35
  );

  const strength = Math.abs(
    pTai - 0.5
  ) * 2;

  return makePrediction(
    pTai,
    strength
  );
}

// ============ 9. FIVE ALGORITHMS ============

const ALGOS = [
  { name: 'bayesian', fn: algoBayesian },
  { name: 'markov_transition', fn: algoMarkov },
  { name: 'run_length', fn: algoRunLength },
  { name: 'ewma', fn: algoEWMA },
  { name: 'sequential_momentum', fn: algoMomentum }
];

// ============ 10. ENSEMBLE ============

function ensemblePredict(seq) {
  if (seq.length < 10) {
    return {
      pick: 0,
      confidence: 50
    };
  }

  const predictions = ALGOS.map(algo => {
    return algo.fn(seq);
  });

  // Trọng số cố định, không học từ Backtesting.
  const weights = [
    1.0,
    1.2,
    0.8,
    1.0,
    1.0
  ];

  let weightedTai = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    weightedTai += predictions[i].pTai * weights[i];
    totalWeight += weights[i];
  }

  const pTai = weightedTai / totalWeight;

  // Tính mức đồng thuận của 5 thuật toán.
  let agreement = 0;

  for (const prediction of predictions) {
    if (
      (prediction.pTai > 0.5 && pTai > 0.5) ||
      (prediction.pTai < 0.5 && pTai < 0.5)
    ) {
      agreement++;
    }
  }

  const agreementRate = agreement / predictions.length;

  // Cường độ tín hiệu.
  const signalStrength = Math.abs(pTai - 0.5) * 2;

  // Độ tin cậy biểu thị mức đồng thuận và độ mạnh tín hiệu,
  // không phải xác suất thắng đã được kiểm chứng.
  const confidence = 50 +
    signalStrength * 25 +
    agreementRate * 25;

  const pick = pTai > 0.5 ? 1 : 0;

  return {
    pick,

    confidence: Math.round(
      clamp(confidence, 50, 99) * 100
    ) / 100,

    pTai: Math.round(pTai * 10000) / 10000,

    agreement: `${agreement}/${predictions.length}`,

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
