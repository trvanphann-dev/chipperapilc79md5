
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// =====================================================
// 1. RENDER CONFIGURATION
// =====================================================

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});

const PORT = process.env.PORT || 3000;

// Nếu đã gắn Persistent Disk trên Render:
// Mount Path: /var/data
const DATA_DIR = process.env.DATA_DIR || __dirname;

fs.mkdirSync(DATA_DIR, { recursive: true });

const DATA_FILE = path.join(
  DATA_DIR,
  'lichsulc79md5.json'
);

const AI_STATE_FILE = path.join(
  DATA_DIR,
  'ai_learning_state.json'
);

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;

// =====================================================
// 2. STORAGE
// =====================================================

function atomicWrite(file, data) {
  const temp = file + '.tmp';

  fs.writeFileSync(
    temp,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(temp, file);
}

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    atomicWrite(DATA_FILE, []);
    console.log('[INIT] History created');
  }
}

function readHistory() {
  try {
    const parsed = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[READ ERROR]', err.message);
    return [];
  }
}

function writeHistory(data) {
  atomicWrite(DATA_FILE, data);
}

// =====================================================
// 3. AI DEFAULT STATE
// =====================================================

const MODEL_NAMES = [
  'bayes',
  'markov',
  'pattern',
  'trend',
  'streak'
];

function createDefaultState() {
  const modelStats = {};

  for (const name of MODEL_NAMES) {
    modelStats[name] = {
      correct: 0,
      total: 0,
      brier: 0
    };
  }

  return {
    version: 2,

    weights: {
      bayes: 1,
      markov: 1,
      pattern: 1,
      trend: 1,
      streak: 1
    },

    modelStats,

    stats: {
      total: 0,
      win: 0,
      lose: 0
    },

    compare: [],
    caution: [],

    pending: null,

    lastUpdated: null
  };
}

// =====================================================
// 4. LOAD AI STATE
// =====================================================

function loadAIState() {
  const defaults = createDefaultState();

  try {
    if (!fs.existsSync(AI_STATE_FILE)) {
      return defaults;
    }

    const saved = JSON.parse(
      fs.readFileSync(AI_STATE_FILE, 'utf8')
    );

    const state = {
      ...defaults,
      ...saved,

      weights: {
        ...defaults.weights,
        ...(saved.weights || {})
      },

      modelStats: {
        ...defaults.modelStats,
        ...(saved.modelStats || {})
      },

      stats: {
        ...defaults.stats,
        ...(saved.stats || {})
      },

      compare: Array.isArray(saved.compare)
        ? saved.compare.slice(-10)
        : [],

      caution: Array.isArray(saved.caution)
        ? saved.caution.slice(-10)
        : [],

      pending: saved.pending || null
    };

    for (const name of MODEL_NAMES) {
      state.modelStats[name] = {
        ...defaults.modelStats[name],
        ...(state.modelStats[name] || {})
      };
    }

    return state;

  } catch (err) {
    console.error('[AI LOAD ERROR]', err.message);
    return defaults;
  }
}

let AI = loadAIState();

function saveAIState() {
  AI.lastUpdated = new Date().toISOString();

  try {
    atomicWrite(AI_STATE_FILE, AI);
  } catch (err) {
    console.error('[AI SAVE ERROR]', err.message);
  }
}

// =====================================================
// 5. GLOBAL STATE
// =====================================================

let lastId = 0;
let syncInProgress = false;
let historyResetRequested = false;

const STATS = AI.stats;
const COMPARE = AI.compare;
const CAUTION = AI.caution;

let lastPredict = AI.pending;

// =====================================================
// 6. RESET HISTORY
// =====================================================

function resetHistory() {
  try {
    writeHistory([]);

    AI.caution = [];
    AI.pending = null;

    lastPredict = null;

    historyResetRequested = true;

    saveAIState();

    console.log('[RESET] History reset');

  } catch (err) {
    console.error('[RESET ERROR]', err.message);
  }
}

// =====================================================
// 7. UTILITIES
// =====================================================

function clamp(value, min = 0.001, max = 0.999) {
  return Math.max(min, Math.min(max, value));
}

function average(arr) {
  if (!arr.length) return 0;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function entropy(seq, windowSize = 30) {
  if (seq.length < windowSize) return 0;

  const data = seq.slice(-windowSize);

  const p1 = average(data);
  const p0 = 1 - p1;

  if (p0 === 0 || p1 === 0) return 0;

  return -(
    p0 * Math.log2(p0) +
    p1 * Math.log2(p1)
  );
}

function chaosBlock(seq) {
  return (
    seq.length >= 30 &&
    entropy(seq, 30) > 0.92
  );
}

function getStreak(seq) {
  if (!seq.length) {
    return {
      value: 0,
      length: 0
    };
  }

  const value = seq[seq.length - 1];

  let length = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] !== value) break;
    length++;
  }

  return {
    value,
    length
  };
}

// =====================================================
// 8. MODEL 1 - ADAPTIVE BAYESIAN
// =====================================================

function bayesianPredict(seq) {
  let alpha = 2;
  let beta = 2;

  const data = seq.slice(-150);

  data.forEach((result, index) => {
    const weight = Math.exp(
      -(data.length - 1 - index) / 50
    );

    if (result === 1) {
      alpha += weight;
    } else {
      beta += weight;
    }
  });

  return clamp(alpha / (alpha + beta));
}

// =====================================================
// 9. MODEL 2 - MULTI ORDER MARKOV
// =====================================================

function markovPredict(seq) {
  if (seq.length < 3) return 0.5;

  let weightedProbability = 0;
  let totalWeight = 0;

  for (let order = 1; order <= 10; order++) {
    if (seq.length <= order) continue;

    const target = seq.slice(-order);

    let tai = 1.5;
    let xiu = 1.5;
    let support = 0;

    for (
      let i = 0;
      i + order < seq.length;
      i++
    ) {
      let match = true;

      for (let j = 0; j < order; j++) {
        if (seq[i + j] !== target[j]) {
          match = false;
          break;
        }
      }

      if (!match) continue;

      const next = seq[i + order];

      const recency = Math.exp(
        -(seq.length - i - order) / 100
      );

      if (next === 1) {
        tai += recency;
      } else {
        xiu += recency;
      }

      support += recency;
    }

    if (support < 0.5) continue;

    const probability = tai / (tai + xiu);

    const reliability = Math.min(
      1,
      support / 8
    );

    const weight =
      reliability * (1 + order * 0.1);

    weightedProbability += probability * weight;
    totalWeight += weight;
  }

  if (!totalWeight) return 0.5;

  return clamp(
    weightedProbability / totalWeight
  );
}

// =====================================================
// 10. MODEL 3 - PATTERN RECOGNITION
// =====================================================

function patternPredict(seq) {
  if (seq.length < 10) return 0.5;

  let tai = 1;
  let xiu = 1;
  let support = 0;

  const maxOrder = Math.min(
    12,
    seq.length - 1
  );

  for (let order = 3; order <= maxOrder; order++) {
    const target = seq.slice(-order);

    for (
      let i = 0;
      i + order < seq.length;
      i++
    ) {
      let matches = 0;

      for (let j = 0; j < order; j++) {
        if (seq[i + j] === target[j]) {
          matches++;
        }
      }

      const similarity = matches / order;

      if (similarity < 0.85) continue;

      const recency = Math.exp(
        -(seq.length - i - order) / 100
      );

      const weight =
        similarity * similarity * recency;

      const next = seq[i + order];

      if (next === 1) {
        tai += weight;
      } else {
        xiu += weight;
      }

      support += weight;
    }
  }

  if (support < 0.1) return 0.5;

  return clamp(tai / (tai + xiu));
}

// =====================================================
// 11. MODEL 4 - TREND ANALYSIS
// =====================================================

function trendPredict(seq) {
  const windows = [10, 20, 40, 80, 150];

  let result = 0;
  let totalWeight = 0;

  windows.forEach((size, index) => {
    if (seq.length < size) return;

    const data = seq.slice(-size);

    const probability = average(data);

    const weight = 1 / Math.pow(
      index + 1,
      0.8
    );

    result += probability * weight;
    totalWeight += weight;
  });

  if (!totalWeight) return 0.5;

  return clamp(result / totalWeight);
}

// =====================================================
// 12. MODEL 5 - STREAK INTELLIGENCE
// =====================================================

function streakPredict(seq) {
  if (seq.length < 5) return 0.5;

  const current = getStreak(seq);

  let tai = 1;
  let xiu = 1;
  let support = 0;

  for (
    let i = 0;
    i < seq.length - current.length;
    i++
  ) {
    let count = 1;

    while (
      i + count < seq.length &&
      seq[i + count] === seq[i]
    ) {
      count++;
    }

    if (count !== current.length) continue;

    if (i + count >= seq.length) continue;

    const next = seq[i + count];

    const weight = Math.exp(
      -(seq.length - i - count) / 100
    );

    if (next === 1) {
      tai += weight;
    } else {
      xiu += weight;
    }

    support += weight;
  }

  if (support < 0.1) return 0.5;

  return clamp(tai / (tai + xiu));
}

// =====================================================
// 13. GET ALL MODEL PREDICTIONS
// =====================================================

function getModelPredictions(seq) {
  return {
    bayes: bayesianPredict(seq),
    markov: markovPredict(seq),
    pattern: patternPredict(seq),
    trend: trendPredict(seq),
    streak: streakPredict(seq)
  };
}

// =====================================================
// 14. ONLINE LEARNING
// =====================================================

const LEARNING_RATE = 0.12;
const MIN_WEIGHT = 0.15;
const MAX_WEIGHT = 4;

function updateModelLearning(predictions, actual) {
  for (const name of MODEL_NAMES) {
    const probability = clamp(
      Number(predictions[name] ?? 0.5)
    );

    // Brier Score
    const loss = Math.pow(
      probability - actual,
      2
    );

    const predicted =
      probability >= 0.5 ? 1 : 0;

    const correct = predicted === actual;

    const stats = AI.modelStats[name];

    stats.total++;

    if (correct) {
      stats.correct++;
    }

    // Cập nhật Brier trung bình
    stats.brier += (
      loss - stats.brier
    ) / stats.total;

    // Adaptive Weight
    const adjustment = Math.exp(
      LEARNING_RATE * (0.25 - loss)
    );

    AI.weights[name] = clamp(
      AI.weights[name] * adjustment,
      MIN_WEIGHT,
      MAX_WEIGHT
    );
  }
}

// =====================================================
// 15. ENSEMBLE ENGINE
// =====================================================

function calculateFinal(seq) {
  const models = getModelPredictions(seq);

  if (seq.length < 10) {
    return {
      final: 'TÀI',
      confidence: 50,
      probability: 0.5,
      caution: true,
      models
    };
  }

  let weighted = 0;
  let totalWeight = 0;

  for (const name of MODEL_NAMES) {
    const probability = clamp(models[name]);

    const weight = clamp(
      AI.weights[name] || 1,
      MIN_WEIGHT,
      MAX_WEIGHT
    );

    weighted += probability * weight;
    totalWeight += weight;
  }

  let probability = totalWeight
    ? weighted / totalWeight
    : 0.5;

  const caution = chaosBlock(seq);

  if (caution) {
    probability =
      0.5 + (probability - 0.5) * 0.5;
  }

  probability = clamp(probability);

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  const confidence = Math.round(
    50 + Math.abs(probability - 0.5) * 100
  );

  return {
    final,

    confidence: Math.min(
      95,
      confidence
    ),

    probability,
    caution,
    models
  };
}

// =====================================================
// 16. SETTLE PREVIOUS PREDICTION
// =====================================================

function settlePrediction(session) {
  const pending = AI.pending;

  if (!pending) return;

  // Chỉ chấm đúng phiên đã được dự đoán
  if (Number(pending.session) !== Number(session.id)) {
    console.warn(
      '[AI] Pending session mismatch:',
      pending.session,
      session.id
    );

    // Không gán nhầm kết quả cho dự đoán khác
    AI.pending = null;
    lastPredict = null;

    return;
  }

  const actual = session.result;

  const real = actual === 1
    ? 'TÀI'
    : 'XỈU';

  const win = pending.predict === real;

  AI.stats.total++;

  if (win) {
    AI.stats.win++;
  } else {
    AI.stats.lose++;
  }

  // Học từng mô hình độc lập
  updateModelLearning(
    pending.models,
    actual
  );

  AI.compare.push({
    phien: session.id,
    predict: pending.predict,
    real,
    win,
    confidence: pending.confidence
  });

  if (AI.compare.length > 10) {
    AI.compare.shift();
  }

  // Theo dõi 10 lần Lưu Ý
  if (pending.caution) {
    AI.caution.push(win ? 1 : 0);

    if (AI.caution.length > 10) {
      AI.caution.shift();
    }

    if (AI.caution.length === 10) {
      const wins = AI.caution.reduce(
        (a, b) => a + b,
        0
      );

      const losses = 10 - wins;

      // Giữ cơ chế reset lịch sử cũ
      if (losses >= wins) {
        console.log(
          '[AI] Caution reset condition reached'
        );

        resetHistory();
      }
    }
  }

  console.log(
    `[AI LEARNING] ${pending.predict} -> ${real} | ` +
    `${win ? 'WIN' : 'LOSE'} | ` +
    `Accuracy: ${AI.stats.win}/${AI.stats.total}`
  );

  AI.pending = null;
  lastPredict = null;
}

// =====================================================
// 17. GENERATE NEXT PREDICTION
// =====================================================

function generateNextPrediction(history) {
  if (!history.length) return null;

  const seq = history.map(h => h.result);

  const result = calculateFinal(seq);

  const current = history[history.length - 1];

  const pending = {
    session: Number(current.id) + 1,

    predict: result.final,

    confidence: result.confidence,

    probability: result.probability,

    caution: result.caution,

    models: result.models,

    createdAt: new Date().toISOString()
  };

  AI.pending = pending;
  lastPredict = pending;

  return pending;
}

// =====================================================
// 18. SOURCE API
// =====================================================

async function fetchSource() {
  const res = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(10000)
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }

  return await res.json();
}

// =====================================================
// 19. NORMALIZE SESSION
// =====================================================

function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;

  const rawDices =
    raw.dices ??
    raw.dice ??
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
    (
      typeof raw.result === 'string'
        ? raw.result
        : ''
    )
  ).trim().toUpperCase();

  let result;

  if (
    ['TAI', 'TÀI', 'BIG'].includes(sourceResult)
  ) {
    result = 1;
  } else if (
    ['XIU', 'XỈU', 'SMALL'].includes(sourceResult)
  ) {
    result = 0;
  } else {
    result = sum >= 11 ? 1 : 0;
  }

  return {
    id: Number(id),
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// =====================================================
// 20. SYNC DATA
// =====================================================

async function syncData() {
  if (syncInProgress) return;

  syncInProgress = true;

  try {
    const raw = await fetchSource();

    const list = Array.isArray(raw)
      ? raw
      : raw.data ?? raw.sessions ?? raw.list ?? [];

    if (!Array.isArray(list) || !list.length) {
      return;
    }

    let history = readHistory();

    const knownIds = new Set(
      history.map(h => Number(h.id))
    );

    const fresh = list
      .map(normalizeSession)
      .filter(s =>
        Number.isFinite(s.id) &&
        Number.isFinite(s.sum) &&
        [0, 1].includes(s.result) &&
        s.id > lastId &&
        !knownIds.has(s.id)
      )
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    for (const session of fresh) {
      // 1. Chấm dự đoán phiên hiện tại
      settlePrediction(session);

      // Nếu reset được kích hoạt
      if (historyResetRequested) {
        history = [];
        historyResetRequested = false;
      }

      // 2. Lưu kết quả thực tế
      history.push(session);

      history.sort((a, b) => a.id - b.id);

      // 3. Giới hạn lịch sử
      if (history.length > MAX_HISTORY) {
        history.splice(
          0,
          history.length - MAX_HISTORY
        );
      }

      lastId = session.id;

      // 4. Tạo dự đoán phiên tiếp theo
      generateNextPrediction(history);

      // 5. Lưu lịch sử và trạng thái AI
      writeHistory(history);
      saveAIState();

      console.log(
        `[SYNC] Session ${session.id} | ` +
        `Next: ${AI.pending?.predict}`
      );
    }

    console.log(
      `[SYNC] +${fresh.length} sessions`
    );

  } catch (err) {
    console.error(
      '[SYNC ERROR]',
      err.message
    );
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 21. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace(
    'Z',
    '+07:00'
  );
}

// =====================================================
// 22. BUILD API RESPONSE
// =====================================================

function buildResponse() {
  const history = readHistory();

  if (!history.length) {
    return {
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime: formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  // Khôi phục dự đoán nếu chưa có
  let pending = AI.pending;

  if (
    !pending ||
    Number(pending.session) !== Number(current.id) + 1
  ) {
    pending = generateNextPrediction(history);
    saveAIState();
  }

  const accuracy = AI.stats.total
    ? Math.round(
        AI.stats.win / AI.stats.total * 100
      )
    : 0;

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: pending.session,

    dudoan: pending.predict === 'TÀI'
      ? 'tai'
      : 'xiu',

    dotincay: `${pending.confidence}%`,

    ai: {
      version: 'SELF-LEARNING V2',

      total: AI.stats.total,

      win: AI.stats.win,

      lose: AI.stats.lose,

      accuracy: `${accuracy}%`,

      caution: pending.caution,

      probability: pending.probability,

      models: pending.models,

      weights: AI.weights,

      modelStats: AI.modelStats
    },

    uptime: formatTimeVN()
  };
}

// =====================================================
// 23. MAIN API
// =====================================================

app.get('/', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[API ERROR]', err.message);

    res.status(500).json({
      error: 'Internal server error'
    });
  }
});

app.get('/predict', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[PREDICT ERROR]', err.message);

    res.status(500).json({
      error: 'Prediction error'
    });
  }
});

app.get('/ping', (req, res) => {
  res.send('pong');
});

app.get('/history', (req, res) => {
  try {
    const h = readHistory();

    res.json({
      count: h.length,
      last20: h.slice(-20)
    });
  } catch (err) {
    res.status(500).json({
      error: 'History error'
    });
  }
});

// =====================================================
// 24. AI STATUS
// =====================================================

app.get('/ai-status', (req, res) => {
  res.json({
    version: 'SELF-LEARNING V2',

    stats: AI.stats,

    weights: AI.weights,

    modelStats: AI.modelStats,

    pending: AI.pending,

    lastUpdated: AI.lastUpdated,

    storage: {
      directory: DATA_DIR,
      historyExists: fs.existsSync(DATA_FILE),
      aiStateExists: fs.existsSync(AI_STATE_FILE)
    }
  });
});

// =====================================================
// 25. BOOT SERVER
// =====================================================

initDataFile();

{
  const history = readHistory();

  lastId = history.length
    ? Math.max(...history.map(h => Number(h.id)))
    : 0;

  const current = history[history.length - 1];

  // Chỉ khôi phục pending nếu đúng phiên kế tiếp
  if (
    !current ||
    !AI.pending ||
    Number(AI.pending.session) !== Number(current.id) + 1
  ) {
    AI.pending = null;
  }

  if (history.length && !AI.pending) {
    generateNextPrediction(history);
  }

  lastPredict = AI.pending;

  saveAIState();
}

// Đồng bộ lần đầu
syncData();

// Đồng bộ mỗi 15 giây
const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

// =====================================================
// 26. START SERVER
// =====================================================

app.listen(PORT, '0.0.0.0', () => {
  console.log('====================================');
  console.log('[BOOT] RENDER AI SELF-LEARNING V2');
  console.log(`[BOOT] PORT: ${PORT}`);
  console.log(`[BOOT] DATA: ${DATA_DIR}`);
  console.log('[BOOT] Bayesian loaded');
  console.log('[BOOT] Markov loaded');
  console.log('[BOOT] Pattern Recognition loaded');
  console.log('[BOOT] Trend Analysis loaded');
  console.log('[BOOT] Streak Intelligence loaded');
  console.log('[BOOT] Online Learning enabled');
  console.log('[BOOT] Persistent State enabled');
  console.log('====================================');
});
