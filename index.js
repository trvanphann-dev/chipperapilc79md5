
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// =====================================================
// 1. CONFIG
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

const DATA_DIR = process.env.DATA_DIR || __dirname;

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

const DATA_FILE = path.join(DATA_DIR, 'lichsulc79md5.json');
const AI_FILE = path.join(DATA_DIR, 'ai_learning_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const PORT = process.env.PORT || 3000;

// =====================================================
// 2. STORAGE
// =====================================================

function atomicWrite(file, data) {
  const temp = file + '.tmp';
  fs.writeFileSync(temp, data, 'utf8');
  fs.renameSync(temp, file);
}

function readHistory() {
  try {
    if (!fs.existsSync(DATA_FILE)) {
      atomicWrite(DATA_FILE, '[]');
      return [];
    }

    const parsed = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[HISTORY READ]', err.message);
    return [];
  }
}

function writeHistory(history) {
  atomicWrite(
    DATA_FILE,
    JSON.stringify(history.slice(-MAX_HISTORY), null, 2)
  );
}

function defaultAIState() {
  return {
    version: 'AI-ADAPTIVE-V3',
    weights: {
      recent: 1,
      markov: 1,
      pattern: 1,
      trend: 1,
      streak: 1
    },
    modelStats: {
      recent: { total: 0, correct: 0, brier: 0 },
      markov: { total: 0, correct: 0, brier: 0 },
      pattern: { total: 0, correct: 0, brier: 0 },
      trend: { total: 0, correct: 0, brier: 0 },
      streak: { total: 0, correct: 0, brier: 0 }
    },
    stats: {
      total: 0,
      win: 0,
      lose: 0
    },
    compare: [],
    pending: null,
    lastLearnedId: null
  };
}

function loadAIState() {
  const defaults = defaultAIState();

  try {
    if (!fs.existsSync(AI_FILE)) {
      return defaults;
    }

    const saved = JSON.parse(
      fs.readFileSync(AI_FILE, 'utf8')
    );

    return {
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
        ? saved.compare
        : [],
      pending: saved.pending || null
    };
  } catch (err) {
    console.error('[AI STATE READ]', err.message);
    return defaults;
  }
}

let AI = loadAIState();

function saveAIState() {
  atomicWrite(AI_FILE, JSON.stringify(AI, null, 2));
}

// =====================================================
// 3. SOURCE API
// =====================================================

async function fetchSource() {
  const response = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(10000)
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json();
}

function normalizeSession(raw) {
  const id = Number(
    raw.id ?? raw.session ?? raw.issue
  );

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
    (typeof raw.result === 'string' ? raw.result : '')
  ).trim().toUpperCase();

  let result;

  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) {
    result = 1;
  } else if (['XIU', 'XỈU', 'SMALL'].includes(sourceResult)) {
    result = 0;
  } else if (sum >= 11 && sum <= 18) {
    result = 1;
  } else if (sum >= 3 && sum <= 10) {
    result = 0;
  } else {
    return null;
  }

  if (!Number.isFinite(id)) return null;

  return {
    id,
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// =====================================================
// 4. COMMON FUNCTIONS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function average(arr) {
  if (!arr.length) return 0.5;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function binaryEntropy(p) {
  p = clamp(p, 0.000001, 0.999999);

  return -(
    p * Math.log2(p) +
    (1 - p) * Math.log2(1 - p)
  );
}

// =====================================================
// 5. MODEL 1: RECENCY BAYES
// Dữ liệu mới có trọng số lớn hơn dữ liệu cũ.
// =====================================================

function predictRecent(seq) {
  if (!seq.length) return 0.5;

  let weightedTai = 1;
  let weightedXiu = 1;
  let weight = 1;

  for (let i = seq.length - 1; i >= 0; i--) {
    if (seq[i] === 1) {
      weightedTai += weight;
    } else {
      weightedXiu += weight;
    }

    weight *= 0.975;

    if (weight < 0.03) break;
  }

  return weightedTai / (weightedTai + weightedXiu);
}

// =====================================================
// 6. MODEL 2: MULTI-ORDER MARKOV
// Học xác suất kết quả tiếp theo theo nhiều cấp cầu.
// =====================================================

function predictMarkov(seq) {
  if (seq.length < 5) return 0.5;

  const predictions = [];
  const weights = [];

  for (let order = 1; order <= 5; order++) {
    if (seq.length <= order + 2) continue;

    const key = seq.slice(-order).join('');

    let tai = 1;
    let xiu = 1;
    let matches = 0;

    for (let i = order; i < seq.length; i++) {
      const previous = seq
        .slice(i - order, i)
        .join('');

      if (previous !== key) continue;

      if (seq[i] === 1) tai++;
      else xiu++;

      matches++;
    }

    if (matches >= 2) {
      predictions.push(tai / (tai + xiu));
      weights.push(Math.sqrt(matches) * order);
    }
  }

  if (!predictions.length) return 0.5;

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < predictions.length; i++) {
    numerator += predictions[i] * weights[i];
    denominator += weights[i];
  }

  return numerator / denominator;
}

// =====================================================
// 7. MODEL 3: PATTERN RECOGNITION
// Tìm những chuỗi tương tự trong lịch sử.
// =====================================================

function predictPattern(seq) {
  if (seq.length < 8) return 0.5;

  const predictions = [];
  const weights = [];

  for (let length = 3; length <= 8; length++) {
    if (seq.length <= length + 3) continue;

    const target = seq.slice(-length);

    let tai = 1;
    let xiu = 1;
    let matches = 0;

    for (
      let i = length;
      i < seq.length;
      i++
    ) {
      const oldPattern = seq.slice(
        i - length,
        i
      );

      if (
        oldPattern.every(
          (value, index) => value === target[index]
        )
      ) {
        if (seq[i] === 1) tai++;
        else xiu++;

        matches++;
      }
    }

    if (matches >= 1) {
      const probability = tai / (tai + xiu);

      predictions.push(probability);
      weights.push(Math.sqrt(matches) * length);
    }
  }

  if (!predictions.length) return 0.5;

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < predictions.length; i++) {
    numerator += predictions[i] * weights[i];
    denominator += weights[i];
  }

  return numerator / denominator;
}

// =====================================================
// 8. MODEL 4: TREND ANALYSIS
// Phân tích xu hướng nhiều cửa sổ thời gian.
// =====================================================

function predictTrend(seq) {
  if (!seq.length) return 0.5;

  const windows = [8, 15, 30, 60, 120];
  const weights = [1, 1.3, 1.5, 1.2, 0.8];

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < windows.length; i++) {
    const size = windows[i];

    if (seq.length < Math.min(size, 5)) {
      continue;
    }

    const sample = seq.slice(-size);
    const p = average(sample);

    numerator += p * weights[i];
    denominator += weights[i];
  }

  if (!denominator) return 0.5;

  return numerator / denominator;
}

// =====================================================
// 9. MODEL 5: STREAK / REVERSAL
// Không mặc định cầu bệt hay cầu đảo.
// Dùng thống kê thực tế sau các chuỗi tương tự.
// =====================================================

function predictStreak(seq) {
  if (seq.length < 8) return 0.5;

  const last = seq[seq.length - 1];

  let streak = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === last) streak++;
    else break;
  }

  let tai = 1;
  let xiu = 1;
  let matches = 0;

  for (let i = 1; i < seq.length; i++) {
    let oldStreak = 1;

    for (let j = i - 2; j >= 0; j--) {
      if (seq[j] === seq[i - 1]) {
        oldStreak++;
      } else {
        break;
      }
    }

    if (
      seq[i - 1] === last &&
      oldStreak === streak
    ) {
      if (seq[i] === 1) tai++;
      else xiu++;

      matches++;
    }
  }

  if (matches < 2) return 0.5;

  return tai / (tai + xiu);
}

// =====================================================
// 10. GET ALL MODEL PREDICTIONS
// =====================================================

function getModelPredictions(seq) {
  return {
    recent: clamp(predictRecent(seq), 0.01, 0.99),
    markov: clamp(predictMarkov(seq), 0.01, 0.99),
    pattern: clamp(predictPattern(seq), 0.01, 0.99),
    trend: clamp(predictTrend(seq), 0.01, 0.99),
    streak: clamp(predictStreak(seq), 0.01, 0.99)
  };
}

// =====================================================
// 11. ADAPTIVE ENSEMBLE
// Tự điều chỉnh trọng số theo hiệu quả từng mô hình.
// =====================================================

function calculatePrediction(history) {
  const seq = history.map(h => h.result);
  const models = getModelPredictions(seq);

  let numerator = 0;
  let denominator = 0;

  for (const name of Object.keys(models)) {
    const weight = clamp(
      Number(AI.weights[name]) || 1,
      0.2,
      5
    );

    numerator += models[name] * weight;
    denominator += weight;
  }

  let probability = denominator
    ? numerator / denominator
    : 0.5;

  // Ít dữ liệu học được -> giảm độ cực đoan.
  const learned = AI.stats.total;
  const reliability = Math.min(
    1,
    Math.sqrt(learned / 40)
  );

  probability =
    0.5 + (probability - 0.5) * reliability;

  // Khi dữ liệu gần đây quá khó phân biệt,
  // không ép mô hình đưa ra xác suất quá cực đoan.
  const recent = seq.slice(-30);
  const uncertainty = recent.length >= 15
    ? binaryEntropy(average(recent))
    : 1;

  if (uncertainty > 0.97) {
    probability =
      0.5 + (probability - 0.5) * 0.75;
  }

  probability = clamp(probability, 0.01, 0.99);

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  // Đây là điểm độ mạnh dự đoán, không phải
  // xác suất thắng đã được kiểm chứng.
  const confidence = clamp(
    50 + Math.abs(probability - 0.5) * 100,
    50,
    75
  );

  return {
    final,
    probability,
    confidence: Math.round(confidence),
    models
  };
}

// =====================================================
// 12. ONLINE LEARNING
// Chấm từng mô hình riêng biệt bằng Brier Score.
// =====================================================

function learnFromPrediction(pending, actual) {
  if (!pending || !pending.models) return;

  const actualValue = actual === 1 ? 1 : 0;
  const predictedFinal = pending.probability >= 0.5 ? 1 : 0;
  const win = predictedFinal === actualValue;

  AI.stats.total++;

  if (win) AI.stats.win++;
  else AI.stats.lose++;

  const names = Object.keys(AI.weights);

  for (const name of names) {
    const p = clamp(
      Number(pending.models[name] ?? 0.5),
      0,
      1
    );

    const loss = Math.pow(p - actualValue, 2);
    const stats = AI.modelStats[name];

    stats.total++;
    stats.brier += loss;

    if ((p >= 0.5 ? 1 : 0) === actualValue) {
      stats.correct++;
    }

    // Mô hình có Brier loss thấp hơn 0.25
    // được tăng trọng số; mô hình kém bị giảm.
    const eta = 0.12;
    const adjustment = Math.exp(
      eta * (0.25 - loss)
    );

    AI.weights[name] = clamp(
      AI.weights[name] * adjustment,
      0.2,
      5
    );
  }

  AI.compare.push({
    phien: actual.id,
    predict: pending.final,
    real: actualValue === 1 ? 'TÀI' : 'XỈU',
    win,
    probability: Number(
      pending.probability.toFixed(4)
    ),
    time: new Date().toISOString()
  });

  if (AI.compare.length > 100) {
    AI.compare = AI.compare.slice(-100);
  }

  AI.lastLearnedId = actual.id;

  console.log(
    `[LEARN] Session ${actual.id} | ` +
    `${win ? 'WIN' : 'LOSE'} | ` +
    `Accuracy ${(
      AI.stats.win / AI.stats.total * 100
    ).toFixed(2)}%`
  );
}

// =====================================================
// 13. CREATE NEXT PREDICTION
// =====================================================

function createPending(history, targetId) {
  if (!history.length) return null;

  const prediction = calculatePrediction(history);

  return {
    targetId,
    final: prediction.final,
    probability: prediction.probability,
    confidence: prediction.confidence,
    models: prediction.models,
    createdAt: new Date().toISOString()
  };
}

// =====================================================
// 14. SESSION PROCESSING
// Chấm đúng từng phiên, kể cả khi API trả về nhiều phiên.
// =====================================================

function processSession(session, history) {
  if (AI.pending) {
    if (AI.pending.targetId === session.id) {
      learnFromPrediction(AI.pending, session);
      AI.pending = null;
    } else if (AI.pending.targetId < session.id) {
      // Bị thiếu phiên: không lấy kết quả sai phiên
      // để chấm điểm dự đoán cũ.
      console.log(
        `[LEARN] Skip missing target ${AI.pending.targetId}`
      );

      AI.pending = null;
    }
  }

  history.push(session);

  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  // Dự đoán cho phiên kế tiếp.
  AI.pending = createPending(
    history,
    session.id + 1
  );
}

// =====================================================
// 15. SYNC SOURCE
// =====================================================

let syncInProgress = false;
let lastId = 0;

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

    const history = readHistory();
    const knownIds = new Set(history.map(h => Number(h.id)));

    const fresh = list
      .map(normalizeSession)
      .filter(s =>
        s &&
        Number.isFinite(s.id) &&
        [0, 1].includes(s.result) &&
        !knownIds.has(s.id)
      )
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    let added = 0;

    for (const session of fresh) {
      if (knownIds.has(session.id)) continue;

      processSession(session, history);

      knownIds.add(session.id);
      lastId = Math.max(lastId, session.id);
      added++;
    }

    if (added > 0) {
      history.sort((a, b) => a.id - b.id);

      writeHistory(history);
      saveAIState();

      console.log(
        `[SYNC] +${added} sessions | ` +
        `History ${history.length} | ` +
        `Learned ${AI.stats.total}`
      );
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 16. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  return new Date(
    Date.now() + 7 * 3600 * 1000
  ).toISOString().replace('Z', '+07:00');
}

// =====================================================
// 17. API RESPONSE
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

  const prediction = calculatePrediction(history);

  const targetId = current.id + 1;

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,

    dudoanphien: targetId,
    dudoan: prediction.final === 'TÀI'
      ? 'tai'
      : 'xiu',

    dotincay: `${prediction.confidence}%`,
    uptime: formatTimeVN(),

    ai: {
      version: AI.version,
      probability: Number(
        prediction.probability.toFixed(4)
      ),
      models: Object.fromEntries(
        Object.entries(prediction.models).map(
          ([key, value]) => [
            key,
            Number(value.toFixed(4))
          ]
        )
      ),
      weights: AI.weights,
      learned: AI.stats.total,
      accuracy: AI.stats.total
        ? Number(
            (
              AI.stats.win /
              AI.stats.total *
              100
            ).toFixed(2)
          )
        : 0
    }
  };
}

// =====================================================
// 18. API ROUTES
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
    const history = readHistory();

    res.json({
      count: history.length,
      last20: history.slice(-20)
    });
  } catch (err) {
    res.status(500).json({
      error: 'History error'
    });
  }
});

app.get('/ai-status', (req, res) => {
  const accuracy = AI.stats.total
    ? AI.stats.win / AI.stats.total * 100
    : 0;

  const modelStatus = {};

  for (const name of Object.keys(AI.weights)) {
    const stats = AI.modelStats[name];

    modelStatus[name] = {
      weight: Number(
        AI.weights[name].toFixed(4)
      ),
      total: stats.total,
      accuracy: stats.total
        ? Number(
            (
              stats.correct /
              stats.total *
              100
            ).toFixed(2)
          )
        : 0,
      averageBrier: stats.total
        ? Number(
            (
              stats.brier /
              stats.total
            ).toFixed(4)
          )
        : null
    };
  }

  res.json({
    version: AI.version,
    total: AI.stats.total,
    win: AI.stats.win,
    lose: AI.stats.lose,
    accuracy: Number(accuracy.toFixed(2)),
    weights: AI.weights,
    models: modelStatus,
    pending: AI.pending,
    lastLearnedId: AI.lastLearnedId
  });
});

// =====================================================
// 19. BOOT
// =====================================================

if (!fs.existsSync(DATA_FILE)) {
  atomicWrite(DATA_FILE, '[]');
}

const bootHistory = readHistory();

if (bootHistory.length) {
  lastId = Math.max(
    ...bootHistory.map(h => Number(h.id) || 0)
  );

  // Nếu chưa có dự đoán chờ, tạo lại dự đoán
  // cho phiên kế tiếp từ lịch sử đã lưu.
  const current = bootHistory[bootHistory.length - 1];

  if (
    !AI.pending ||
    AI.pending.targetId !== current.id + 1
  ) {
    AI.pending = createPending(
      bootHistory,
      current.id + 1
    );

    saveAIState();
  }
}

console.log('[BOOT] Adaptive AI V3');
console.log('[BOOT] History:', bootHistory.length);
console.log('[BOOT] Learned:', AI.stats.total);

syncData();

const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `[SERVER] Running on port ${PORT}`
  );

  console.log(
    `[SERVER] ${formatTimeVN()}`
  );
});
