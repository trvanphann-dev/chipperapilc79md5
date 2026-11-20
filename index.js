
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// CORS
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const MODEL_FILE = path.join(__dirname, 'model_lc79.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const PORT = process.env.PORT || 3000;

// =====================================================
// 1. DATA STORAGE
// =====================================================

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, '[]', 'utf8');
    console.log('[INIT] Created history file');
  } else {
    console.log('[INIT] Existing history preserved');
  }
}

function readHistory() {
  try {
    const data = fs.readFileSync(DATA_FILE, 'utf8');
    const parsed = JSON.parse(data || '[]');

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[READ ERROR]', err.message);
    return [];
  }
}

function writeJsonAtomic(file, data) {
  const tempFile = file + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, file);
}

function writeHistory(arr) {
  writeJsonAtomic(DATA_FILE, arr);
}

function saveModel() {
  try {
    writeJsonAtomic(MODEL_FILE, {
      version: 2,
      models: MODELS,
      pending: lastPredict,
      stats: STATS,
      compare: COMPARE,
      cautionData: CAUTION.data
    });
  } catch (err) {
    console.error('[MODEL SAVE ERROR]', err.message);
  }
}

// =====================================================
// 2. SOURCE API SYNC
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

function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;

  const rawDices =
    raw.dices ??
    raw.dice ??
    (Array.isArray(raw.result) ? raw.result : []);

  const dices = Array.isArray(rawDices)
    ? rawDices.map(Number)
    : [];

  const validDices = dices.length === 3 &&
    dices.every(v => Number.isInteger(v) && v >= 1 && v <= 6);

  const sum = validDices
    ? dices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? 0);

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    (typeof raw.result === 'string' ? raw.result : '')
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
    _id: String(raw._id ?? id),
    dices: validDices ? dices : [],
    sum,
    result
  };
}

// =====================================================
// 3. ALGORITHM ENGINE
// 5 ALGORITHMS - ONLINE LEARNING
// =====================================================

// Khởi tạo trạng thái học
function createModels() {
  return {
    // 1. BAYESIAN ADAPTIVE
    bayes: {
      tai: 1,
      xiu: 1
    },

    // 2. MARKOV CHAIN
    markov: {
      counts: [
        [1, 1],
        [1, 1]
      ]
    },

    // 3. PATTERN RECOGNITION
    pattern: {},

    // 4. LOGISTIC REGRESSION
    logistic: {
      weights: [
        0, 0, 0, 0, 0, 0, 0, 0
      ],
      learningRate: 0.08
    },

    // 5. ADAPTIVE ENSEMBLE
    ensemble: {
      weights: {
        bayes: 1,
        markov: 1,
        pattern: 1,
        logistic: 1
      }
    }
  };
}

let MODELS = createModels();

let lastPredict = null;

let lastId = 0;

const STATS = {
  total: 0,
  win: 0,
  lose: 0
};

const CAUTION = {
  data: []
};

const COMPARE = [];

// -----------------------------------------------------
// UTILITIES
// -----------------------------------------------------

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-clamp(x, -20, 20)));
}

function average(arr) {
  if (!arr.length) return 0.5;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Entropy
function entropy(seq, windowSize = 30) {
  if (seq.length < windowSize) return 0;

  const recent = seq.slice(-windowSize);

  const p1 = average(recent);
  const p0 = 1 - p1;

  if (p1 <= 0 || p0 <= 0) return 0;

  return -(
    p1 * Math.log2(p1) +
    p0 * Math.log2(p0)
  );
}

function chaosBlock(seq) {
  if (seq.length < 30) return false;

  return entropy(seq, 30) > 0.92;
}

// -----------------------------------------------------
// FEATURES FOR LOGISTIC REGRESSION
// -----------------------------------------------------

function extractFeatures(seq) {
  const last5 = seq.slice(-5);
  const last10 = seq.slice(-10);
  const last20 = seq.slice(-20);

  const last = seq.length
    ? seq[seq.length - 1]
    : 0.5;

  // Streak
  let streak = 0;

  if (seq.length) {
    const value = seq[seq.length - 1];

    for (let i = seq.length - 1; i >= 0; i--) {
      if (seq[i] === value) streak++;
      else break;
    }

    streak *= value === 1 ? 1 : -1;
  }

  // Alternation ratio
  let alternations = 0;

  const last6 = seq.slice(-6);

  for (let i = 1; i < last6.length; i++) {
    if (last6[i] !== last6[i - 1]) {
      alternations++;
    }
  }

  const alternationRatio =
    last6.length > 1
      ? alternations / (last6.length - 1)
      : 0.5;

  return [
    1,
    average(last5) - 0.5,
    average(last10) - 0.5,
    average(last20) - 0.5,
    last - 0.5,
    clamp(streak / 5, -1, 1),
    alternationRatio - 0.5,
    average(last5) - average(last20)
  ];
}

// =====================================================
// ALGORITHM 1: BAYESIAN ADAPTIVE
// =====================================================

function predictBayes() {
  const b = MODELS.bayes;

  return b.tai / (b.tai + b.xiu);
}

function updateBayes(result) {
  if (result === 1) {
    MODELS.bayes.tai++;
  } else {
    MODELS.bayes.xiu++;
  }
}

// =====================================================
// ALGORITHM 2: MARKOV CHAIN
// =====================================================

function predictMarkov(seq) {
  if (!seq.length) return 0.5;

  const previous = seq[seq.length - 1];

  const row = MODELS.markov.counts[previous];

  return row[1] / (row[0] + row[1]);
}

function updateMarkov(seq, result) {
  if (!seq.length) return;

  const previous = seq[seq.length - 1];

  MODELS.markov.counts[previous][result]++;
}

// =====================================================
// ALGORITHM 3: PATTERN RECOGNITION
// =====================================================

function predictPattern(seq) {
  if (!seq.length) return 0.5;

  // Ưu tiên chuỗi 3, sau đó 2 và 1
  for (let size = 3; size >= 1; size--) {
    if (seq.length < size) continue;

    const key = seq.slice(-size).join('');

    const record = MODELS.pattern[key];

    if (record && record.total >= 2) {
      return (
        record.tai + 1
      ) / (
        record.tai + record.xiu + 2
      );
    }
  }

  return 0.5;
}

function updatePattern(seq, result) {
  for (let size = 1; size <= 3; size++) {
    if (seq.length < size) continue;

    const key = seq.slice(-size).join('');

    if (!MODELS.pattern[key]) {
      MODELS.pattern[key] = {
        tai: 0,
        xiu: 0,
        total: 0
      };
    }

    const record = MODELS.pattern[key];

    if (result === 1) {
      record.tai++;
    } else {
      record.xiu++;
    }

    record.total++;
  }
}

// =====================================================
// ALGORITHM 4: LOGISTIC REGRESSION
// =====================================================

function predictLogistic(features) {
  const weights = MODELS.logistic.weights;

  let score = 0;

  for (let i = 0; i < features.length; i++) {
    score += weights[i] * features[i];
  }

  return sigmoid(score);
}

function trainLogistic(features, actual) {
  const model = MODELS.logistic;

  const prediction = predictLogistic(features);

  const error = actual - prediction;

  for (let i = 0; i < model.weights.length; i++) {
    const regularization =
      i === 0 ? 0 : 0.001 * model.weights[i];

    model.weights[i] = clamp(
      model.weights[i] +
      model.learningRate * (
        error * features[i] - regularization
      ),
      -5,
      5
    );
  }
}

// =====================================================
// ALGORITHM 5: ADAPTIVE ENSEMBLE
// =====================================================

function updateEnsemble(experts, actual) {
  const weights = MODELS.ensemble.weights;

  for (const name of Object.keys(weights)) {
    const probability = experts[name];

    if (!Number.isFinite(probability)) continue;

    const predicted = probability >= 0.5 ? 1 : 0;

    const correct = predicted === actual;

    // Tăng trọng số nếu đúng, giảm nếu sai
    weights[name] *= correct ? 1.04 : 0.96;

    weights[name] = clamp(
      weights[name],
      0.1,
      5
    );
  }
}

// =====================================================
// 4. FINAL PREDICTION
// =====================================================

function calculateFinal(seq) {
  const features = extractFeatures(seq);

  const experts = {
    bayes: predictBayes(),
    markov: predictMarkov(seq),
    pattern: predictPattern(seq),
    logistic: predictLogistic(features)
  };

  const weights = MODELS.ensemble.weights;

  let weightedProbability = 0;
  let totalWeight = 0;

  for (const name of Object.keys(experts)) {
    const weight = weights[name];

    weightedProbability += experts[name] * weight;
    totalWeight += weight;
  }

  const probability =
    totalWeight > 0
      ? weightedProbability / totalWeight
      : 0.5;

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  const caution = chaosBlock(seq);

  // Độ tự tin mô hình, không phải xác suất thắng đã được bảo đảm
  const confidence = clamp(
    50 + Math.abs(probability - 0.5) * 90,
    50,
    90
  );

  return {
    final,
    probability,
    confidence: Math.round(confidence * 100) / 100,
    caution,
    experts,
    features
  };
}

// =====================================================
// 5. ONLINE LEARNING
// =====================================================

// Học từ kết quả vừa xuất hiện
function learnFromOutcome(session, seqBefore) {
  const actual = session.result;

  // Chấm điểm dự đoán trước
  if (lastPredict) {
    STATS.total++;

    const real = actual === 1 ? 'TÀI' : 'XỈU';

    const win = lastPredict.predict === real;

    if (win) {
      STATS.win++;
    } else {
      STATS.lose++;
    }

    // Học trọng số của từng thuật toán
    updateEnsemble(lastPredict.experts, actual);

    // Huấn luyện Logistic Regression
    trainLogistic(lastPredict.features, actual);

    COMPARE.push({
      phien: session.id,
      predict: lastPredict.predict,
      real,
      win
    });

    if (COMPARE.length > 10) {
      COMPARE.shift();
    }

    // Theo dõi những lần cảnh báo
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);

      if (CAUTION.data.length > 10) {
        CAUTION.data.shift();
      }

      if (CAUTION.data.length === 10) {
        const winCount = CAUTION.data.reduce(
          (a, b) => a + b,
          0
        );

        const loseCount = 10 - winCount;

        if (loseCount >= winCount) {
          console.log(
            '[LEARNING] Caution accuracy is low. Adjusting model weights.'
          );

          // Không xóa lịch sử.
          // Không xóa dữ liệu học đã tích lũy.
        }
      }
    }
  }

  // Cập nhật 4 mô hình dựa trên kết quả thực tế
  updateBayes(actual);
  updateMarkov(seqBefore, actual);
  updatePattern(seqBefore, actual);
}

// Xử lý một phiên mới
function processNewSession(session, history) {
  const seqBefore = history.map(h => h.result);

  // Học từ phiên vừa xuất hiện
  learnFromOutcome(session, seqBefore);

  // Thêm phiên mới vào lịch sử
  history.push(session);

  history.sort((a, b) => a.id - b.id);

  // Giới hạn dung lượng lịch sử
  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  // Dự đoán phiên tiếp theo
  const seq = history.map(h => h.result);

  const prediction = calculateFinal(seq);

  lastPredict = {
    predict: prediction.final,
    experts: prediction.experts,
    features: prediction.features,
    caution: prediction.caution,
    confidence: prediction.confidence
  };

  console.log(
    `[LEARNING] Session ${session.id} | ` +
    `Actual: ${session.result === 1 ? 'TAI' : 'XIU'} | ` +
    `Next: ${prediction.final} | ` +
    `Confidence: ${prediction.confidence}%`
  );

  return prediction;
}

// =====================================================
// 6. RESTORE LEARNING STATE
// =====================================================

function restoreModels(history) {
  let restored = false;

  if (fs.existsSync(MODEL_FILE)) {
    try {
      const saved = JSON.parse(
        fs.readFileSync(MODEL_FILE, 'utf8')
      );

      if (
        saved.version === 2 &&
        saved.models &&
        saved.models.bayes &&
        saved.models.markov &&
        saved.models.logistic &&
        saved.models.ensemble
      ) {
        MODELS = saved.models;

        lastPredict = saved.pending || null;

        Object.assign(STATS, saved.stats || {});

        COMPARE.push(...(saved.compare || []).slice(-10));

        CAUTION.data.push(
          ...(saved.cautionData || []).slice(-10)
        );

        restored = true;

        console.log('[RESTORE] Learning model restored');
      }
    } catch (err) {
      console.error('[RESTORE ERROR]', err.message);
    }
  }

  if (!restored) {
    console.log('[RESTORE] Rebuilding model from history');

    MODELS = createModels();

    lastPredict = null;

    STATS.total = 0;
    STATS.win = 0;
    STATS.lose = 0;

    COMPARE.length = 0;
    CAUTION.data.length = 0;

    // Replay lịch sử theo đúng thứ tự thời gian.
    // Mỗi kết quả chỉ được dùng để cập nhật mô hình
    // sau khi dự đoán của phiên trước đã được chấm điểm.
    const sorted = [...history].sort((a, b) => a.id - b.id);

    const replayHistory = [];

    for (const session of sorted) {
      processNewSession(session, replayHistory);
    }

    console.log(
      `[RESTORE] Replayed ${replayHistory.length} sessions`
    );
  }

  saveModel();
}

// =====================================================
// 7. SYNC
// =====================================================

let syncInProgress = false;

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
      processNewSession(session, history);

      knownIds.add(session.id);
      lastId = Math.max(lastId, session.id);
    }

    // Lưu lịch sử
    writeHistory(history);

    // Lưu trạng thái học
    saveModel();

    console.log(
      `[SYNC] +${fresh.length} sessions | ` +
      `Total: ${history.length}`
    );

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 8. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace('Z', '+07:00');
}

// =====================================================
// 9. API RESPONSE
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

  const seq = history.map(h => h.result);

  const prediction = calculateFinal(seq);

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: current.id + 1,

    dudoan: prediction.final === 'TÀI'
      ? 'tai'
      : 'xiu',

    dotincay: `${Math.round(prediction.confidence)}%`,

    uptime: formatTimeVN()
  };
}

// =====================================================
// 10. API ROUTES
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
// 11. STATISTICS API
// =====================================================

app.get('/stats', (req, res) => {
  const accuracy = STATS.total > 0
    ? (STATS.win / STATS.total) * 100
    : 0;

  res.json({
    total: STATS.total,
    win: STATS.win,
    lose: STATS.lose,
    accuracy: `${accuracy.toFixed(2)}%`,
    algorithms: {
      bayes: MODELS.bayes,
      markov: MODELS.markov,
      patternCount: Object.keys(MODELS.pattern).length,
      logisticWeights: MODELS.logistic.weights,
      ensembleWeights: MODELS.ensemble.weights
    },
    last10: COMPARE,
    uptime: formatTimeVN()
  });
});

// =====================================================
// 12. BOOT SERVER
// =====================================================

initDataFile();

const initialHistory = readHistory();

if (initialHistory.length) {
  lastId = Math.max(
    ...initialHistory.map(h => Number(h.id) || 0)
  );
}

// Khôi phục trạng thái học
restoreModels(initialHistory);

// Đồng bộ lần đầu
syncData();

// Đồng bộ định kỳ
const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

app.listen(PORT, () => {
  console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`);

  console.log(
    '[BOOT] 5 Algorithms: Bayesian + Markov + Pattern + Logistic + Ensemble'
  );

  console.log('[BOOT] Online Learning Enabled');
  console.log('[BOOT] History Persistence Enabled');
});
