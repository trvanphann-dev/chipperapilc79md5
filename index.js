
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const MODEL_FILE = path.join(__dirname, 'model_taixiu.json');
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

      // Chỉ học khi đã có dự đoán được lưu trước đó.
      settlePrediction(s.id, s.result);

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

// ============ 3. MODEL STORAGE ============

function defaultModel() {
  return {
    weights: [1, 1, 1, 1, 1],
    logistic: [0, 0, 0, 0, 0],
    pending: {},
    predictions: {},
    calibration: Array.from({ length: 10 }, () => ({
      total: 0,
      correct: 0
    })),
    updates: 0
  };
}

// RESET TOÀN BỘ MÔ HÌNH MỖI LẦN KHỞI ĐỘNG
function initModelFile() {
  fs.writeFileSync(
    MODEL_FILE,
    JSON.stringify(defaultModel(), null, 2),
    'utf8'
  );

  console.log('[INIT] Đã reset toàn bộ trọng số và dữ liệu học');
}

function readModel() {
  try {
    const model = JSON.parse(
      fs.readFileSync(MODEL_FILE, 'utf8')
    );

    if (
      !Array.isArray(model.weights) ||
      model.weights.length !== 5 ||
      !Array.isArray(model.logistic) ||
      model.logistic.length !== 5 ||
      !model.pending ||
      typeof model.pending !== 'object'
    ) {
      return defaultModel();
    }

    // Bổ sung các trường cho model phiên bản cũ.
    if (!model.predictions) model.predictions = {};

    if (
      !Array.isArray(model.calibration) ||
      model.calibration.length !== 10
    ) {
      model.calibration = defaultModel().calibration;
    }

    if (!Number.isFinite(model.updates)) {
      model.updates = 0;
    }

    return model;
  } catch {
    return defaultModel();
  }
}

function writeModel(model) {
  fs.writeFileSync(
    MODEL_FILE,
    JSON.stringify(model, null, 2),
    'utf8'
  );
}

// ============ 4. STATISTICAL FUNCTIONS ============

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getRate(seq) {
  if (!seq.length) return 0.5;

  const valid = seq.filter(v => v === 0 || v === 1);

  if (!valid.length) return 0.5;

  return valid.reduce((sum, value) => sum + value, 0) /
    valid.length;
}

function sigmoid(value) {
  const x = clamp(value, -20, 20);
  return 1 / (1 + Math.exp(-x));
}

function logLoss(probability, actual) {
  const p = clamp(probability, 0.001, 0.999);

  return -(
    actual * Math.log(p) +
    (1 - actual) * Math.log(1 - p)
  );
}

function makePrediction(pTai) {
  const p = clamp(pTai, 0.001, 0.999);

  return {
    pick: p > 0.5 ? 1 : 0,
    pTai: p
  };
}

// ============ 5. SHARED FEATURES ============

function calculateFeatures(seq) {
  const n = seq.length;

  const globalRate = getRate(seq);

  const recentSize = Math.min(20, n);
  const recent = seq.slice(-recentSize);
  const recentRate = getRate(recent);

  const alpha = 0.15;
  let ewma = globalRate;

  for (const value of seq) {
    ewma = alpha * value + (1 - alpha) * ewma;
  }

  const current = seq[n - 1];

  let transitionTai = 0;
  let transitionTotal = 0;

  for (let i = 1; i < n; i++) {
    if (seq[i - 1] !== current) continue;

    transitionTotal++;

    if (seq[i] === 1) {
      transitionTai++;
    }
  }

  const markovRate = transitionTotal > 0
    ? (
        transitionTai + globalRate * 10
      ) / (
        transitionTotal + 10
      )
    : globalRate;

  return {
    globalRate,
    recentRate,
    ewma,
    markovRate,
    logistic: [
      1,
      globalRate - 0.5,
      recentRate - globalRate,
      ewma - globalRate,
      markovRate - globalRate
    ]
  };
}

// ============ 6. ALGORITHM 1: BAYESIAN ADAPTIVE ============

function algoBayesian(seq, features) {
  const n = seq.length;

  if (!n) return makePrediction(0.5);

  const tai = seq.reduce((sum, value) => sum + value, 0);
  const xiu = n - tai;

  const alpha = 2;
  const beta = 2;

  const pTai = (tai + alpha) /
    (tai + xiu + alpha + beta);

  return makePrediction(pTai);
}

// ============ 7. ALGORITHM 2: MARKOV ============

function algoMarkov(seq, features) {
  if (seq.length < 2) {
    return makePrediction(features.globalRate);
  }

  const current = seq[seq.length - 1];

  let tai = 0;
  let total = 0;

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] !== current) continue;

    total++;

    if (seq[i] === 1) tai++;
  }

  const priorWeight = 12;

  const pTai = (
    tai + features.globalRate * priorWeight
  ) / (
    total + priorWeight
  );

  return makePrediction(pTai);
}

// ============ 8. ALGORITHM 3: REGIME ADAPTIVE ============

function algoRegime(seq, features) {
  const n = seq.length;

  if (n < 10) {
    return makePrediction(features.globalRate);
  }

  const recentSize = Math.min(20, n);
  const recent = seq.slice(-recentSize);

  const olderEnd = n - recentSize;
  const olderStart = Math.max(0, olderEnd - 100);
  const older = seq.slice(olderStart, olderEnd);

  const recentRate = getRate(recent);

  const olderRate = older.length
    ? getRate(older)
    : features.globalRate;

  const difference = recentRate - olderRate;

  const reliability = recent.length /
    (recent.length + 20);

  const adjustment = difference * reliability * 0.5;

  const pTai = clamp(
    features.globalRate + adjustment,
    0.05,
    0.95
  );

  return makePrediction(pTai);
}

// ============ 9. ALGORITHM 4: EWMA ADAPTIVE ============

function algoEWMA(seq, features) {
  if (seq.length < 3) {
    return makePrediction(features.globalRate);
  }

  let average = features.globalRate;

  const alpha = 0.12;

  for (const value of seq) {
    average =
      alpha * value +
      (1 - alpha) * average;
  }

  const pTai =
    features.globalRate * 0.65 +
    average * 0.35;

  return makePrediction(pTai);
}

// ============ 10. ALGORITHM 5: ONLINE LOGISTIC ============

function algoOnlineLogistic(seq, features, model) {
  if (seq.length < 10) {
    return makePrediction(features.globalRate);
  }

  const coefficients = model.logistic;

  let score = 0;

  for (let i = 0; i < coefficients.length; i++) {
    score += coefficients[i] * features.logistic[i];
  }

  return makePrediction(sigmoid(score));
}

// ============ 11. FIVE ALGORITHMS ============

const ALGOS = [
  { name: 'bayesian', fn: algoBayesian },
  { name: 'markov', fn: algoMarkov },
  { name: 'regime_adaptive', fn: algoRegime },
  { name: 'ewma_adaptive', fn: algoEWMA },
  { name: 'online_logistic', fn: algoOnlineLogistic }
];

function runAlgorithms(seq, model) {
  const features = calculateFeatures(seq);

  const predictions = ALGOS.map(algo => {
    return algo.fn(seq, features, model);
  });

  return {
    features,
    predictions
  };
}

// ============ 12. ADAPTIVE WEIGHT LEARNING ============

function normalizeWeights(weights) {
  const safe = weights.map(w => clamp(
    Number.isFinite(w) ? w : 1,
    0.2,
    5
  ));

  const total = safe.reduce((sum, value) => sum + value, 0);

  if (total <= 0) return [1, 1, 1, 1, 1];

  return safe.map(w => w * 5 / total);
}

// Chọn nhóm xác suất để hiệu chỉnh độ tin cậy.
function getCalibrationBucket(pTai) {
  const selectedProbability = Math.max(pTai, 1 - pTai);

  return clamp(
    Math.floor((selectedProbability - 0.5) / 0.05),
    0,
    9
  );
}

// Ước lượng độ chính xác từ các dự đoán đã có kết quả.
// Prior 20 mẫu giả định ở mức 50% để tránh phóng đại khi ít dữ liệu.
function getCalibratedConfidence(pTai, model) {
  const bucket = getCalibrationBucket(pTai);

  const data = model.calibration[bucket] || {
    total: 0,
    correct: 0
  };

  const priorTotal = 20;
  const priorCorrect = 10;

  const confidence = (
    data.correct + priorCorrect
  ) / (
    data.total + priorTotal
  );

  return Math.round(
    clamp(confidence * 100, 0, 100) * 100
  ) / 100;
}

function settlePrediction(sessionId, actual) {
  const model = readModel();

  const key = String(sessionId);
  const pending = model.pending[key];

  if (!pending) return;

  if (
    !Array.isArray(pending.probabilities) ||
    pending.probabilities.length !== 5
  ) {
    delete model.pending[key];
    writeModel(model);
    return;
  }

  const losses = pending.probabilities.map(p => {
    return logLoss(p, actual);
  });

  const averageLoss = losses.reduce(
    (sum, value) => sum + value,
    0
  ) / losses.length;

  const learningRate = 0.35;

  for (let i = 0; i < model.weights.length; i++) {
    const improvement = averageLoss - losses[i];

    model.weights[i] *= Math.exp(
      learningRate * improvement
    );
  }

  model.weights = normalizeWeights(model.weights);

  // Cập nhật Logistic bằng gradient descent.
  if (
    Array.isArray(pending.features) &&
    pending.features.length === 5
  ) {
    const coefficients = model.logistic;

    const oldScore = coefficients.reduce(
      (sum, coefficient, i) =>
        sum + coefficient * pending.features[i],
      0
    );

    const predicted = sigmoid(oldScore);
    const error = actual - predicted;

    const logisticRate = 0.04;

    for (let i = 0; i < coefficients.length; i++) {
      coefficients[i] +=
        logisticRate * error * pending.features[i];

      coefficients[i] = clamp(
        coefficients[i],
        -5,
        5
      );
    }
  }

  // Cập nhật độ chính xác thực tế theo nhóm xác suất.
  if (Number.isInteger(pending.calibrationBucket)) {
    const bucket = pending.calibrationBucket;

    if (!model.calibration[bucket]) {
      model.calibration[bucket] = {
        total: 0,
        correct: 0
      };
    }

    const predictedPick = pending.pick;

    model.calibration[bucket].total++;

    if (predictedPick === actual) {
      model.calibration[bucket].correct++;
    }
  }

  model.updates = (model.updates || 0) + 1;

  delete model.pending[key];

  writeModel(model);

  console.log(
    `[LEARN] Phiên ${sessionId} | Kết quả ${actual === 1 ? 'TAI' : 'XIU'} | Updates ${model.updates}`
  );
}

// ============ 13. FROZEN PREDICTION ============

function getFrozenPrediction(seq, targetSessionId) {
  const model = readModel();

  const key = String(targetSessionId);

  // Nếu đã dự đoán phiên này thì trả đúng kết quả cũ.
  if (model.predictions[key]) {
    return model.predictions[key];
  }

  if (seq.length < 10) {
    const result = {
      pick: 0,
      pTai: 0.5,
      confidence: 50,
      weights: normalizeWeights(model.weights),
      updates: model.updates
    };

    model.predictions[key] = result;

    writeModel(model);

    return result;
  }

  const result = runAlgorithms(seq, model);

  const predictions = result.predictions;
  const weights = normalizeWeights(model.weights);

  let weightedTai = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    weightedTai += predictions[i].pTai * weights[i];
    totalWeight += weights[i];
  }

  const pTai = clamp(
    weightedTai / totalWeight,
    0.001,
    0.999
  );

  const pick = pTai > 0.5 ? 1 : 0;

  // Độ tin cậy được tính từ hiệu chỉnh các dự đoán đã hoàn thành.
  const confidence = getCalibratedConfidence(
    pTai,
    model
  );

  const frozen = {
    pick,
    pTai: Math.round(pTai * 10000) / 10000,
    confidence,
    weights: weights.map(w =>
      Math.round(w * 1000) / 1000
    ),
    updates: model.updates
  };

  // Lưu dự đoán và thông tin học trước khi trả kết quả.
  model.predictions[key] = frozen;

  model.pending[key] = {
    probabilities: predictions.map(p => p.pTai),
    features: result.features.logistic,
    ensembleProbability: pTai,
    pick,
    calibrationBucket: getCalibrationBucket(pTai),
    createdAt: Date.now()
  };

  // Giới hạn số phiên lưu để file không tăng vô hạn.
  const predictionKeys = Object.keys(model.predictions);

  if (predictionKeys.length > 100) {
    predictionKeys
      .sort((a, b) => Number(a) - Number(b))
      .slice(0, predictionKeys.length - 100)
      .forEach(oldKey => {
        delete model.predictions[oldKey];
      });
  }

  const pendingKeys = Object.keys(model.pending);

  if (pendingKeys.length > 100) {
    pendingKeys
      .sort((a, b) => Number(a) - Number(b))
      .slice(0, pendingKeys.length - 100)
      .forEach(oldKey => {
        delete model.pending[oldKey];
      });
  }

  writeModel(model);

  return frozen;
}

// ============ 14. TIME VN ============

function formatTimeVN() {
  const now = new Date(Date.now() + 7 * 3600 * 1000);

  return now.toISOString().replace('Z', '+07:00');
}

// ============ 15. RESPONSE ============

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

  const nextSessionId = current.id + 1;

  // Luôn lấy dự đoán đã khóa cho phiên này nếu tồn tại.
  const ens = getFrozenPrediction(
    seq,
    nextSessionId
  );

  return {
    "id": "noname09",
    "phienhientai": current.id,
    "ketqua": current.result === 1 ? "tai" : "xiu",
    "xucxac": current.dices,
    "tong": current.sum,
    "dudoanphien": nextSessionId,
    "dudoan": ens.pick === 1 ? "tai" : "xiu",
    "dotincay": `${ens.confidence}%`,
    "uptime": formatTimeVN()
  };
}

// ============ 16. ROUTES ============

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

// ============ 17. BOOT ============

initDataFile();
initModelFile();

syncData();

setInterval(syncData, POLL_INTERVAL);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`);
});
