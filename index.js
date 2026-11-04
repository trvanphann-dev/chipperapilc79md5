
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
  console.log('[INIT] Da reset lich su');
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

    if (!Array.isArray(list) || !list.length) return;

    const history = readHistory();
    const knownIds = new Set(history.map(h => h.id));

    let added = 0;

    // Sắp xếp phiên tăng dần trước khi xử lý kết quả.
    const normalized = list
      .map(normalizeSession)
      .filter(s =>
        Number.isFinite(s.id) &&
        Number.isFinite(s.sum) &&
        [0, 1].includes(s.result)
      )
      .sort((a, b) => a.id - b.id);

    for (const s of normalized) {
      if (knownIds.has(s.id)) continue;

      settlePrediction(s.id, s.result);

      history.push(s);
      knownIds.add(s.id);
      added++;
    }

    if (added > 0) {
      history.sort((a, b) => a.id - b.id);

      writeHistory(history.slice(-500));

      console.log(
        `[SYNC] +${added} phien, tong ${Math.min(500, history.length)}`
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

    performance: {
      count: 0,
      modelLoss: 0,
      baselineLoss: 0
    },

    updates: 0
  };
}

// RESET TOAN BO MO HINH MOI LAN KHOI DONG
function initModelFile() {
  fs.writeFileSync(
    MODEL_FILE,
    JSON.stringify(defaultModel(), null, 2),
    'utf8'
  );

  console.log('[INIT] Da reset toan bo trong so hoc');
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

    if (!model.predictions) {
      model.predictions = {};
    }

    if (
      !Array.isArray(model.calibration) ||
      model.calibration.length !== 10
    ) {
      model.calibration = defaultModel().calibration;
    }

    if (
      !model.performance ||
      !Number.isFinite(model.performance.count) ||
      !Number.isFinite(model.performance.modelLoss) ||
      !Number.isFinite(model.performance.baselineLoss)
    ) {
      model.performance = defaultModel().performance;
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

// Khong ep chon Tai hoac Xiu khi xac suat bang nhau.
function makePrediction(pTai) {
  const p = clamp(pTai, 0.001, 0.999);

  let pick = null;

  if (p > 0.500001) {
    pick = 1;
  } else if (p < 0.499999) {
    pick = 0;
  }

  return {
    pick,
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
        transitionTai + 0.5 * 20
      ) / (
        transitionTotal + 20
      )
    : 0.5;

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

// ============ 6. ALGORITHM 1: BAYESIAN ============

function algoBayesian(seq, features) {
  const n = seq.length;

  if (!n) return makePrediction(0.5);

  const tai = seq.reduce((sum, value) => sum + value, 0);
  const xiu = n - tai;

  // Prior doi xung: 20 Tai va 20 Xiu.
  const alpha = 20;
  const beta = 20;

  const pTai = (tai + alpha) /
    (tai + xiu + alpha + beta);

  return makePrediction(pTai);
}

// ============ 7. ALGORITHM 2: MARKOV ============

function algoMarkov(seq, features) {
  if (seq.length < 2) {
    return makePrediction(0.5);
  }

  const current = seq[seq.length - 1];

  let tai = 0;
  let total = 0;

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] !== current) continue;

    total++;

    if (seq[i] === 1) tai++;
  }

  // Prior trung lap, khong lay globalRate lam huong mac dinh.
  const priorWeight = 20;

  const pTai = (
    tai + 0.5 * priorWeight
  ) / (
    total + priorWeight
  );

  return makePrediction(pTai);
}

// ============ 8. ALGORITHM 3: REGIME ADAPTIVE ============

function algoRegime(seq, features) {
  const n = seq.length;

  if (n < 20) {
    return makePrediction(0.5);
  }

  const recentSize = Math.min(20, n);
  const recent = seq.slice(-recentSize);

  const olderEnd = n - recentSize;
  const olderStart = Math.max(0, olderEnd - 100);
  const older = seq.slice(olderStart, olderEnd);

  if (!older.length) {
    return makePrediction(0.5);
  }

  const recentRate = getRate(recent);
  const olderRate = getRate(older);

  // Chi su dung do lech gan day so voi giai doan truoc.
  const difference = recentRate - olderRate;

  const reliability = recent.length /
    (recent.length + 30);

  const pTai = clamp(
    0.5 + difference * reliability * 0.5,
    0.05,
    0.95
  );

  return makePrediction(pTai);
}

// ============ 9. ALGORITHM 4: EWMA ADAPTIVE ============

function algoEWMA(seq, features) {
  if (seq.length < 5) {
    return makePrediction(0.5);
  }

  let average = features.globalRate;

  const alpha = 0.12;

  for (const value of seq) {
    average =
      alpha * value +
      (1 - alpha) * average;
  }

  // Chi tinh xu huong EWMA lech khoi trung binh.
  const reliability = seq.length /
    (seq.length + 50);

  const pTai = clamp(
    0.5 + (average - features.globalRate) * reliability,
    0.05,
    0.95
  );

  return makePrediction(pTai);
}

// ============ 10. ALGORITHM 5: ONLINE LOGISTIC ============

function algoOnlineLogistic(seq, features, model) {
  if (seq.length < 10) {
    return makePrediction(0.5);
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

function getCalibrationBucket(pTai) {
  const selectedProbability = Math.max(pTai, 1 - pTai);

  return clamp(
    Math.floor((selectedProbability - 0.5) / 0.05),
    0,
    9
  );
}

// Dung can duoi Wilson 95%.
// Khong du du lieu thi khong tu nang do tin cay.
function getCalibratedConfidence(pTai, model) {
  const bucket = getCalibrationBucket(pTai);

  const data = model.calibration[bucket] || {
    total: 0,
    correct: 0
  };

  const n = data.total;
  const correct = data.correct;

  if (n < 30) {
    return 50;
  }

  const z = 1.96;
  const z2 = z * z;

  const observed = correct / n;

  const denominator = 1 + z2 / n;

  const center = observed + z2 / (2 * n);

  const margin = z * Math.sqrt(
    (
      observed * (1 - observed) +
      z2 / (4 * n)
    ) / n
  );

  const lowerBound = (center - margin) / denominator;

  return Math.round(
    clamp(lowerBound * 100, 50, 99.9) * 100
  ) / 100;
}

// Danh gia xem mo hinh co thuc su tot hon moc 50/50 hay khong.
function getPredictionReliability(model) {
  const performance = model.performance;

  const count = performance.count || 0;

  // Chua du mau thi khong cho phep mo hinh nghieng manh.
  if (count < 30) return 0;

  const modelAverage = performance.modelLoss / count;
  const baselineAverage = performance.baselineLoss / count;

  const improvement = baselineAverage - modelAverage;

  // Neu mo hinh khong tot hon moc 50/50 thi trung lap.
  if (improvement <= 0) return 0;

  const evidence = clamp(improvement / 0.03, 0, 1);

  const sampleReliability = count / (count + 100);

  return clamp(
    evidence * sampleReliability,
    0,
    1
  );
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

  // Tinh Log Loss cua tung mo hinh.
  const losses = pending.probabilities.map(p => {
    return logLoss(p, actual);
  });

  const averageLoss = losses.reduce(
    (sum, value) => sum + value,
    0
  ) / losses.length;

  // Mo hinh tot hon trung binh duoc tang trong so.
  // Mo hinh kem hon bi giam trong so.
  const learningRate = 0.35;

  for (let i = 0; i < model.weights.length; i++) {
    const improvement = averageLoss - losses[i];

    model.weights[i] *= Math.exp(
      learningRate * improvement
    );

    model.weights[i] = clamp(
      model.weights[i],
      0.2,
      5
    );
  }

  model.weights = normalizeWeights(model.weights);

  // Cap nhat Logistic bang gradient descent.
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

  // Danh gia ensemble so voi moc du doan 50/50.
  const ensembleProbability = clamp(
    Number(pending.ensembleProbability ?? 0.5),
    0.001,
    0.999
  );

  model.performance.count++;

  model.performance.modelLoss += logLoss(
    ensembleProbability,
    actual
  );

  model.performance.baselineLoss += logLoss(
    0.5,
    actual
  );

  // Hieu chinh do tin cay.
  if (
    Number.isInteger(pending.calibrationBucket) &&
    (pending.pick === 0 || pending.pick === 1)
  ) {
    const bucket = pending.calibrationBucket;

    if (!model.calibration[bucket]) {
      model.calibration[bucket] = {
        total: 0,
        correct: 0
      };
    }

    model.calibration[bucket].total++;

    if (pending.pick === actual) {
      model.calibration[bucket].correct++;
    }
  }

  model.updates = (model.updates || 0) + 1;

  delete model.pending[key];

  writeModel(model);

  console.log(
    `[LEARN] Phien ${sessionId} | Ket qua ${actual === 1 ? 'TAI' : 'XIU'} | Updates ${model.updates} | Weights ${model.weights.map(x => x.toFixed(2)).join(',')}`
  );
}

// ============ 13. FROZEN PREDICTION ============

function getFrozenPrediction(seq, targetSessionId) {
  const model = readModel();

  const key = String(targetSessionId);

  // Neu da du doan phien nay thi tra lai ket qua cu.
  if (model.predictions[key]) {
    return model.predictions[key];
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

  // Xac suat goc cua ensemble.
  const rawPTai = clamp(
    weightedTai / totalWeight,
    0.001,
    0.999
  );

  // Chi cho phep nghieng khi mo hinh da chung minh co loi the.
  const reliability = getPredictionReliability(model);

  // Neu mo hinh chua tot hon moc 50/50 thi keo ve trung lap.
  const pTai = clamp(
    0.5 + (rawPTai - 0.5) * reliability,
    0.001,
    0.999
  );

  const prediction = makePrediction(pTai);

  const confidence = prediction.pick === null
    ? 50
    : getCalibratedConfidence(pTai, model);

  const frozen = {
    pick: prediction.pick,
    pTai: Math.round(pTai * 10000) / 10000,
    rawPTai: Math.round(rawPTai * 10000) / 10000,
    confidence,
    reliability: Math.round(reliability * 10000) / 10000,
    weights: weights.map(w =>
      Math.round(w * 1000) / 1000
    ),
    updates: model.updates
  };

  model.predictions[key] = frozen;

  // Luu du lieu de hoc sau khi phien co ket qua.
  model.pending[key] = {
    probabilities: predictions.map(p => p.pTai),
    features: result.features.logistic,
    ensembleProbability: rawPTai,
    pick: prediction.pick,
    calibrationBucket: getCalibrationBucket(pTai),
    createdAt: Date.now()
  };

  // Gioi han so phien luu.
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
      id: 'noname09',
      error: 'Chua co du lieu, cho sync',
      uptime: formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  const seq = history.map(h => h.result);

  const nextSessionId = current.id + 1;

  const ens = getFrozenPrediction(
    seq,
    nextSessionId
  );

  let predictionText = 'can_bang';

  if (ens.pick === 1) {
    predictionText = 'tai';
  } else if (ens.pick === 0) {
    predictionText = 'xiu';
  }

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,
    dudoanphien: nextSessionId,
    dudoan: predictionText,
    dotincay: `${ens.confidence}%`,
    xacsuatTai: `${(ens.pTai * 100).toFixed(2)}%`,
    xacsuatXiu: `${((1 - ens.pTai) * 100).toFixed(2)}%`,
    do_tin_hieu: ens.reliability,
    trongso: ens.weights,
    solan_hoc: ens.updates,
    uptime: formatTimeVN()
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
