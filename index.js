
'use strict';

const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '1mb' }));

// =====================================================
// 1. CONFIG
// =====================================================

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const LEARNING_FILE = path.join(__dirname, 'learning_state.json');
const DEPLOY_MARKER = path.join(__dirname, 'deploy_marker.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const PORT = process.env.PORT || 3000;
const POLL_INTERVAL = 15000;

const MAX_HISTORY = 500;
const MAX_CONTEXT = 100;
const MAX_PROCESSED_IDS = 3000;

// Render cung cấp mã commit cho mỗi phiên bản deploy.
const DEPLOY_ID =
  process.env.RENDER_GIT_COMMIT ||
  process.env.DEPLOY_ID ||
  'local';

// =====================================================
// 2. LEARNING CONFIG
// =====================================================

const CFG = {
  bayesianPrior: 15,
  bayesianDecay: 0.995,

  markovPrior: 8,
  markovDecay: 0.99,

  // Số mẫu tối thiểu trước khi tin tưởng pattern.
  patternMinSamples: 4,

  // Hệ số giới hạn ảnh hưởng của Markov.
  markovInfluence: 0.30,

  logisticRate: 0.004,
  logisticL2: 0.002,

  neuralRate: 0.0025,

  ewmaAlpha: 0.08,

  metaRate: 0.025,
  minWeight: 0.15,
  maxWeight: 5,

  minProbability: 0.10,
  maxProbability: 0.90,

  // Phát hiện thay đổi trạng thái.
  changeWindow: 12,
  changeThreshold: 0.22
};

const ALGO_NAMES = [
  'bayesian_online',
  'pattern_markov',
  'online_logistic',
  'adaptive_neural_network',
  'ewma_online'
];

// =====================================================
// 3. COMMON
// =====================================================

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-clamp(x, -25, 25)));
}

function atomicSave(file, data) {
  const temp = file + '.tmp';

  fs.writeFileSync(
    temp,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(temp, file);
}

function removeFile(file) {
  for (const target of [file, file + '.tmp']) {
    if (fs.existsSync(target)) {
      fs.unlinkSync(target);
    }
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

// =====================================================
// 4. DEPLOY RESET
// =====================================================

// Chỉ reset khi nhận diện commit mới.
// Restart cùng commit không xóa dữ liệu.
//
// Lần đầu chạy: giữ lại file cũ nếu có.
//
// Nếu muốn reset thủ công, xóa file dữ liệu và marker
// trong môi trường lưu trữ có thể truy cập được.

function handleDeploy() {
  let marker = null;

  try {
    if (fs.existsSync(DEPLOY_MARKER)) {
      marker = JSON.parse(
        fs.readFileSync(DEPLOY_MARKER, 'utf8')
      );
    }
  } catch (err) {
    console.error('[DEPLOY]', err.message);
  }

  if (
    marker &&
    marker.deployId &&
    DEPLOY_ID !== 'local' &&
    marker.deployId !== DEPLOY_ID
  ) {
    console.log('[DEPLOY] New commit detected');
    console.log('[DEPLOY] Resetting data and weights');

    removeFile(DATA_FILE);
    removeFile(LEARNING_FILE);
  } else {
    console.log('[DEPLOY] Keeping existing data');
  }

  atomicSave(DEPLOY_MARKER, {
    deployId: DEPLOY_ID,
    updatedAt: new Date().toISOString()
  });
}

// =====================================================
// 5. HISTORY STORAGE
// =====================================================

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    atomicSave(DATA_FILE, []);
  }
}

function readHistory() {
  try {
    const data = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    return Array.isArray(data) ? data : [];
  } catch (err) {
    console.error('[HISTORY READ]', err.message);
    return [];
  }
}

function writeHistory(data) {
  atomicSave(DATA_FILE, data);
}

function mergeHistory(oldHistory, newSessions) {
  const map = new Map();

  for (const item of oldHistory) {
    const id = Number(item.id);
    if (Number.isFinite(id)) map.set(id, item);
  }

  for (const item of newSessions) {
    map.set(Number(item.id), item);
  }

  return [...map.values()]
    .sort((a, b) => Number(a.id) - Number(b.id))
    .slice(-MAX_HISTORY);
}

// =====================================================
// 6. FETCH AND NORMALIZE
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

  return res.json();
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
    ? rawDices.map(Number).filter(Number.isFinite)
    : [];

  const diceSum = dices.length
    ? dices.reduce((a, b) => a + b, 0)
    : 0;

  const sum = diceSum || num(raw.sum, 0);

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    (Array.isArray(raw.result) ? '' : raw.result ?? '')
  ).trim().toUpperCase();

  let result;

  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) {
    result = 1;
  } else if (['XIU', 'XỈU', 'SMALL'].includes(sourceResult)) {
    result = 0;
  } else {
    result = sum >= 11 ? 1 : 0;
  }

  return {
    id,
    dices,
    sum,
    result
  };
}

// =====================================================
// 7. LEARNING STATE
// =====================================================

function createLearningState() {
  return {
    version: 4,

    initialized: false,
    lastProcessedId: null,

    bayesian: {
      alpha: CFG.bayesianPrior,
      beta: CFG.bayesianPrior
    },

    markov: {
      // transitions[previous][next]
      transitions: [
        [CFG.markovPrior, CFG.markovPrior],
        [CFG.markovPrior, CFG.markovPrior]
      ]
    },

    logistic: {
      weights: [0, 0, 0, 0, 0]
    },

    neural: {
      hidden: [
        [0.04, -0.03, 0.02, 0.03, -0.02],
        [-0.03, 0.04, -0.02, 0.02, 0.03],
        [0.02, 0.03, -0.04, -0.02, 0.04],
        [-0.02, 0.02, 0.03, 0.04, -0.03]
      ],

      output: [0.03, -0.02, 0.02, 0.01],
      bias: 0
    },

    ewma: {
      value: 0.5
    },

    metaWeights: [1, 1, 1, 1, 1],

    recent: [],

    processedIds: [],
    pendingSessions: [],

    stats: {
      total: 0,
      hits: 0,
      lastAccuracy: 0,
      logLoss: 0,
      recentAccuracy: 0,
      recentLogLoss: 0
    },

    recentPredictions: []
  };
}

// =====================================================
// 8. LOAD STATE AND MIGRATE OLD VERSION
// =====================================================

function loadLearningState() {
  if (!fs.existsSync(LEARNING_FILE)) {
    return createLearningState();
  }

  try {
    const old = JSON.parse(
      fs.readFileSync(LEARNING_FILE, 'utf8')
    );

    const state = createLearningState();

    if (old.bayesian) {
      state.bayesian = {
        ...state.bayesian,
        ...old.bayesian
      };
    }

    if (
      Array.isArray(old.logistic?.weights) &&
      old.logistic.weights.length >= 4
    ) {
      state.logistic.weights =
        old.logistic.weights.slice(0, 5);

      while (state.logistic.weights.length < 5) {
        state.logistic.weights.push(0);
      }
    }

    if (old.ewma) {
      state.ewma = {
        ...state.ewma,
        ...old.ewma
      };
    }

    if (
      Array.isArray(old.neural?.hidden) &&
      old.neural.hidden.length === 4 &&
      Array.isArray(old.neural?.output) &&
      old.neural.output.length === 4
    ) {
      state.neural = old.neural;

      state.neural.hidden =
        state.neural.hidden.map(row => {
          const result = row.slice(0, 5);

          while (result.length < 5) result.push(0);

          return result;
        });
    }

    if (
      Array.isArray(old.markov?.transitions) &&
      old.markov.transitions.length === 2
    ) {
      state.markov = old.markov;
    }

    if (
      Array.isArray(old.metaWeights) &&
      old.metaWeights.length >= 5
    ) {
      state.metaWeights = old.metaWeights
        .slice(0, 5)
        .map(v => clamp(num(v, 1), CFG.minWeight, CFG.maxWeight));
    }

    if (Array.isArray(old.recent)) {
      state.recent = old.recent
        .filter(v => v === 0 || v === 1)
        .slice(-MAX_CONTEXT);
    }

    if (Array.isArray(old.processedIds)) {
      state.processedIds =
        old.processedIds.slice(-MAX_PROCESSED_IDS);
    }

    if (Array.isArray(old.pendingSessions)) {
      state.pendingSessions = old.pendingSessions;
    }

    if (old.stats) {
      state.stats = {
        ...state.stats,
        ...old.stats
      };
    }

    if (Array.isArray(old.recentPredictions)) {
      state.recentPredictions =
        old.recentPredictions.slice(-100);
    }

    state.initialized = Boolean(old.initialized);
    state.lastProcessedId = old.lastProcessedId ?? null;
    state.version = 4;

    return state;

  } catch (err) {
    console.error('[STATE LOAD]', err.message);

    // Không tự xóa file lỗi.
    // Giữ file gốc để tránh mất dữ liệu.
    return createLearningState();
  }
}

let learningState = null;

function saveLearningState() {
  atomicSave(LEARNING_FILE, learningState);
}

// =====================================================
// 9. PATTERN DETECTION
// =====================================================

function getPatternInfo(recent) {
  if (!recent.length) {
    return {
      type: 'insufficient_data',
      streak: 0,
      last: null,
      alternating: false
    };
  }

  const last = recent[recent.length - 1];

  let streak = 0;

  for (let i = recent.length - 1; i >= 0; i--) {
    if (recent[i] !== last) break;
    streak++;
  }

  const checkLength = Math.min(6, recent.length);
  let alternating = checkLength >= 4;

  if (alternating) {
    const start = recent.length - checkLength;

    for (let i = start + 1; i < recent.length; i++) {
      if (recent[i] === recent[i - 1]) {
        alternating = false;
        break;
      }
    }
  }

  let type = 'mixed';

  if (streak >= 3) {
    type = 'streak';
  } else if (alternating) {
    type = 'alternating';
  } else if (recent.length < 5) {
    type = 'insufficient_data';
  }

  return {
    type,
    streak,
    last,
    alternating
  };
}

// Tìm các lần xuất hiện của mẫu cuối chuỗi.
// Ví dụ: Tài-Xỉu-Tài hoặc Tài-Tài-Tài.
// Chỉ sử dụng kết quả đã có trước thời điểm dự đoán.

function getNgramProbability(recent) {
  const maxLength = Math.min(4, recent.length);

  if (maxLength < 2) {
    return {
      pTai: 0.5,
      samples: 0,
      patternLength: 0
    };
  }

  for (let length = maxLength; length >= 2; length--) {
    const pattern = recent.slice(-length);

    let tai = 0;
    let xiu = 0;

    for (
      let i = 0;
      i + length < recent.length;
      i++
    ) {
      let matched = true;

      for (let j = 0; j < length; j++) {
        if (recent[i + j] !== pattern[j]) {
          matched = false;
          break;
        }
      }

      if (matched) {
        const next = recent[i + length];

        if (next === 1) tai++;
        else xiu++;
      }
    }

    const samples = tai + xiu;

    if (samples >= CFG.patternMinSamples) {
      const pTai =
        (tai + CFG.markovPrior * 0.5) /
        (samples + CFG.markovPrior);

      return {
        pTai,
        samples,
        patternLength: length
      };
    }
  }

  return {
    pTai: 0.5,
    samples: 0,
    patternLength: 0
  };
}

// =====================================================
// 10. CHANGE DETECTION
// =====================================================

function detectRegimeChange(recent) {
  const window = CFG.changeWindow;

  if (recent.length < window * 2) {
    return {
      changed: false,
      difference: 0
    };
  }

  const oldPart = recent.slice(-window * 2, -window);
  const newPart = recent.slice(-window);

  const oldMean =
    oldPart.reduce((a, b) => a + b, 0) / oldPart.length;

  const newMean =
    newPart.reduce((a, b) => a + b, 0) / newPart.length;

  const difference = Math.abs(newMean - oldMean);

  return {
    changed: difference >= CFG.changeThreshold,
    difference
  };
}

// =====================================================
// 11. FEATURE ENGINEERING
// =====================================================

function getFeatures(state) {
  const recent = state.recent;

  const last = recent.length
    ? recent[recent.length - 1]
    : 0.5;

  const mean = recent.length
    ? recent.reduce((a, b) => a + b, 0) / recent.length
    : 0.5;

  let changes = 0;

  for (let i = 1; i < recent.length; i++) {
    if (recent[i] !== recent[i - 1]) changes++;
  }

  const changeRate = recent.length > 1
    ? changes / (recent.length - 1)
    : 0.5;

  const pattern = getPatternInfo(recent);

  const signedStreak =
    pattern.streak *
    (last === 1 ? 1 : -1);

  return [
    1,
    last * 2 - 1,
    clamp((mean - 0.5) * 2, -1, 1),
    clamp((changeRate - 0.5) * 2, -1, 1),
    clamp(signedStreak / 8, -1, 1)
  ];
}

// =====================================================
// 12. MODEL 1: BAYESIAN
// =====================================================

function algoBayesian(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  return a + b > 0 ? a / (a + b) : 0.5;
}

function updateBayesian(state, actual) {
  const prior = CFG.bayesianPrior;

  state.bayesian.alpha =
    prior +
    (state.bayesian.alpha - prior) * CFG.bayesianDecay;

  state.bayesian.beta =
    prior +
    (state.bayesian.beta - prior) * CFG.bayesianDecay;

  if (actual === 1) state.bayesian.alpha++;
  else state.bayesian.beta++;
}

// =====================================================
// 13. MODEL 2: PATTERN MARKOV
// =====================================================

function algoMarkov(state) {
  const recent = state.recent;

  if (!recent.length) return 0.5;

  const previous = recent[recent.length - 1];

  const row = state.markov.transitions[previous];

  const transitionTotal = row[0] + row[1];

  const transitionP = transitionTotal > 0
    ? row[1] / transitionTotal
    : 0.5;

  const ngram = getNgramProbability(recent);

  const pattern = getPatternInfo(recent);

  let patternWeight = ngram.samples >= CFG.patternMinSamples
    ? Math.min(0.55, ngram.samples / 20)
    : 0;

  // Khi có cầu bệt hoặc xen kẽ, ưu tiên mẫu phù hợp
  // nếu đã có đủ dữ liệu lặp lại.
  if (
    pattern.type === 'streak' ||
    pattern.type === 'alternating'
  ) {
    patternWeight = Math.min(0.65, patternWeight + 0.10);
  }

  const combined =
    transitionP * (1 - patternWeight) +
    ngram.pTai * patternWeight;

  // Shrink về 50%, tránh chạy theo vài mẫu ngắn.
  return 0.5 +
    (combined - 0.5) * CFG.markovInfluence;
}

function updateMarkov(state, actual) {
  const recent = state.recent;

  if (!recent.length) return;

  const previous = recent[recent.length - 1];
  const transitions = state.markov.transitions;
  const prior = CFG.markovPrior;

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      transitions[i][j] =
        prior +
        (transitions[i][j] - prior) * CFG.markovDecay;
    }
  }

  transitions[previous][actual]++;
}

// =====================================================
// 14. MODEL 3: LOGISTIC
// =====================================================

function algoLogistic(state) {
  const features = getFeatures(state);
  const weights = state.logistic.weights;

  let z = 0;

  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }

  return sigmoid(z);
}

function updateLogistic(state, actual, features) {
  const weights = state.logistic.weights;

  let z = 0;

  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }

  const prediction = sigmoid(z);
  const error = actual - prediction;

  for (let i = 0; i < weights.length; i++) {
    const gradient =
      error * features[i] -
      CFG.logisticL2 * weights[i];

    weights[i] += CFG.logisticRate * gradient;
  }
}

// =====================================================
// 15. MODEL 4: NEURAL NETWORK
// =====================================================

function neuralForward(state, features) {
  const neural = state.neural;

  const hidden = neural.hidden.map(row => {
    let sum = 0;

    for (let i = 0; i < features.length; i++) {
      sum += row[i] * features[i];
    }

    return Math.tanh(sum);
  });

  let output = neural.bias;

  for (let i = 0; i < hidden.length; i++) {
    output += hidden[i] * neural.output[i];
  }

  return {
    hidden,
    probability: sigmoid(output)
  };
}

function algoNeural(state) {
  const features = getFeatures(state);

  return neuralForward(state, features).probability;
}

function updateNeural(state, actual, features) {
  const neural = state.neural;
  const forward = neuralForward(state, features);

  const hidden = forward.hidden;
  const error = actual - forward.probability;

  const oldOutput = [...neural.output];

  for (let i = 0; i < neural.output.length; i++) {
    neural.output[i] +=
      CFG.neuralRate * error * hidden[i];
  }

  neural.bias += CFG.neuralRate * error;

  for (let h = 0; h < neural.hidden.length; h++) {
    const derivative = 1 - hidden[h] * hidden[h];

    const hiddenError =
      error * oldOutput[h] * derivative;

    for (let i = 0; i < features.length; i++) {
      neural.hidden[h][i] +=
        CFG.neuralRate *
        hiddenError *
        features[i];
    }
  }
}

// =====================================================
// 16. MODEL 5: EWMA
// =====================================================

function algoEWMA(state) {
  return state.ewma.value;
}

function updateEWMA(state, actual) {
  state.ewma.value =
    CFG.ewmaAlpha * actual +
    (1 - CFG.ewmaAlpha) * state.ewma.value;
}

// =====================================================
// 17. MODEL REGISTRY
// =====================================================

const ALGOS = [
  { name: ALGO_NAMES[0], fn: algoBayesian },
  { name: ALGO_NAMES[1], fn: algoMarkov },
  { name: ALGO_NAMES[2], fn: algoLogistic },
  { name: ALGO_NAMES[3], fn: algoNeural },
  { name: ALGO_NAMES[4], fn: algoEWMA }
];

function getPredictions(state) {
  return ALGOS.map(algo => {
    try {
      return clamp(
        algo.fn(state),
        CFG.minProbability,
        CFG.maxProbability
      );
    } catch (err) {
      console.error('[MODEL ERROR]', algo.name, err.message);
      return 0.5;
    }
  });
}

// =====================================================
// 18. ENSEMBLE
// =====================================================

function combinePredictions(predictions, weights) {
  let total = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    const w = clamp(
      num(weights[i], 1),
      CFG.minWeight,
      CFG.maxWeight
    );

    total += predictions[i] * w;
    totalWeight += w;
  }

  return totalWeight > 0
    ? total / totalWeight
    : 0.5;
}

function updateMetaWeights(predictions, actual, state) {
  const baselineLoss = Math.log(2);

  for (let i = 0; i < predictions.length; i++) {
    const p = clamp(predictions[i], 0.01, 0.99);

    const loss = actual === 1
      ? -Math.log(p)
      : -Math.log(1 - p);

    const factor = Math.exp(
      -CFG.metaRate * (loss - baselineLoss)
    );

    state.metaWeights[i] = clamp(
      state.metaWeights[i] * factor,
      CFG.minWeight,
      CFG.maxWeight
    );
  }
}

// =====================================================
// 19. ONLINE LEARNING
// =====================================================

function learnFromNewResult(actual) {
  const state = learningState;

  // Feature và dự đoán phải được tính trước khi biết kết quả.
  const features = getFeatures(state);
  const predictions = getPredictions(state);

  const ensembleP = combinePredictions(
    predictions,
    state.metaWeights
  );

  const predicted = ensembleP >= 0.5 ? 1 : 0;

  const safeP = clamp(ensembleP, 0.01, 0.99);

  const loss = actual === 1
    ? -Math.log(safeP)
    : -Math.log(1 - safeP);

  state.stats.total++;

  if (predicted === actual) {
    state.stats.hits++;
  }

  state.stats.lastAccuracy =
    state.stats.hits / state.stats.total;

  state.stats.logLoss +=
    (loss - state.stats.logLoss) /
    state.stats.total;

  state.recentPredictions.push({
    prediction: predicted,
    actual,
    pTai: ensembleP,
    loss
  });

  if (state.recentPredictions.length > 100) {
    state.recentPredictions.shift();
  }

  const recentPredictions =
    state.recentPredictions.slice(-50);

  state.stats.recentAccuracy =
    recentPredictions.length
      ? recentPredictions.filter(
          x => x.prediction === x.actual
        ).length / recentPredictions.length
      : 0;

  state.stats.recentLogLoss =
    recentPredictions.length
      ? recentPredictions.reduce(
          (sum, x) => sum + x.loss, 0
        ) / recentPredictions.length
      : 0;

  // Chấm điểm chuyên gia trước khi cập nhật mô hình.
  updateMetaWeights(predictions, actual, state);

  // Cập nhật các mô hình.
  updateBayesian(state, actual);
  updateMarkov(state, actual);
  updateLogistic(state, actual, features);
  updateNeural(state, actual, features);
  updateEWMA(state, actual);

  // Chỉ thêm kết quả vào context sau khi dự đoán xong.
  state.recent.push(actual);

  if (state.recent.length > MAX_CONTEXT) {
    state.recent.shift();
  }

  console.log(
    `[LEARN] id=${state.lastProcessedId} ` +
    `actual=${actual} predicted=${predicted} ` +
    `accuracy=${(state.stats.lastAccuracy * 100).toFixed(2)}%`
  );
}

// =====================================================
// 20. PREDICTION
// =====================================================

function ensemblePredict() {
  const state = learningState;

  const predictions = getPredictions(state);

  let pTai = combinePredictions(
    predictions,
    state.metaWeights
  );

  const pattern = getPatternInfo(state.recent);
  const regime = detectRegimeChange(state.recent);

  // Khi phát hiện đổi trạng thái, giảm độ lệch cực đoan.
  if (regime.changed) {
    pTai = 0.5 + (pTai - 0.5) * 0.75;
  }

  pTai = clamp(
    pTai,
    CFG.minProbability,
    CFG.maxProbability
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  const agreement = predictions.filter(
    p => (p >= 0.5) === (pick === 1)
  ).length;

  const spread =
    Math.max(...predictions) -
    Math.min(...predictions);

  const weakSignal =
    Math.abs(pTai - 0.5) < 0.06 ||
    spread > 0.45 ||
    agreement <= 2;

  // Đây là điểm tín hiệu nội bộ, không phải xác suất thắng.
  const signalStrength = clamp(
    Math.abs(pTai - 0.5) * 200 +
    agreement * 4 -
    spread * 20,
    0,
    100
  );

  return {
    pick,
    pTai,
    pXiu: 1 - pTai,

    predictions,

    agreement: `${agreement}/${ALGOS.length}`,

    pattern,
    regime,

    weakSignal,

    signalStrength,

    historicalAccuracy: state.stats.lastAccuracy,
    recentAccuracy: state.stats.recentAccuracy,
    logLoss: state.stats.logLoss,
    recentLogLoss: state.stats.recentLogLoss,

    trainedSamples: state.stats.total,

    algorithms: ALGOS.map((algo, i) => ({
      name: algo.name,
      pTai: predictions[i],
      pick: predictions[i] >= 0.5 ? 1 : 0,
      weight: state.metaWeights[i]
    }))
  };
}

// =====================================================
// 21. PENDING QUEUE - PREVENT LOST TRAINING
// =====================================================

function addPendingSessions(sessions) {
  const pendingIds = new Set(
    learningState.pendingSessions.map(s => Number(s.id))
  );

  const processedIds = new Set(
    learningState.processedIds.map(Number)
  );

  let added = 0;

  for (const session of sessions) {
    const id = Number(session.id);

    if (
      !Number.isFinite(id) ||
      pendingIds.has(id) ||
      processedIds.has(id)
    ) {
      continue;
    }

    learningState.pendingSessions.push(session);
    pendingIds.add(id);
    added++;
  }

  learningState.pendingSessions.sort(
    (a, b) => Number(a.id) - Number(b.id)
  );

  return added;
}

function processPendingSessions() {
  while (learningState.pendingSessions.length) {
    const session = learningState.pendingSessions[0];

    const id = Number(session.id);
    const actual = Number(session.result);

    const alreadyProcessed =
      learningState.processedIds.some(
        x => Number(x) === id
      );

    if (alreadyProcessed) {
      const next = clone(learningState);

      next.pendingSessions =
        next.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      atomicSave(LEARNING_FILE, next);
      learningState = next;
      continue;
    }

    if (
      !Number.isFinite(id) ||
      ![0, 1].includes(actual)
    ) {
      const next = clone(learningState);
      next.pendingSessions.shift();

      atomicSave(LEARNING_FILE, next);
      learningState = next;
      continue;
    }

    const previous = learningState;
    const next = clone(previous);

    learningState = next;

    try {
      learnFromNewResult(actual);

      next.processedIds.push(id);
      next.processedIds =
        next.processedIds.slice(-MAX_PROCESSED_IDS);

      next.lastProcessedId =
        next.lastProcessedId === null
          ? id
          : Math.max(Number(next.lastProcessedId), id);

      next.pendingSessions =
        next.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      // Model + processedIds + pending được lưu cùng nhau.
      atomicSave(LEARNING_FILE, next);

      learningState = next;

    } catch (err) {
      learningState = previous;
      throw err;
    }
  }
}

// =====================================================
// 22. SYNC
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

    if (!Array.isArray(list)) {
      throw new Error('Invalid source API response');
    }

    const history = readHistory();

    const knownIds = new Set(
      history.map(s => Number(s.id))
    );

    const newSessions = [];

    for (const item of list) {
      const session = normalizeSession(item);

      if (
        !Number.isFinite(session.id) ||
        !Number.isFinite(session.sum) ||
        ![0, 1].includes(session.result)
      ) {
        continue;
      }

      if (knownIds.has(session.id)) continue;

      knownIds.add(session.id);
      newSessions.push(session);
    }

    // Khởi tạo lần đầu: không học lịch sử cũ.
    if (!learningState.initialized) {
      const merged = mergeHistory(
        history,
        newSessions
      );

      writeHistory(merged);

      learningState.initialized = true;

      learningState.lastProcessedId = merged.length
        ? Math.max(...merged.map(s => Number(s.id)))
        : null;

      learningState.pendingSessions = [];
      learningState.processedIds = [];

      saveLearningState();

      console.log('[LEARNING] Baseline initialized');
      console.log('[LEARNING] Old history not trained');

      return;
    }

    // Lưu pending trước khi ghi history.
    const added = addPendingSessions(newSessions);

    if (added > 0) {
      saveLearningState();
    }

    if (newSessions.length > 0) {
      const merged = mergeHistory(
        history,
        newSessions
      );

      writeHistory(merged);

      console.log(
        `[SYNC] Added ${newSessions.length} sessions`
      );
    }

    // Tiếp tục xử lý cả những phiên bị bỏ dở từ lần trước.
    if (learningState.pendingSessions.length) {
      processPendingSessions();
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 23. TIME
// =====================================================

function formatTimeVN() {
  return new Date().toLocaleString('sv-SE', {
    timeZone: 'Asia/Ho_Chi_Minh'
  }).replace(' ', 'T') + '+07:00';
}

// =====================================================
// 24. RESPONSE
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
  const prediction = ensemblePredict();

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,
    tong: current.sum,

    dudoanphien: Number(current.id) + 1,

    dudoan: prediction.pick === 1
      ? 'tai'
      : 'xiu',

    xacsuattai:
      `${(prediction.pTai * 100).toFixed(2)}%`,

    xacsuatxiu:
      `${(prediction.pXiu * 100).toFixed(2)}%`,

    // Không gọi đây là xác suất thắng.
    dotincay:
      prediction.weakSignal
        ? 'Tín hiệu yếu'
        : 'Tín hiệu tương đối rõ',

    mucdotinhieu:
      Number(prediction.signalStrength.toFixed(2)),

    trangthaicau:
      prediction.pattern.type,

    do_dai_cau:
      prediction.pattern.streak,

    phat_hien_doi_trang_thai:
      prediction.regime.changed,

    dongthuan:
      prediction.agreement,

    dochinhxaclichsu:
      `${(prediction.historicalAccuracy * 100).toFixed(2)}%`,

    dochinhxac50phien:
      `${(prediction.recentAccuracy * 100).toFixed(2)}%`,

    logloss:
      Number(prediction.logLoss.toFixed(5)),

    logloss50phien:
      Number(prediction.recentLogLoss.toFixed(5)),

    solanhoctap:
      prediction.trainedSamples,

    trongso: prediction.algorithms.map(a => ({
      thuattoan: a.name,

      du_doan: a.pick === 1
        ? 'tai'
        : 'xiu',

      xac_suat_tai:
        Number(a.pTai.toFixed(4)),

      trong_so:
        Number(a.weight.toFixed(4))
    })),

    uptime: formatTimeVN()
  };
}

// =====================================================
// 25. STATUS API
// =====================================================

function buildStatus() {
  const state = learningState;
  const history = readHistory();
  const prediction = ensemblePredict();

  return {
    status: 'online',

    deployId: DEPLOY_ID,

    initialized: state.initialized,

    currentSession: history.length
      ? history[history.length - 1].id
      : null,

    historyCount: history.length,

    lastProcessedId: state.lastProcessedId,

    pending: state.pendingSessions.length,

    trainedSamples: state.stats.total,

    accuracy: Number(
      (state.stats.lastAccuracy * 100).toFixed(2)
    ),

    recentAccuracy: Number(
      (state.stats.recentAccuracy * 100).toFixed(2)
    ),

    logLoss: Number(
      state.stats.logLoss.toFixed(5)
    ),

    recentLogLoss: Number(
      state.stats.recentLogLoss.toFixed(5)
    ),

    pattern: prediction.pattern,

    regimeChange: prediction.regime,

    weakSignal: prediction.weakSignal,

    prediction: {
      pick: prediction.pick === 1 ? 'tai' : 'xiu',

      pTai: Number(
        (prediction.pTai * 100).toFixed(2)
      ),

      pXiu: Number(
        (prediction.pXiu * 100).toFixed(2)
      ),

      agreement: prediction.agreement
    },

    algorithms: prediction.algorithms.map(a => ({
      name: a.name,
      prediction: a.pick === 1 ? 'tai' : 'xiu',
      pTai: Number((a.pTai * 100).toFixed(2)),
      weight: Number(a.weight.toFixed(4))
    })),

    uptime: formatTimeVN()
  };
}

// =====================================================
// 26. API ROUTES
// =====================================================

app.get('/', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[API]', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/predict', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[PREDICT]', err.message);
    res.status(500).json({ error: 'Prediction error' });
  }
});

app.get('/status', (req, res) => {
  try {
    res.json(buildStatus());
  } catch (err) {
    console.error('[STATUS]', err.message);
    res.status(500).json({ error: 'Status error' });
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
    res.status(500).json({ error: 'History error' });
  }
});

// =====================================================
// 27. BOOT
// =====================================================

handleDeploy();
initDataFile();

learningState = loadLearningState();

console.log('[BOOT] Deploy:', DEPLOY_ID);
console.log('[BOOT] Five learning models loaded');
console.log('[BOOT] Pattern recognition enabled');
console.log('[BOOT] Change detection enabled');
console.log('[BOOT] Log Loss meta-learning enabled');
console.log('[BOOT] Pending session queue enabled');
console.log('[BOOT] Status endpoint: /status');

syncData();

const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

const server = app.listen(PORT, () => {
  console.log(`[BOOT] Server running on port ${PORT}`);
  console.log(`[BOOT] Vietnam time: ${formatTimeVN()}`);
});

function shutdown() {
  console.log('[SHUTDOWN] Saving learning state');

  try {
    saveLearningState();
  } catch (err) {
    console.error('[SHUTDOWN SAVE]', err.message);
  }

  clearInterval(syncTimer);

  server.close(() => {
    process.exit(0);
  });

  setTimeout(() => {
    process.exit(1);
  }, 10000).unref();
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
