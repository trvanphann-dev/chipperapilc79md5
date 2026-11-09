
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '1mb' }));

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const LEARNING_FILE = path.join(__dirname, 'learning_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const MAX_CONTEXT = 64;
const MAX_PROCESSED_IDS = 2000;
const PORT = process.env.PORT || 3000;

// =====================================================
// 1. STORAGE
// =====================================================

function atomicSave(file, data) {
  const tempFile = file + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, file);
}

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    atomicSave(DATA_FILE, []);
    console.log('[INIT] Created history file');
  } else {
    console.log('[INIT] Existing history preserved');
  }
}

function readHistory() {
  try {
    const parsed = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[HISTORY READ]', err.message);
    return [];
  }
}

function writeHistory(data) {
  atomicSave(DATA_FILE, data);
}

function cloneObject(obj) {
  return JSON.parse(JSON.stringify(obj));
}

// =====================================================
// 2. SOURCE API
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

  const rawSum = Number(raw.sum ?? 0);

  const sum = dices.length
    ? dices.reduce((a, b) => a + b, 0)
    : rawSum;

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    (Array.isArray(raw.result) ? '' : raw.result ?? '')
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
    id,
    dices,
    sum,
    result
  };
}

// =====================================================
// 3. LEARNING CONFIGURATION
// =====================================================

const LEARNING = {
  initialWeight: 1,
  minWeight: 0.1,
  maxWeight: 10,

  // Ensemble Log Loss
  metaLearningRate: 0.04,

  // Giảm tốc độ học Logistic
  logisticLearningRate: 0.015,

  // Giảm tốc độ cập nhật Neural Network
  neuralLearningRate: 0.01,

  // Bayesian thích nghi dữ liệu mới
  bayesianDecay: 0.985,

  // Markov transition decay
  markovDecay: 0.99,

  // EWMA
  ewmaAlpha: 0.15
};

const ALGO_NAMES = [
  'bayesian_online',
  'markov_online',
  'online_logistic',
  'adaptive_neural_network',
  'ewma_online'
];

// =====================================================
// 4. INITIAL STATE
// =====================================================

function createLearningState() {
  return {
    version: 2,
    initialized: false,
    lastProcessedId: null,

    // Bayesian
    bayesian: {
      alpha: 2,
      beta: 2
    },

    // Markov: hàng là kết quả trước
    // cột 0 = Xỉu, cột 1 = Tài
    markov: {
      transitions: [
        [2, 2],
        [2, 2]
      ]
    },

    // Logistic
    logistic: {
      weights: [0, 0, 0, 0]
    },

    // Neural Network
    neural: {
      inputSize: 4,

      hidden: [
        [0.12, -0.08, 0.05, 0.10],
        [-0.06, 0.11, 0.08, -0.04],
        [0.09, 0.04, -0.12, 0.07],
        [-0.10, 0.06, 0.03, 0.12]
      ],

      output: [0.1, -0.1, 0.08, 0.05],
      bias: 0
    },

    // EWMA
    ewma: {
      value: 0.5
    },

    // Ensemble weights
    metaWeights: [1, 1, 1, 1, 1],

    // Context chỉ chứa kết quả đã học
    recent: [],

    // Chống học trùng
    processedIds: [],

    // Hàng đợi phiên chưa học
    pendingSessions: [],

    stats: {
      total: 0,
      hits: 0,
      lastAccuracy: 0
    }
  };
}

// =====================================================
// 5. LOAD AND MIGRATE LEARNING STATE
// =====================================================

function loadLearningState() {
  if (!fs.existsSync(LEARNING_FILE)) {
    console.log('[LEARNING] Creating new state');
    return createLearningState();
  }

  try {
    const old = JSON.parse(
      fs.readFileSync(LEARNING_FILE, 'utf8')
    );

    if (!old || typeof old !== 'object') {
      throw new Error('Invalid learning state');
    }

    const base = createLearningState();

    // Không làm mất dữ liệu học phiên bản cũ
    if (old.bayesian) {
      base.bayesian = {
        ...base.bayesian,
        ...old.bayesian
      };
    }

    if (old.logistic?.weights?.length === 4) {
      base.logistic.weights = old.logistic.weights;
    }

    if (
      Array.isArray(old.neural?.hidden) &&
      old.neural.hidden.length === 4
    ) {
      base.neural = {
        ...base.neural,
        ...old.neural
      };
    }

    if (old.ewma) {
      base.ewma = {
        ...base.ewma,
        ...old.ewma
      };
    }

    if (
      Array.isArray(old.metaWeights) &&
      old.metaWeights.length === 5
    ) {
      base.metaWeights = old.metaWeights.map(w =>
        clamp(Number(w) || 1, LEARNING.minWeight, LEARNING.maxWeight)
      );
    }

    if (Array.isArray(old.recent)) {
      base.recent = old.recent
        .filter(v => v === 0 || v === 1)
        .slice(-MAX_CONTEXT);
    }

    if (old.stats) {
      base.stats = {
        ...base.stats,
        ...old.stats
      };
    }

    // Giữ Markov nếu đã tồn tại
    if (
      Array.isArray(old.markov?.transitions) &&
      old.markov.transitions.length === 2
    ) {
      base.markov = old.markov;
    }

    base.version = 2;
    base.initialized = Boolean(old.initialized);
    base.lastProcessedId = old.lastProcessedId ?? null;

    base.processedIds = Array.isArray(old.processedIds)
      ? old.processedIds.slice(-MAX_PROCESSED_IDS)
      : [];

    base.pendingSessions = Array.isArray(old.pendingSessions)
      ? old.pendingSessions
      : [];

    console.log('[LEARNING] State loaded/migrated');

    return base;

  } catch (err) {
    console.error('[LEARNING LOAD]', err.message);

    // Không xóa file cũ khi đọc lỗi.
    // Khởi tạo trạng thái mới trong bộ nhớ.
    return createLearningState();
  }
}

let learningState = null;

function saveLearningState() {
  atomicSave(LEARNING_FILE, learningState);
}

// =====================================================
// 6. COMMON FUNCTIONS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  x = clamp(x, -30, 30);
  return 1 / (1 + Math.exp(-x));
}

function makePrediction(pTai) {
  const p = clamp(
    Number.isFinite(pTai) ? pTai : 0.5,
    0.01,
    0.99
  );

  return {
    pick: p >= 0.5 ? 1 : 0,
    pTai: p
  };
}

// =====================================================
// 7. FEATURE ENGINEERING
// =====================================================

function getFeatures(state) {
  const recent = state.recent || [];

  const last = recent.length
    ? recent[recent.length - 1]
    : 0.5;

  let streak = 0;

  if (recent.length) {
    const current = recent[recent.length - 1];

    for (let i = recent.length - 1; i >= 0; i--) {
      if (recent[i] !== current) break;
      streak++;
    }

    streak *= current === 1 ? 1 : -1;
  }

  return [
    1,
    last * 2 - 1,
    (state.ewma.value - 0.5) * 2,
    clamp(streak / 10, -1, 1)
  ];
}

// =====================================================
// 8. ALGORITHM 1: BAYESIAN WITH DECAY
// =====================================================

function algoBayesian(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  return makePrediction(a / (a + b));
}

function updateBayesian(state, actual) {
  const decay = LEARNING.bayesianDecay;

  // Co số liệu cũ về prior 2,2
  state.bayesian.alpha =
    2 + (state.bayesian.alpha - 2) * decay;

  state.bayesian.beta =
    2 + (state.bayesian.beta - 2) * decay;

  if (actual === 1) {
    state.bayesian.alpha++;
  } else {
    state.bayesian.beta++;
  }
}

// =====================================================
// 9. ALGORITHM 2: MARKOV TRANSITION MODEL
// =====================================================

function algoMarkov(state) {
  const recent = state.recent;

  if (!recent.length) {
    return makePrediction(0.5);
  }

  const previous = recent[recent.length - 1];

  const row = state.markov.transitions[previous];

  const total = row[0] + row[1];

  if (!total) {
    return makePrediction(0.5);
  }

  return makePrediction(row[1] / total);
}

function updateMarkov(state, actual) {
  const recent = state.recent;

  // Không có kết quả trước để xác định chuyển trạng thái
  if (!recent.length) return;

  const previous = recent[recent.length - 1];

  const transitions = state.markov.transitions;
  const decay = LEARNING.markovDecay;

  // Decay về prior, không để dữ liệu cũ chi phối mãi
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      transitions[i][j] =
        2 + (transitions[i][j] - 2) * decay;
    }
  }

  transitions[previous][actual]++;
}

// =====================================================
// 10. ALGORITHM 3: ONLINE LOGISTIC
// =====================================================

function algoLogistic(state) {
  const features = getFeatures(state);
  const weights = state.logistic.weights;

  let z = 0;

  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }

  return makePrediction(sigmoid(z));
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
    weights[i] +=
      LEARNING.logisticLearningRate *
      error *
      features[i];
  }
}

// =====================================================
// 11. ALGORITHM 4: ADAPTIVE NEURAL NETWORK
// =====================================================

function neuralForward(state, features) {
  const neural = state.neural;

  const hiddenValues = neural.hidden.map(row => {
    let sum = 0;

    for (let i = 0; i < features.length; i++) {
      sum += row[i] * features[i];
    }

    return Math.tanh(sum);
  });

  let output = neural.bias;

  for (let i = 0; i < hiddenValues.length; i++) {
    output += hiddenValues[i] * neural.output[i];
  }

  return {
    hiddenValues,
    output: sigmoid(output)
  };
}

function algoNeural(state) {
  const features = getFeatures(state);
  const forward = neuralForward(state, features);

  return makePrediction(forward.output);
}

function updateNeural(state, actual, features) {
  const neural = state.neural;

  const forward = neuralForward(state, features);

  const hiddenValues = forward.hiddenValues;
  const prediction = forward.output;

  const error = actual - prediction;

  // Lưu trọng số cũ trước khi cập nhật
  const oldOutput = [...neural.output];

  // Output layer
  for (let i = 0; i < neural.output.length; i++) {
    neural.output[i] +=
      LEARNING.neuralLearningRate *
      error *
      hiddenValues[i];
  }

  neural.bias +=
    LEARNING.neuralLearningRate * error;

  // Hidden layer
  for (let h = 0; h < neural.hidden.length; h++) {
    const derivative =
      1 - hiddenValues[h] * hiddenValues[h];

    const hiddenError =
      error * oldOutput[h] * derivative;

    for (let i = 0; i < features.length; i++) {
      neural.hidden[h][i] +=
        LEARNING.neuralLearningRate *
        hiddenError *
        features[i];
    }
  }
}

// =====================================================
// 12. ALGORITHM 5: EWMA ONLINE
// =====================================================

function algoEWMA(state) {
  return makePrediction(state.ewma.value);
}

function updateEWMA(state, actual) {
  const alpha = LEARNING.ewmaAlpha;

  state.ewma.value =
    alpha * actual +
    (1 - alpha) * state.ewma.value;
}

// =====================================================
// 13. REGISTER FIVE ALGORITHMS
// =====================================================

const ALGOS = [
  { name: ALGO_NAMES[0], fn: algoBayesian },
  { name: ALGO_NAMES[1], fn: algoMarkov },
  { name: ALGO_NAMES[2], fn: algoLogistic },
  { name: ALGO_NAMES[3], fn: algoNeural },
  { name: ALGO_NAMES[4], fn: algoEWMA }
];

function getAlgoPredictions(state) {
  return ALGOS.map(algo => {
    try {
      return algo.fn(state);
    } catch (err) {
      console.error('[ALGO ERROR]', algo.name, err.message);
      return makePrediction(0.5);
    }
  });
}

// =====================================================
// 14. ENSEMBLE: LOG LOSS
// =====================================================

function combinePredictions(predictions, weights) {
  let total = 0;
  let weightTotal = 0;

  for (let i = 0; i < predictions.length; i++) {
    const weight = Number(weights[i]) || 0;

    total += predictions[i].pTai * weight;
    weightTotal += weight;
  }

  return weightTotal > 0
    ? total / weightTotal
    : 0.5;
}

function updateMetaWeights(predictions, actual, state) {
  const weights = state.metaWeights;

  // Log Loss của dự đoán ngẫu nhiên 50/50
  const baselineLoss = Math.log(2);

  for (let i = 0; i < predictions.length; i++) {
    const p = clamp(
      predictions[i].pTai,
      0.01,
      0.99
    );

    const loss = actual === 1
      ? -Math.log(p)
      : -Math.log(1 - p);

    // Mô hình có Log Loss thấp hơn baseline
    // sẽ được tăng trọng số tương đối.
    const factor = Math.exp(
      -LEARNING.metaLearningRate *
      (loss - baselineLoss)
    );

    weights[i] = clamp(
      weights[i] * factor,
      LEARNING.minWeight,
      LEARNING.maxWeight
    );
  }
}

// =====================================================
// 15. ONLINE LEARNING ENGINE
// =====================================================

// Chỉ học phiên mới.
// Dự đoán được tính trước khi cập nhật bằng kết quả thực tế.

function learnFromNewResult(actual) {
  const state = learningState;

  const features = getFeatures(state);

  // Dự đoán trước khi biết kết quả
  const predictions = getAlgoPredictions(state);

  const ensembleP = combinePredictions(
    predictions,
    state.metaWeights
  );

  const predicted = ensembleP >= 0.5 ? 1 : 0;

  // Thống kê hiệu suất thực tế
  state.stats.total++;

  if (predicted === actual) {
    state.stats.hits++;
  }

  state.stats.lastAccuracy =
    state.stats.total > 0
      ? state.stats.hits / state.stats.total
      : 0;

  // Cập nhật trọng số ensemble bằng Log Loss
  updateMetaWeights(
    predictions,
    actual,
    state
  );

  // Cập nhật từng mô hình
  updateBayesian(state, actual);

  updateMarkov(state, actual);

  updateLogistic(
    state,
    actual,
    features
  );

  updateNeural(
    state,
    actual,
    features
  );

  updateEWMA(
    state,
    actual
  );

  // Chỉ thêm kết quả sau khi đã học
  state.recent.push(actual);

  if (state.recent.length > MAX_CONTEXT) {
    state.recent.shift();
  }

  console.log(
    `[LEARNING] Actual=${actual} | ` +
    `Prediction=${predicted} | ` +
    `Accuracy=${(state.stats.lastAccuracy * 100).toFixed(2)}%`
  );
}

// =====================================================
// 16. PREDICTION ENGINE
// =====================================================

function ensemblePredict() {
  const state = learningState || createLearningState();

  const predictions = getAlgoPredictions(state);

  const pTai = combinePredictions(
    predictions,
    state.metaWeights
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  const agreement = predictions.filter(
    p => (p.pTai >= 0.5) === (pTai >= 0.5)
  ).length;

  const accuracy = state.stats.lastAccuracy || 0;

  // Chỉ là chỉ số tham khảo, không phải xác suất thắng.
  const confidence = clamp(
    50 + Math.max(0, accuracy - 0.5) * 50,
    50,
    75
  );

  return {
    pick,

    confidence: Math.round(confidence * 100) / 100,

    pTai: Math.round(pTai * 10000) / 10000,

    agreement: `${agreement}/${ALGOS.length}`,

    historicalAccuracy:
      Math.round(accuracy * 10000) / 100,

    trainedSamples: state.stats.total,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,
      pick: predictions[index].pick,

      probabilityTai:
        Math.round(predictions[index].pTai * 10000) / 10000,

      weight:
        Math.round(state.metaWeights[index] * 10000) / 10000
    }))
  };
}

// =====================================================
// 17. VIETNAM TIME
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
// 18. HISTORY MERGE
// =====================================================

function mergeHistory(oldHistory, newSessions) {
  const map = new Map();

  for (const item of oldHistory) {
    if (
      item &&
      Number.isFinite(Number(item.id))
    ) {
      map.set(Number(item.id), item);
    }
  }

  for (const item of newSessions) {
    map.set(Number(item.id), item);
  }

  return [...map.values()]
    .sort((a, b) => Number(a.id) - Number(b.id))
    .slice(-MAX_HISTORY);
}

// =====================================================
// 19. PENDING QUEUE
// =====================================================

function addPendingSessions(sessions) {
  const state = learningState;

  const pendingIds = new Set(
    state.pendingSessions.map(s => Number(s.id))
  );

  const processed = new Set(
    state.processedIds.map(Number)
  );

  let added = 0;

  for (const session of sessions) {
    const id = Number(session.id);

    if (
      !Number.isFinite(id) ||
      pendingIds.has(id) ||
      processed.has(id)
    ) {
      continue;
    }

    state.pendingSessions.push(session);
    pendingIds.add(id);
    added++;
  }

  state.pendingSessions.sort(
    (a, b) => Number(a.id) - Number(b.id)
  );

  return added;
}

// =====================================================
// 20. PROCESS PENDING SESSIONS SAFELY
// =====================================================

// Mỗi phiên được lưu trạng thái học và processedIds
// trong cùng một lần atomicSave.
//
// Nếu server chết trước khi lưu:
// phiên vẫn còn trong hàng đợi và sẽ được học lại
// từ trạng thái cũ.
//
// Nếu server chết sau khi lưu:
// phiên đã nằm trong processedIds nên không học trùng.

function processPendingSessions() {
  while (learningState.pendingSessions.length > 0) {
    const session = learningState.pendingSessions[0];

    const id = Number(session.id);
    const actual = Number(session.result);

    if (
      !Number.isFinite(id) ||
      ![0, 1].includes(actual)
    ) {
      const candidate = cloneObject(learningState);

      candidate.pendingSessions.shift();

      atomicSave(LEARNING_FILE, candidate);
      learningState = candidate;

      console.error('[LEARNING] Removed invalid pending session');
      continue;
    }

    if (
      learningState.processedIds
        .map(Number)
        .includes(id)
    ) {
      const candidate = cloneObject(learningState);

      candidate.pendingSessions =
        candidate.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      atomicSave(LEARNING_FILE, candidate);
      learningState = candidate;

      continue;
    }

    // Thao tác trên bản sao để tránh thay đổi state
    // trong bộ nhớ nếu việc lưu file thất bại.
    const candidate = cloneObject(learningState);

    const previousState = learningState;

    learningState = candidate;

    try {
      learnFromNewResult(actual);

      candidate.processedIds.push(id);

      if (
        candidate.processedIds.length > MAX_PROCESSED_IDS
      ) {
        candidate.processedIds =
          candidate.processedIds.slice(-MAX_PROCESSED_IDS);
      }

      candidate.lastProcessedId =
        candidate.lastProcessedId === null
          ? id
          : Math.max(Number(candidate.lastProcessedId), id);

      candidate.pendingSessions =
        candidate.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      atomicSave(LEARNING_FILE, candidate);

      learningState = candidate;

    } catch (err) {
      learningState = previousState;
      throw err;
    }
  }
}

// =====================================================
// 21. SOURCE SYNC + ONLINE LEARNING
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
      throw new Error('Invalid source data');
    }

    const history = readHistory();

    const knownIds = new Set(
      history.map(h => Number(h.id))
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

      if (knownIds.has(session.id)) {
        continue;
      }

      knownIds.add(session.id);
      newSessions.push(session);
    }

    // -------------------------------------------------
    // FIRST STARTUP
    // -------------------------------------------------

    if (!learningState.initialized) {
      const merged = mergeHistory(
        history,
        newSessions
      );

      if (newSessions.length > 0) {
        writeHistory(merged);
      }

      const latest = merged.length
        ? Math.max(...merged.map(s => Number(s.id)))
        : null;

      learningState.initialized = true;
      learningState.lastProcessedId = latest;

      // Không học lịch sử cũ.
      learningState.pendingSessions = [];
      learningState.processedIds = [];

      saveLearningState();

      console.log('[LEARNING] Initial baseline established');
      console.log('[LEARNING] Old history NOT used for training');

      return;
    }

    // -------------------------------------------------
    // PERSIST QUEUE BEFORE HISTORY
    // -------------------------------------------------

    const added = addPendingSessions(newSessions);

    if (added > 0) {
      // Lưu hàng đợi trước khi ghi lịch sử.
      saveLearningState();
    }

    // -------------------------------------------------
    // SAVE HISTORY
    // -------------------------------------------------

    if (newSessions.length > 0) {
      const merged = mergeHistory(
        history,
        newSessions
      );

      writeHistory(merged);

      console.log(
        `[SYNC] +${newSessions.length} sessions | Total ${merged.length}`
      );
    }

    // -------------------------------------------------
    // PROCESS PENDING QUEUE
    // -------------------------------------------------

    if (learningState.pendingSessions.length > 0) {
      const before = learningState.pendingSessions.length;

      processPendingSessions();

      console.log(
        `[LEARNING] Queue processed: ${before} sessions`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 22. API RESPONSE
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

  const ens = ensemblePredict();

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: Number(current.id) + 1,

    dudoan: ens.pick === 1
      ? 'tai'
      : 'xiu',

    dotincay: `${ens.confidence}%`,

    xacsuattai:
      `${(ens.pTai * 100).toFixed(2)}%`,

    xacsuatxiu:
      `${((1 - ens.pTai) * 100).toFixed(2)}%`,

    dochinhxaclichsu:
      `${ens.historicalAccuracy}%`,

    solanhoctap:
      ens.trainedSamples,

    dongthuan:
      ens.agreement,

    trongso: ens.algorithms.map(a => ({
      thuattoan: a.name,

      du_doan: a.pick === 1
        ? 'tai'
        : 'xiu',

      xac_suat_tai:
        a.probabilityTai,

      trong_so:
        a.weight
    })),

    uptime: formatTimeVN()
  };
}

// =====================================================
// 23. API ROUTES
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

// =====================================================
// 24. BOOT SERVER
// =====================================================

initDataFile();

learningState = loadLearningState();

// Đồng bộ ngay khi khởi động
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

  console.log(`[BOOT] ${ALGOS.length} algorithms loaded`);

  console.log('[BOOT] Bayesian decay enabled');
  console.log('[BOOT] Markov transition model enabled');
  console.log('[BOOT] Logistic slow learning enabled');
  console.log('[BOOT] Neural slow learning enabled');
  console.log('[BOOT] Ensemble Log Loss enabled');

  console.log('[BOOT] Persistent online learning enabled');

  console.log(`[BOOT] History: ${DATA_FILE}`);
  console.log(`[BOOT] Learning: ${LEARNING_FILE}`);
});
