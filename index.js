
'use strict';

const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '1mb' }));

// =====================================================
// 1. CONFIGURATION
// =====================================================

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const LEARNING_FILE = path.join(__dirname, 'learning_state.json');
const DEPLOY_MARKER = path.join(__dirname, 'deploy_marker.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const MAX_CONTEXT = 64;
const MAX_PROCESSED_IDS = 3000;

const PORT = process.env.PORT || 3000;

// Render cung cấp RENDER_GIT_COMMIT.
// Local có thể tự đặt DEPLOY_ID.
const DEPLOY_ID =
  process.env.RENDER_GIT_COMMIT ||
  process.env.DEPLOY_ID ||
  'local';

// =====================================================
// 2. LEARNING CONFIGURATION
// =====================================================

const LEARNING = {
  // Bayesian prior mạnh hơn để tránh phản ứng quá nhanh
  bayesianPrior: 20,
  bayesianDecay: 0.99,

  // Markov có prior lớn và ảnh hưởng bị giới hạn
  markovPrior: 20,
  markovInfluence: 0.20,
  markovDecay: 0.995,

  // Logistic
  logisticLearningRate: 0.005,
  logisticL2: 0.001,

  // Neural
  neuralLearningRate: 0.003,

  // EWMA
  ewmaAlpha: 0.08,

  // Ensemble Log Loss
  metaLearningRate: 0.025,
  minWeight: 0.2,
  maxWeight: 5,

  // Giới hạn xác suất tránh dự đoán cực đoan
  minProbability: 0.10,
  maxProbability: 0.90
};

const ALGO_NAMES = [
  'bayesian_online',
  'markov_transition',
  'online_logistic',
  'adaptive_neural_network',
  'ewma_online'
];

// =====================================================
// 3. COMMON FUNCTIONS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  x = clamp(x, -25, 25);
  return 1 / (1 + Math.exp(-x));
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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

function removeFileIfExists(file) {
  for (const target of [file, file + '.tmp']) {
    if (fs.existsSync(target)) {
      fs.unlinkSync(target);
    }
  }
}

// =====================================================
// 4. DEPLOY RESET
// =====================================================

// Quy tắc:
// - Lần chạy đầu: giữ dữ liệu cũ nếu có.
// - Cùng commit: giữ dữ liệu.
// - Commit mới: tạo lại history và learning state.
// - Restart cùng commit: không xóa dữ liệu.

function handleDeployReset() {
  let previous = null;

  try {
    if (fs.existsSync(DEPLOY_MARKER)) {
      previous = JSON.parse(
        fs.readFileSync(DEPLOY_MARKER, 'utf8')
      );
    }
  } catch (err) {
    console.error('[DEPLOY MARKER]', err.message);
  }

  if (
    previous &&
    previous.deployId &&
    previous.deployId !== DEPLOY_ID
  ) {
    console.log('[DEPLOY] New commit detected');
    console.log('[DEPLOY] Resetting history and learning weights');

    removeFileIfExists(DATA_FILE);
    removeFileIfExists(LEARNING_FILE);

    console.log('[DEPLOY] Old data removed');
  } else if (!previous) {
    console.log('[DEPLOY] First marker creation');
    console.log('[DEPLOY] Existing data will be preserved');
  } else {
    console.log('[DEPLOY] Same commit - keeping saved data');
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
    console.log('[INIT] Created history file');
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

function writeHistory(history) {
  atomicSave(DATA_FILE, history);
}

function mergeHistory(oldHistory, newSessions) {
  const map = new Map();

  for (const item of oldHistory) {
    const id = Number(item?.id);

    if (Number.isFinite(id)) {
      map.set(id, item);
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
// 6. SOURCE API
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

  const sumFromDice = dices.length
    ? dices.reduce((a, b) => a + b, 0)
    : 0;

  const sum = sumFromDice ||
    safeNumber(raw.sum, 0);

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
// 7. INITIAL LEARNING STATE
// =====================================================

function createLearningState() {
  return {
    version: 3,

    initialized: false,
    lastProcessedId: null,

    bayesian: {
      alpha: LEARNING.bayesianPrior,
      beta: LEARNING.bayesianPrior
    },

    // Markov:
    // transitions[0] = phiên trước Xỉu
    // transitions[1] = phiên trước Tài
    // cột 0 = Xỉu tiếp theo
    // cột 1 = Tài tiếp theo
    markov: {
      transitions: [
        [LEARNING.markovPrior, LEARNING.markovPrior],
        [LEARNING.markovPrior, LEARNING.markovPrior]
      ]
    },

    logistic: {
      weights: [0, 0, 0, 0]
    },

    neural: {
      hidden: [
        [0.05, -0.03, 0.04, 0.02],
        [-0.04, 0.03, 0.02, -0.05],
        [0.03, 0.04, -0.03, 0.02],
        [-0.02, 0.05, -0.04, 0.03]
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
      logLoss: 0
    }
  };
}

// =====================================================
// 8. LOAD AND MIGRATE STATE
// =====================================================

function loadLearningState() {
  if (!fs.existsSync(LEARNING_FILE)) {
    return createLearningState();
  }

  try {
    const old = JSON.parse(
      fs.readFileSync(LEARNING_FILE, 'utf8')
    );

    const base = createLearningState();

    if (old.bayesian) {
      base.bayesian = {
        ...base.bayesian,
        ...old.bayesian
      };
    }

    if (
      Array.isArray(old.logistic?.weights) &&
      old.logistic.weights.length === 4
    ) {
      base.logistic.weights = old.logistic.weights;
    }

    if (old.ewma) {
      base.ewma = {
        ...base.ewma,
        ...old.ewma
      };
    }

    if (
      Array.isArray(old.neural?.hidden) &&
      old.neural.hidden.length === 4 &&
      Array.isArray(old.neural?.output) &&
      old.neural.output.length === 4
    ) {
      base.neural = old.neural;
    }

    if (
      Array.isArray(old.markov?.transitions) &&
      old.markov.transitions.length === 2
    ) {
      base.markov = old.markov;
    }

    if (
      Array.isArray(old.metaWeights) &&
      old.metaWeights.length === 5
    ) {
      base.metaWeights = old.metaWeights.map(w =>
        clamp(
          safeNumber(w, 1),
          LEARNING.minWeight,
          LEARNING.maxWeight
        )
      );
    }

    if (Array.isArray(old.recent)) {
      base.recent = old.recent
        .filter(v => v === 0 || v === 1)
        .slice(-MAX_CONTEXT);
    }

    if (Array.isArray(old.processedIds)) {
      base.processedIds =
        old.processedIds.slice(-MAX_PROCESSED_IDS);
    }

    if (Array.isArray(old.pendingSessions)) {
      base.pendingSessions = old.pendingSessions;
    }

    if (old.stats) {
      base.stats = {
        ...base.stats,
        ...old.stats
      };
    }

    base.initialized = Boolean(old.initialized);
    base.lastProcessedId = old.lastProcessedId ?? null;
    base.version = 3;

    return base;

  } catch (err) {
    console.error('[LEARNING LOAD]', err.message);
    console.error('[LEARNING] Using fresh in-memory state');

    // Không tự xóa file lỗi để tránh mất dữ liệu cũ.
    return createLearningState();
  }
}

let learningState = null;

function saveLearningState() {
  atomicSave(LEARNING_FILE, learningState);
}

// =====================================================
// 9. FEATURE ENGINEERING
// =====================================================

// Không sử dụng streak.
// Không sử dụng quy tắc "bệt".
// Không có feature đặc biệt cho tổng 11.
//
// Các feature chỉ mô tả thống kê tổng hợp.
// Chúng không được xem là tín hiệu chắc chắn cho phiên sau.

function getFeatures(state) {
  const recent = state.recent || [];

  const ewma = state.ewma.value;

  const mean = recent.length
    ? recent.reduce((a, b) => a + b, 0) / recent.length
    : 0.5;

  let changes = 0;

  for (let i = 1; i < recent.length; i++) {
    if (recent[i] !== recent[i - 1]) {
      changes++;
    }
  }

  const changeRate = recent.length > 1
    ? changes / (recent.length - 1)
    : 0.5;

  return [
    1,
    clamp((ewma - 0.5) * 2, -1, 1),
    clamp((mean - 0.5) * 2, -1, 1),
    clamp((changeRate - 0.5) * 2, -1, 1)
  ];
}

// =====================================================
// 10. ALGORITHM 1: BAYESIAN
// =====================================================

function algoBayesian(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  return a + b > 0
    ? a / (a + b)
    : 0.5;
}

function updateBayesian(state, actual) {
  const prior = LEARNING.bayesianPrior;
  const decay = LEARNING.bayesianDecay;

  state.bayesian.alpha =
    prior + (state.bayesian.alpha - prior) * decay;

  state.bayesian.beta =
    prior + (state.bayesian.beta - prior) * decay;

  if (actual === 1) {
    state.bayesian.alpha++;
  } else {
    state.bayesian.beta++;
  }
}

// =====================================================
// 11. ALGORITHM 2: MARKOV
// =====================================================

function algoMarkov(state) {
  const recent = state.recent;

  if (!recent.length) return 0.5;

  const previous = recent[recent.length - 1];

  const row = state.markov.transitions[previous];

  const total = row[0] + row[1];

  const rawProbability = total > 0
    ? row[1] / total
    : 0.5;

  // Co xác suất về 50%.
  // Markov chỉ được đóng góp tối đa 20% độ lệch.
  const shrunk =
    0.5 + (rawProbability - 0.5) *
    LEARNING.markovInfluence;

  return shrunk;
}

function updateMarkov(state, actual) {
  const recent = state.recent;

  if (!recent.length) return;

  const previous = recent[recent.length - 1];

  const transitions = state.markov.transitions;
  const prior = LEARNING.markovPrior;
  const decay = LEARNING.markovDecay;

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      transitions[i][j] =
        prior + (transitions[i][j] - prior) * decay;
    }
  }

  transitions[previous][actual]++;
}

// =====================================================
// 12. ALGORITHM 3: LOGISTIC
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
      LEARNING.logisticL2 * weights[i];

    weights[i] +=
      LEARNING.logisticLearningRate * gradient;
  }
}

// =====================================================
// 13. ALGORITHM 4: NEURAL NETWORK
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

  const hidden = forward.hiddenValues;
  const prediction = forward.probability;

  const error = actual - prediction;
  const oldOutput = [...neural.output];

  for (let i = 0; i < neural.output.length; i++) {
    neural.output[i] +=
      LEARNING.neuralLearningRate *
      error *
      hidden[i];
  }

  neural.bias +=
    LEARNING.neuralLearningRate * error;

  for (let h = 0; h < neural.hidden.length; h++) {
    const derivative = 1 - hidden[h] * hidden[h];

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
// 14. ALGORITHM 5: EWMA
// =====================================================

function algoEWMA(state) {
  return state.ewma.value;
}

function updateEWMA(state, actual) {
  const alpha = LEARNING.ewmaAlpha;

  state.ewma.value =
    alpha * actual +
    (1 - alpha) * state.ewma.value;
}

// =====================================================
// 15. INDEPENDENT ALGORITHM PREDICTIONS
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
      return clamp(
        algo.fn(state),
        LEARNING.minProbability,
        LEARNING.maxProbability
      );
    } catch (err) {
      console.error('[ALGO ERROR]', algo.name, err.message);
      return 0.5;
    }
  });
}

// =====================================================
// 16. ENSEMBLE
// =====================================================

function combinePredictions(predictions, weights) {
  let weighted = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    const weight = safeNumber(weights[i], 1);

    weighted += predictions[i] * weight;
    totalWeight += weight;
  }

  if (!totalWeight) return 0.5;

  return weighted / totalWeight;
}

function updateMetaWeights(predictions, actual, state) {
  const weights = state.metaWeights;
  const baselineLoss = Math.log(2);

  for (let i = 0; i < predictions.length; i++) {
    const p = clamp(predictions[i], 0.01, 0.99);

    const loss = actual === 1
      ? -Math.log(p)
      : -Math.log(1 - p);

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
// 17. ONLINE LEARNING
// =====================================================

function learnFromNewResult(actual) {
  const state = learningState;

  const features = getFeatures(state);

  // Tính toàn bộ dự đoán trước khi nhìn kết quả thật.
  const predictions = getAlgoPredictions(state);

  const ensembleP = combinePredictions(
    predictions,
    state.metaWeights
  );

  const predicted = ensembleP >= 0.5 ? 1 : 0;

  // Log Loss thực tế của ensemble
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

  // Meta-learning
  updateMetaWeights(
    predictions,
    actual,
    state
  );

  // Học từng thuật toán
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

  // Chỉ cập nhật context sau khi học
  state.recent.push(actual);

  if (state.recent.length > MAX_CONTEXT) {
    state.recent.shift();
  }

  console.log(
    `[LEARN] Actual=${actual} ` +
    `Prediction=${predicted} ` +
    `Accuracy=${(state.stats.lastAccuracy * 100).toFixed(2)}% ` +
    `LogLoss=${state.stats.logLoss.toFixed(4)}`
  );
}

// =====================================================
// 18. PREDICTION ENGINE
// =====================================================

function ensemblePredict() {
  const state = learningState;

  const predictions = getAlgoPredictions(state);

  const pTai = clamp(
    combinePredictions(
      predictions,
      state.metaWeights
    ),
    LEARNING.minProbability,
    LEARNING.maxProbability
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  const agreement = predictions.filter(
    p => (p >= 0.5) === (pick === 1)
  ).length;

  return {
    pick,
    pTai,

    pXiu: 1 - pTai,

    agreement: `${agreement}/${ALGOS.length}`,

    historicalAccuracy:
      state.stats.lastAccuracy || 0,

    trainedSamples:
      state.stats.total,

    logLoss:
      state.stats.logLoss,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,

      pTai: predictions[index],

      pick: predictions[index] >= 0.5 ? 1 : 0,

      weight: state.metaWeights[index]
    }))
  };
}

// =====================================================
// 19. PENDING SESSION QUEUE
// =====================================================

function addPendingSessions(sessions) {
  const state = learningState;

  const pendingIds = new Set(
    state.pendingSessions.map(s => Number(s.id))
  );

  const processedIds = new Set(
    state.processedIds.map(Number)
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

    state.pendingSessions.push(session);
    pendingIds.add(id);
    added++;
  }

  state.pendingSessions.sort(
    (a, b) => Number(a.id) - Number(b.id)
  );

  return added;
}

function processPendingSessions() {
  while (learningState.pendingSessions.length > 0) {
    const session = learningState.pendingSessions[0];

    const id = Number(session.id);
    const actual = Number(session.result);

    if (
      !Number.isFinite(id) ||
      ![0, 1].includes(actual)
    ) {
      const candidate = JSON.parse(
        JSON.stringify(learningState)
      );

      candidate.pendingSessions.shift();

      atomicSave(LEARNING_FILE, candidate);
      learningState = candidate;

      continue;
    }

    if (
      learningState.processedIds
        .map(Number)
        .includes(id)
    ) {
      const candidate = JSON.parse(
        JSON.stringify(learningState)
      );

      candidate.pendingSessions =
        candidate.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      atomicSave(LEARNING_FILE, candidate);
      learningState = candidate;

      continue;
    }

    const previous = learningState;

    const candidate = JSON.parse(
      JSON.stringify(previous)
    );

    learningState = candidate;

    try {
      learnFromNewResult(actual);

      candidate.processedIds.push(id);

      candidate.processedIds =
        candidate.processedIds.slice(-MAX_PROCESSED_IDS);

      candidate.lastProcessedId =
        candidate.lastProcessedId === null
          ? id
          : Math.max(
              Number(candidate.lastProcessedId),
              id
            );

      candidate.pendingSessions =
        candidate.pendingSessions.filter(
          s => Number(s.id) !== id
        );

      // Lưu model và processedIds cùng một lần.
      atomicSave(LEARNING_FILE, candidate);

      learningState = candidate;

    } catch (err) {
      learningState = previous;
      throw err;
    }
  }
}

// =====================================================
// 20. SYNC ENGINE
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
      throw new Error('Invalid source response');
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

      if (knownIds.has(session.id)) {
        continue;
      }

      knownIds.add(session.id);
      newSessions.push(session);
    }

    // Lần khởi động đầu tiên:
    // chỉ thiết lập mốc, không học lịch sử cũ.
    if (!learningState.initialized) {
      const merged = mergeHistory(
        history,
        newSessions
      );

      writeHistory(merged);

      const latest = merged.length
        ? Math.max(...merged.map(s => Number(s.id)))
        : null;

      learningState.initialized = true;
      learningState.lastProcessedId = latest;

      learningState.pendingSessions = [];
      learningState.processedIds = [];

      saveLearningState();

      console.log('[LEARNING] Baseline established');
      console.log('[LEARNING] Old history not trained');

      return;
    }

    // Lưu queue trước khi ghi history.
    const added = addPendingSessions(newSessions);

    if (added > 0) {
      saveLearningState();
    }

    // Cập nhật lịch sử
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

    // Xử lý queue kể cả khi phiên bị nhận chậm.
    if (learningState.pendingSessions.length > 0) {
      processPendingSessions();
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 21. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  return new Date().toLocaleString('sv-SE', {
    timeZone: 'Asia/Ho_Chi_Minh'
  }).replace(' ', 'T') + '+07:00';
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

    dotincay:
      `${(ens.pTai * 100).toFixed(2)}%`,

    xacsuattai:
      `${(ens.pTai * 100).toFixed(2)}%`,

    xacsuatxiu:
      `${(ens.pXiu * 100).toFixed(2)}%`,

    dochinhxaclichsu:
      `${(ens.historicalAccuracy * 100).toFixed(2)}%`,

    solanhoctap:
      ens.trainedSamples,

    logloss:
      Number(ens.logLoss.toFixed(5)),

    dongthuan:
      ens.agreement,

    trongso: ens.algorithms.map(a => ({
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

// Trạng thái mô hình
app.get('/status', (req, res) => {
  res.json({
    deployId: DEPLOY_ID,
    initialized: learningState.initialized,
    lastProcessedId: learningState.lastProcessedId,
    pending: learningState.pendingSessions.length,
    trainedSamples: learningState.stats.total,
    accuracy:
      Number((learningState.stats.lastAccuracy * 100).toFixed(2)),
    logLoss:
      Number(learningState.stats.logLoss.toFixed(5)),
    recentCount: learningState.recent.length,
    uptime: formatTimeVN()
  });
});

// =====================================================
// 24. BOOT
// =====================================================

handleDeployReset();

initDataFile();

learningState = loadLearningState();

console.log('[BOOT] Deploy:', DEPLOY_ID);
console.log('[BOOT] Bayesian decay enabled');
console.log('[BOOT] Markov shrinkage enabled');
console.log('[BOOT] Streak feature disabled');
console.log('[BOOT] Logistic slow learning enabled');
console.log('[BOOT] Neural slow learning enabled');
console.log('[BOOT] Ensemble Log Loss enabled');
console.log('[BOOT] Persistent learning enabled');

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

const server = app.listen(PORT, () => {
  console.log(`[BOOT] Server running on port ${PORT}`);
  console.log(`[BOOT] Time: ${formatTimeVN()}`);
});

// Graceful shutdown
function shutdown() {
  console.log('[SHUTDOWN] Saving state');

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
