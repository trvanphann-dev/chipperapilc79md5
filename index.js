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

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const PORT = Number(process.env.PORT) || 3000;
const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const MAX_CONTEXT = 100;
const MAX_PROCESSED_IDS = 2000;

const LEARNING = {
  bayesianDecay: 0.995,
  markovDecay: 0.99,
  patternDecay: 0.99,

  logisticRate: 0.012,
  neuralRate: 0.008,
  metaRate: 0.025,
  ewmaAlpha: 0.08,

  minWeight: 0.15,
  maxWeight: 8
};

const ALGO_NAMES = [
  'bayesian_online',
  'markov_pattern',
  'online_logistic',
  'adaptive_neural_network',
  'ewma_online'
];

// =====================================================
// 2. FILE STORAGE
// =====================================================

function atomicSave(file, data) {
  const temp = file + '.tmp';

  fs.writeFileSync(
    temp,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(temp, file);
}

function readHistory() {
  try {
    if (!fs.existsSync(DATA_FILE)) return [];

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

function initFiles() {
  if (!fs.existsSync(DATA_FILE)) {
    writeHistory([]);
    console.log('[INIT] Created history file');
  }

  if (!fs.existsSync(LEARNING_FILE)) {
    atomicSave(LEARNING_FILE, createLearningState());
    console.log('[INIT] Created learning file');
  }
}

// =====================================================
// 3. LEARNING STATE
// =====================================================

function createLearningState() {
  return {
    version: 2,
    initialized: false,

    baselineId: null,
    lastProcessedId: null,

    // Phiên chờ học, giúp tránh mất dữ liệu khi server restart.
    pendingSessions: [],

    // Danh sách phiên đã học.
    processedIds: [],

    // Bayesian.
    bayesian: {
      alpha: 2,
      beta: 2
    },

    // Markov bậc 1.
    markov: {
      counts: [
        [2, 2],
        [2, 2]
      ]
    },

    // Markov bậc 2: hai kết quả trước dự đoán kết quả kế tiếp.
    patterns: {},

    // Logistic.
    logistic: {
      weights: [0, 0, 0, 0, 0]
    },

    // Neural network.
    neural: {
      hidden: [
        [0.12, -0.08, 0.05, 0.10, 0.04],
        [-0.06, 0.11, 0.08, -0.04, 0.06],
        [0.09, 0.04, -0.12, 0.07, -0.05],
        [-0.10, 0.06, 0.03, 0.12, 0.08]
      ],

      output: [0.1, -0.1, 0.08, 0.05],
      bias: 0
    },

    ewma: {
      value: 0.5
    },

    // Trọng số tự điều chỉnh theo Log Loss.
    metaWeights: [1, 1, 1, 1, 1],

    recent: [],

    stats: {
      total: 0,
      hits: 0,
      accuracy: 0,
      averageLogLoss: 0
    }
  };
}

function loadLearningState() {
  try {
    if (!fs.existsSync(LEARNING_FILE)) {
      return createLearningState();
    }

    const saved = JSON.parse(
      fs.readFileSync(LEARNING_FILE, 'utf8')
    );

    if (saved.version !== 2) {
      console.log('[LEARNING] Old version detected. Creating new model.');
      return createLearningState();
    }

    const defaults = createLearningState();

    return {
      ...defaults,
      ...saved,

      bayesian: {
        ...defaults.bayesian,
        ...(saved.bayesian || {})
      },

      markov: {
        ...defaults.markov,
        ...(saved.markov || {})
      },

      logistic: {
        ...defaults.logistic,
        ...(saved.logistic || {})
      },

      neural: {
        ...defaults.neural,
        ...(saved.neural || {})
      },

      ewma: {
        ...defaults.ewma,
        ...(saved.ewma || {})
      },

      stats: {
        ...defaults.stats,
        ...(saved.stats || {})
      }
    };
  } catch (err) {
    console.error('[LEARNING LOAD]', err.message);
    return createLearningState();
  }
}

let learningState;
let syncInProgress = false;
let lastSyncTime = null;
let lastSyncError = null;
let sourceSessionCount = 0;

// =====================================================
// 4. COMMON FUNCTIONS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  x = clamp(x, -25, 25);
  return 1 / (1 + Math.exp(-x));
}

function makePrediction(pTai) {
  const p = clamp(
    Number.isFinite(pTai) ? pTai : 0.5,
    0.001,
    0.999
  );

  return {
    pTai: p,
    pick: p >= 0.5 ? 1 : 0
  };
}

function logLoss(p, actual) {
  p = clamp(p, 0.001, 0.999);

  return actual === 1
    ? -Math.log(p)
    : -Math.log(1 - p);
}

// =====================================================
// 5. NORMALIZE SOURCE DATA
// =====================================================

function normalizeSession(raw) {
  if (!raw || typeof raw !== 'object') return null;

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

  const validDices =
    dices.length === 3 &&
    dices.every(n =>
      Number.isInteger(n) && n >= 1 && n <= 6
    );

  const sum = validDices
    ? dices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? 0);

  if (
    !Number.isFinite(id) ||
    !Number.isFinite(sum) ||
    sum < 3 ||
    sum > 18
  ) {
    return null;
  }

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    raw.big_small ??
    ''
  ).trim().toUpperCase();

  let result;

  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) {
    result = 1;
  } else if (
    ['XIU', 'XỈU', 'SMALL'].includes(sourceResult)
  ) {
    result = 0;
  } else {
    // Xỉu: 3-10. Tài: 11-18.
    result = sum >= 11 ? 1 : 0;
  }

  return {
    id,
    dices: validDices ? dices : [],
    sum,
    result
  };
}

// =====================================================
// 6. FEATURES
// =====================================================

function getFeatures(state) {
  const recent = state.recent;

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

  const alternating =
    recent.length >= 3 &&
    recent[recent.length - 1] !== recent[recent.length - 2] &&
    recent[recent.length - 2] !== recent[recent.length - 3]
      ? 1
      : 0;

  return [
    1,
    last === 1 ? 1 : -1,
    clamp((state.ewma.value - 0.5) * 2, -1, 1),
    clamp(streak / 8, -1, 1),
    alternating
  ];
}

// =====================================================
// 7. MODEL 1: BAYESIAN WITH DECAY
// =====================================================

function algoBayesian(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  return makePrediction(a / (a + b));
}

function updateBayesian(state, actual) {
  const model = state.bayesian;

  model.alpha *= LEARNING.bayesianDecay;
  model.beta *= LEARNING.bayesianDecay;

  if (actual === 1) {
    model.alpha += 1;
  } else {
    model.beta += 1;
  }
}

// =====================================================
// 8. MODEL 2: MARKOV + PATTERN RECOGNITION
// =====================================================

function algoMarkov(state) {
  const recent = state.recent;
  const markov = state.markov;

  let p1 = 0.5;
  let p2 = 0.5;

  if (recent.length >= 1) {
    const previous = recent[recent.length - 1];
    const counts = markov.counts[previous];

    p1 = counts[1] / (counts[0] + counts[1]);
  }

  // Markov bậc 2.
  if (recent.length >= 2) {
    const a = recent[recent.length - 2];
    const b = recent[recent.length - 1];

    const key = `${a}${b}`;
    const counts = state.patterns[key];

    if (counts) {
      p2 = counts[1] / (counts[0] + counts[1]);
    } else {
      p2 = p1;
    }

    // Kết hợp chuyển trạng thái và mẫu hai phiên.
    return makePrediction(
      0.35 * p1 + 0.65 * p2
    );
  }

  return makePrediction(p1);
}

function updateMarkov(state, actual) {
  const recent = state.recent;

  // Giảm ảnh hưởng dữ liệu cũ.
  for (const row of state.markov.counts) {
    row[0] *= LEARNING.markovDecay;
    row[1] *= LEARNING.markovDecay;
  }

  for (const key of Object.keys(state.patterns)) {
    const counts = state.patterns[key];

    counts[0] *= LEARNING.patternDecay;
    counts[1] *= LEARNING.patternDecay;

    if (counts[0] + counts[1] < 0.2) {
      delete state.patterns[key];
    }
  }

  if (recent.length >= 1) {
    const previous = recent[recent.length - 1];
    state.markov.counts[previous][actual] += 1;
  }

  if (recent.length >= 2) {
    const a = recent[recent.length - 2];
    const b = recent[recent.length - 1];
    const key = `${a}${b}`;

    if (!state.patterns[key]) {
      state.patterns[key] = [2, 2];
    }

    state.patterns[key][actual] += 1;
  }
}

// =====================================================
// 9. MODEL 3: LOGISTIC
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

  const error = actual - sigmoid(z);

  for (let i = 0; i < weights.length; i++) {
    // Learning rate nhỏ, có regularization.
    weights[i] +=
      LEARNING.logisticRate *
      (error * features[i] - 0.002 * weights[i]);
  }
}

// =====================================================
// 10. MODEL 4: NEURAL NETWORK
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
    p: sigmoid(output),
    hiddenValues
  };
}

function algoNeural(state) {
  const features = getFeatures(state);
  const output = neuralForward(state, features);

  return makePrediction(output.p);
}

function updateNeural(state, actual, features) {
  const neural = state.neural;
  const forward = neuralForward(state, features);

  const error = actual - forward.p;
  const oldOutput = [...neural.output];

  for (let i = 0; i < neural.output.length; i++) {
    neural.output[i] +=
      LEARNING.neuralRate *
      error *
      forward.hiddenValues[i];
  }

  neural.bias += LEARNING.neuralRate * error;

  for (let h = 0; h < neural.hidden.length; h++) {
    const hiddenValue = forward.hiddenValues[h];

    const hiddenError =
      error *
      oldOutput[h] *
      (1 - hiddenValue * hiddenValue);

    for (let i = 0; i < features.length; i++) {
      neural.hidden[h][i] +=
        LEARNING.neuralRate *
        hiddenError *
        features[i];
    }
  }
}

// =====================================================
// 11. MODEL 5: EWMA
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
// 12. FIVE INDEPENDENT MODELS
// =====================================================

const ALGOS = [
  algoBayesian,
  algoMarkov,
  algoLogistic,
  algoNeural,
  algoEWMA
];

function getAlgoPredictions(state) {
  return ALGOS.map((fn, index) => {
    try {
      return fn(state);
    } catch (err) {
      console.error('[MODEL ERROR]', ALGO_NAMES[index], err.message);
      return makePrediction(0.5);
    }
  });
}

// =====================================================
// 13. ENSEMBLE + LOG LOSS
// =====================================================

function combinePredictions(predictions, weights) {
  let total = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    const weight = weights[i] || 1;

    total += predictions[i].pTai * weight;
    totalWeight += weight;
  }

  return totalWeight > 0
    ? total / totalWeight
    : 0.5;
}

function updateMetaWeights(predictions, actual, state) {
  const weights = state.metaWeights;
  const baseline = Math.log(2);

  for (let i = 0; i < predictions.length; i++) {
    const loss = logLoss(
      predictions[i].pTai,
      actual
    );

    // Mô hình tốt hơn mức ngẫu nhiên sẽ được tăng trọng số tương đối.
    const adjustment = Math.exp(
      -LEARNING.metaRate * (loss - baseline)
    );

    weights[i] = clamp(
      weights[i] * adjustment,
      LEARNING.minWeight,
      LEARNING.maxWeight
    );
  }
}

// =====================================================
// 14. ONLINE LEARNING
// =====================================================

function learnFromNewResult(actual) {
  const state = learningState;

  // Lấy đặc trưng trước khi đưa kết quả mới vào lịch sử.
  const features = getFeatures(state);

  // Dự đoán trước khi biết kết quả.
  const predictions = getAlgoPredictions(state);

  const ensembleP = combinePredictions(
    predictions,
    state.metaWeights
  );

  const predicted = ensembleP >= 0.5 ? 1 : 0;
  const ensembleLoss = logLoss(ensembleP, actual);

  state.stats.total++;

  if (predicted === actual) {
    state.stats.hits++;
  }

  state.stats.accuracy =
    state.stats.hits / state.stats.total;

  const oldLoss = state.stats.averageLogLoss;

  state.stats.averageLogLoss =
    oldLoss +
    (ensembleLoss - oldLoss) / state.stats.total;

  // Tự đánh giá từng mô hình.
  updateMetaWeights(predictions, actual, state);

  // Cập nhật từng mô hình độc lập.
  updateBayesian(state, actual);
  updateMarkov(state, actual);
  updateLogistic(state, actual, features);
  updateNeural(state, actual, features);
  updateEWMA(state, actual);

  state.recent.push(actual);

  if (state.recent.length > MAX_CONTEXT) {
    state.recent.shift();
  }

  console.log(
    `[LEARN] Actual=${actual} | Pred=${predicted} | ` +
    `Accuracy=${(state.stats.accuracy * 100).toFixed(2)}%`
  );
}

// =====================================================
// 15. PREDICTION OUTPUT
// =====================================================

function ensemblePredict() {
  const state = learningState;

  const predictions = getAlgoPredictions(state);

  const pTai = combinePredictions(
    predictions,
    state.metaWeights
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  const agreement = predictions.filter(
    p => (p.pTai >= 0.5) === (pick === 1)
  ).length;

  const margin = Math.abs(pTai - 0.5) * 2;

  // Đây là điểm tham khảo, không phải xác suất thắng đã được bảo đảm.
  const confidenceScore = Math.round(
    clamp(
      50 + margin * 20 + (agreement / 5) * 10,
      50,
      80
    ) * 100
  ) / 100;

  const cautious =
    Math.abs(pTai - 0.5) < 0.06 ||
    agreement <= 2;

  return {
    pick,
    pTai,
    agreement,
    confidenceScore,
    cautious,

    algorithms: ALGOS.map((fn, index) => ({
      name: ALGO_NAMES[index],
      pick: predictions[index].pick,
      pTai: predictions[index].pTai,
      weight: state.metaWeights[index]
    }))
  };
}

// =====================================================
// 16. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  return new Date().toLocaleString('sv-SE', {
    timeZone: 'Asia/Ho_Chi_Minh'
  }).replace(' ', 'T') + '+07:00';
}

// =====================================================
// 17. HISTORY MERGE
// =====================================================

function mergeHistory(oldHistory, newSessions) {
  const map = new Map();

  for (const item of oldHistory) {
    if (Number.isFinite(Number(item.id))) {
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
// 18. QUEUED LEARNING
// =====================================================

function addPendingSessions(sessions) {
  const state = learningState;

  const pendingIds = new Set(
    state.pendingSessions.map(s => Number(s.id))
  );

  const processedIds = new Set(
    state.processedIds.map(Number)
  );

  for (const session of sessions) {
    if (
      session.id <= Number(state.baselineId || 0) ||
      processedIds.has(session.id) ||
      pendingIds.has(session.id)
    ) {
      continue;
    }

    state.pendingSessions.push(session);
    pendingIds.add(session.id);
  }

  state.pendingSessions.sort((a, b) => a.id - b.id);

  // Lưu hàng chờ trước khi xử lý.
  atomicSave(LEARNING_FILE, state);
}

function processPendingSessions() {
  const state = learningState;

  state.pendingSessions.sort((a, b) => a.id - b.id);

  while (state.pendingSessions.length > 0) {
    const session = state.pendingSessions[0];

    if (state.processedIds.includes(session.id)) {
      state.pendingSessions.shift();
      atomicSave(LEARNING_FILE, state);
      continue;
    }

    // Sao lưu để khôi phục nếu lưu file thất bại.
    const backup = JSON.parse(JSON.stringify(state));

    try {
      learnFromNewResult(session.result);

      state.processedIds.push(session.id);

      if (state.processedIds.length > MAX_PROCESSED_IDS) {
        state.processedIds =
          state.processedIds.slice(-MAX_PROCESSED_IDS);
      }

      state.lastProcessedId = Math.max(
        Number(state.lastProcessedId || 0),
        session.id
      );

      state.pendingSessions.shift();

      // Model update + processed ID + queue removal cùng một lần lưu.
      atomicSave(LEARNING_FILE, state);

    } catch (err) {
      learningState = backup;
      throw err;
    }
  }
}

// =====================================================
// 19. SOURCE API SYNC
// =====================================================

async function fetchSource() {
  const response = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(12000)
  });

  if (!response.ok) {
    throw new Error(`Source API HTTP ${response.status}`);
  }

  return response.json();
}

async function syncData() {
  if (syncInProgress) return;

  syncInProgress = true;

  try {
    const raw = await fetchSource();

    const list = Array.isArray(raw)
      ? raw
      : raw.data ?? raw.sessions ?? raw.list ?? [];

    if (!Array.isArray(list)) {
      throw new Error('Source API không trả về danh sách phiên hợp lệ');
    }

    const sessions = list
      .map(normalizeSession)
      .filter(Boolean)
      .sort((a, b) => a.id - b.id);

    sourceSessionCount = sessions.length;

    if (sessions.length === 0) {
      lastSyncTime = formatTimeVN();
      lastSyncError = null;
      return;
    }

    const history = readHistory();

    // Lần đầu: chỉ thiết lập mốc, không huấn luyện lịch sử cũ.
    if (!learningState.initialized) {
      const merged = mergeHistory(history, sessions);

      writeHistory(merged);

      const latestId = Math.max(
        ...merged.map(s => Number(s.id))
      );

      learningState.initialized = true;
      learningState.baselineId = latestId;
      learningState.lastProcessedId = latestId;
      learningState.pendingSessions = [];
      learningState.processedIds = [];

      atomicSave(LEARNING_FILE, learningState);

      console.log('[LEARNING] Baseline initialized; old history not trained');

      lastSyncTime = formatTimeVN();
      lastSyncError = null;
      return;
    }

    // Lưu phiên mới vào hàng chờ trước.
    addPendingSessions(sessions);

    // Lưu lịch sử.
    const merged = mergeHistory(history, sessions);
    writeHistory(merged);

    // Xử lý phiên chờ theo đúng thứ tự ID.
    processPendingSessions();

    lastSyncTime = formatTimeVN();
    lastSyncError = null;

    console.log(
      `[SYNC] History=${merged.length} | Pending=${learningState.pendingSessions.length}`
    );

  } catch (err) {
    lastSyncError = err.message;
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 20. API RESPONSE
// =====================================================

function buildResponse() {
  const history = readHistory();

  if (!history.length) {
    return {
      id: 'noname09',
      error: 'Chưa có dữ liệu, đang chờ đồng bộ',
      uptime: formatTimeVN()
    };
  }

  const current = history[history.length - 1];
  const prediction = ensemblePredict();

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1 ? 'tai' : 'xiu',

    xucxac: current.dices,
    tong: current.sum,

    dudoanphien: current.id + 1,

    dudoan: prediction.pick === 1 ? 'tai' : 'xiu',

    xacsuattai:
      `${(prediction.pTai * 100).toFixed(2)}%`,

    xacsuatxiu:
      `${((1 - prediction.pTai) * 100).toFixed(2)}%`,

    dotincay:
      `${prediction.confidenceScore}%`,

    chedo:
      prediction.cautious
        ? 'THAN_TRONG'
        : 'BINH_THUONG',

    dongthuan:
      `${prediction.agreement}/5`,

    dochinhxaclichsu:
      `${(learningState.stats.accuracy * 100).toFixed(2)}%`,

    logLoss:
      Number(learningState.stats.averageLogLoss.toFixed(5)),

    solanhoctap:
      learningState.stats.total,

    trongso: prediction.algorithms.map(a => ({
      thuattoan: a.name,
      du_doan: a.pick === 1 ? 'tai' : 'xiu',
      xac_suat_tai: Number(a.pTai.toFixed(4)),
      trong_so: Number(a.weight.toFixed(4))
    })),

    uptime: formatTimeVN()
  };
}

// =====================================================
// 21. STATUS ENDPOINT
// =====================================================

function buildStatus() {
  const history = readHistory();

  return {
    status: 'ok',
    service: 'noname09',

    serverTime: formatTimeVN(),

    uptimeSeconds: Math.floor(process.uptime()),

    sync: {
      inProgress: syncInProgress,
      lastSyncTime,
      lastError: lastSyncError,
      sourceSessions: sourceSessionCount,
      pollIntervalMs: POLL_INTERVAL
    },

    files: {
      historyExists: fs.existsSync(DATA_FILE),
      learningExists: fs.existsSync(LEARNING_FILE),
      historyFile: path.basename(DATA_FILE),
      learningFile: path.basename(LEARNING_FILE)
    },

    data: {
      historyCount: history.length,
      latestSessionId: history.length
        ? history[history.length - 1].id
        : null
    },

    learning: {
      initialized: learningState.initialized,
      baselineId: learningState.baselineId,
      lastProcessedId: learningState.lastProcessedId,
      pendingCount: learningState.pendingSessions.length,
      trainedSamples: learningState.stats.total,
      hits: learningState.stats.hits,
      accuracy: Number(
        (learningState.stats.accuracy * 100).toFixed(2)
      ),
      averageLogLoss: Number(
        learningState.stats.averageLogLoss.toFixed(5)
      ),
      algorithms: ALGO_NAMES.map((name, i) => ({
        name,
        weight: Number(
          learningState.metaWeights[i].toFixed(4)
        )
      }))
    }
  };
}

// =====================================================
// 22. ROUTES
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

// Hai đường dẫn trạng thái.
app.get('/status', (req, res) => {
  try {
    res.json(buildStatus());
  } catch (err) {
    res.status(500).json({
      status: 'error',
      error: err.message
    });
  }
});

app.get('/api/status', (req, res) => {
  try {
    res.json(buildStatus());
  } catch (err) {
    res.status(500).json({
      status: 'error',
      error: err.message
    });
  }
});

app.get('/ping', (req, res) => {
  res.type('text').send('pong');
});

app.get('/history', (req, res) => {
  const history = readHistory();

  res.json({
    count: history.length,
    last20: history.slice(-20)
  });
});

// =====================================================
// 23. MANUAL RESET - PROTECTED
// =====================================================

// Trên Render, đặt biến môi trường RESET_TOKEN.
// Gọi POST /admin/reset với header x-reset-token.
// Không tự xóa dữ liệu khi deploy/restart.

app.post('/admin/reset', (req, res) => {
  const expectedToken = process.env.RESET_TOKEN;
  const suppliedToken = req.headers['x-reset-token'];

  if (!expectedToken) {
    return res.status(503).json({
      error: 'Chưa cấu hình RESET_TOKEN trên Render'
    });
  }

  if (suppliedToken !== expectedToken) {
    return res.status(403).json({
      error: 'Sai reset token'
    });
  }

  try {
    learningState = createLearningState();

    writeHistory([]);
    atomicSave(LEARNING_FILE, learningState);

    lastSyncError = null;

    res.json({
      status: 'reset_ok',
      message: 'Đã tạo lại file lịch sử và file trọng số',
      time: formatTimeVN()
    });

    // Đồng bộ lại sau khi reset.
    setTimeout(() => {
      syncData();
    }, 1000);

  } catch (err) {
    console.error('[RESET ERROR]', err.message);

    res.status(500).json({
      error: 'Reset failed',
      detail: err.message
    });
  }
});

// =====================================================
// 24. BOOT
// =====================================================

initFiles();
learningState = loadLearningState();

// Khôi phục hàng chờ nếu server restart giữa lúc học.
try {
  processPendingSessions();
} catch (err) {
  console.error('[PENDING RECOVERY]', err.message);
}

app.listen(PORT, () => {
  console.log(`[BOOT] Server running on port ${PORT}`);
  console.log(`[BOOT] Time: ${formatTimeVN()}`);
  console.log(`[BOOT] Models: ${ALGOS.length}`);
  console.log('[BOOT] /status enabled');
  console.log('[BOOT] /api/status enabled');
  console.log('[BOOT] Online learning enabled');

  syncData();

  const timer = setInterval(syncData, POLL_INTERVAL);

  if (typeof timer.unref === 'function') {
    timer.unref();
  }
});
