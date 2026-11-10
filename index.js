
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

const MAX_HISTORY = 1000;
const MAX_CONTEXT = 100;
const MAX_PROCESSED_IDS = 5000;
const MAX_PENDING = 1000;
const MAX_FORECASTS = 1000;

const CONFIG = {
  globalDecay: 0.998,
  markovDecay: 0.995,
  patternDecay: 0.99,

  logisticRate: 0.008,
  logisticL2: 0.01,

  metaRate: 0.02,
  minWeight: 0.2,
  maxWeight: 5,

  // Markov bậc 2 chỉ được tin nhiều khi có đủ mẫu.
  patternPrior: 10,

  // Giảm xác suất cực đoan khi dữ liệu còn ít.
  probabilityShrinkage: 0.35
};

const ALGO_NAMES = [
  'bayesian',
  'markov1',
  'markov2',
  'logistic'
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
  }

  if (!fs.existsSync(LEARNING_FILE)) {
    atomicSave(LEARNING_FILE, createLearningState());
  }
}

// =====================================================
// 3. LEARNING STATE
// =====================================================

function createLearningState() {
  return {
    version: 3,
    initialized: false,

    baselineId: null,
    lastProcessedId: null,

    processedIds: [],
    pendingSessions: [],

    // Bayesian: alpha = Tài, beta = Xỉu.
    global: {
      alpha: 2,
      beta: 2
    },

    // Markov bậc 1.
    // Hàng là kết quả trước, cột là kết quả tiếp theo.
    markov1: [
      [2, 2],
      [2, 2]
    ],

    // Markov bậc 2.
    patterns: {},

    // Logistic regression.
    logistic: {
      weights: [0, 0, 0, 0, 0, 0]
    },

    recent: [],

    // Trọng số tự điều chỉnh.
    metaWeights: [1, 1, 1, 1],

    // Dự đoán đã phát ra cho phiên tương lai.
    forecasts: [],

    stats: {
      trainedSamples: 0,

      backtestTotal: 0,
      backtestHits: 0,
      backtestAccuracy: 0,
      backtestLogLoss: 0,

      liveTotal: 0,
      liveHits: 0,
      liveAccuracy: 0,
      liveLogLoss: 0
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

    if (saved.version !== 3) {
      console.log('[LEARNING] Reset model state: version mismatch');
      return createLearningState();
    }

    const defaults = createLearningState();

    return {
      ...defaults,
      ...saved,

      global: {
        ...defaults.global,
        ...(saved.global || {})
      },

      logistic: {
        ...defaults.logistic,
        ...(saved.logistic || {})
      },

      stats: {
        ...defaults.stats,
        ...(saved.stats || {})
      },

      metaWeights:
        Array.isArray(saved.metaWeights) &&
        saved.metaWeights.length === 4
          ? saved.metaWeights
          : defaults.metaWeights,

      markov1:
        Array.isArray(saved.markov1) &&
        saved.markov1.length === 2
          ? saved.markov1
          : defaults.markov1,

      patterns: saved.patterns || {},
      recent: saved.recent || [],
      processedIds: saved.processedIds || [],
      pendingSessions: saved.pendingSessions || [],
      forecasts: saved.forecasts || []
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

function sigmoid(value) {
  value = clamp(value, -20, 20);
  return 1 / (1 + Math.exp(-value));
}

function logLoss(probability, actual) {
  const p = clamp(probability, 0.001, 0.999);

  return actual === 1
    ? -Math.log(p)
    : -Math.log(1 - p);
}

function safeNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function resultName(result) {
  return result === 1 ? 'tai' : 'xiu';
}

function formatTimeVN() {
  return new Date().toLocaleString('sv-SE', {
    timeZone: 'Asia/Ho_Chi_Minh'
  }).replace(' ', 'T') + '+07:00';
}

// =====================================================
// 5. NORMALIZE SOURCE DATA
// =====================================================

function normalizeSession(raw) {
  if (!raw || typeof raw !== 'object') return null;

  const id = Number(
    raw.id ??
    raw.session ??
    raw.issue ??
    raw.phien
  );

  let rawDices =
    raw.dices ??
    raw.dice ??
    raw.xucxac ??
    (Array.isArray(raw.result) ? raw.result : []);

  if (typeof rawDices === 'string') {
    try {
      rawDices = JSON.parse(rawDices);
    } catch (_) {
      rawDices = [];
    }
  }

  const dices = Array.isArray(rawDices)
    ? rawDices.map(Number)
    : [];

  const validDices =
    dices.length === 3 &&
    dices.every(n =>
      Number.isInteger(n) && n >= 1 && n <= 6
    );

  let sum = validDices
    ? dices.reduce((a, b) => a + b, 0)
    : Number(
        raw.sum ??
        raw.total ??
        raw.tong ??
        0
      );

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
    raw.big_small_result ??
    raw.ketqua ??
    ''
  ).trim().toUpperCase();

  let result;

  if (
    ['TAI', 'TÀI', 'BIG', '1'].includes(sourceResult)
  ) {
    result = 1;
  } else if (
    ['XIU', 'XỈU', 'SMALL', '0'].includes(sourceResult)
  ) {
    result = 0;
  } else {
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

  const previous = recent.length >= 2
    ? recent[recent.length - 2]
    : 0.5;

  let streak = 0;

  if (recent.length) {
    const current = last;

    for (let i = recent.length - 1; i >= 0; i--) {
      if (recent[i] !== current) break;
      streak++;
    }
  }

  const signedStreak = recent.length
    ? (last === 1 ? 1 : -1) * Math.min(streak, 6) / 6
    : 0;

  const alternating =
    recent.length >= 3 &&
    recent[recent.length - 1] !== recent[recent.length - 2] &&
    recent[recent.length - 2] !== recent[recent.length - 3]
      ? 1
      : 0;

  const recentAverage = recent.length
    ? recent.slice(-10).reduce((a, b) => a + b, 0) /
      Math.min(recent.length, 10)
    : 0.5;

  return [
    1,
    last === 1 ? 1 : -1,
    previous === 1 ? 1 : -1,
    signedStreak,
    alternating,
    (recentAverage - 0.5) * 2
  ];
}

// =====================================================
// 7. MODEL 1: BAYESIAN
// =====================================================

function predictBayesian(state) {
  const total =
    state.global.alpha + state.global.beta;

  return state.global.alpha / total;
}

function updateBayesian(state, actual) {
  const model = state.global;

  model.alpha *= CONFIG.globalDecay;
  model.beta *= CONFIG.globalDecay;

  if (actual === 1) {
    model.alpha += 1;
  } else {
    model.beta += 1;
  }
}

// =====================================================
// 8. MODEL 2: MARKOV BẬC 1
// =====================================================

function predictMarkov1(state) {
  if (!state.recent.length) return 0.5;

  const previous = state.recent[state.recent.length - 1];
  const counts = state.markov1[previous];

  return counts[1] / (counts[0] + counts[1]);
}

function updateMarkov1(state, actual) {
  const recent = state.recent;

  for (const row of state.markov1) {
    row[0] *= CONFIG.markovDecay;
    row[1] *= CONFIG.markovDecay;
  }

  if (!recent.length) return;

  const previous = recent[recent.length - 1];

  state.markov1[previous][actual] += 1;
}

// =====================================================
// 9. MODEL 3: MARKOV BẬC 2 CÓ SHRINKAGE
// =====================================================

function predictMarkov2(state) {
  const recent = state.recent;

  if (recent.length < 2) {
    return predictMarkov1(state);
  }

  const a = recent[recent.length - 2];
  const b = recent[recent.length - 1];

  const key = `${a}${b}`;
  const counts = state.patterns[key];

  const base = predictMarkov1(state);

  if (!counts) return base;

  const total = counts[0] + counts[1];

  const patternProbability =
    counts[1] / total;

  // Ít mẫu thì kéo mạnh về Markov bậc 1.
  const reliability =
    total / (total + CONFIG.patternPrior);

  return (
    reliability * patternProbability +
    (1 - reliability) * base
  );
}

function updateMarkov2(state, actual) {
  for (const key of Object.keys(state.patterns)) {
    const counts = state.patterns[key];

    counts[0] *= CONFIG.patternDecay;
    counts[1] *= CONFIG.patternDecay;

    if (counts[0] + counts[1] < 0.1) {
      delete state.patterns[key];
    }
  }

  const recent = state.recent;

  if (recent.length < 2) return;

  const a = recent[recent.length - 2];
  const b = recent[recent.length - 1];

  const key = `${a}${b}`;

  if (!state.patterns[key]) {
    state.patterns[key] = [4, 4];
  }

  state.patterns[key][actual] += 1;
}

// =====================================================
// 10. MODEL 4: ONLINE LOGISTIC
// =====================================================

function predictLogistic(state) {
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

  const error = actual - sigmoid(z);

  for (let i = 0; i < weights.length; i++) {
    weights[i] += CONFIG.logisticRate * (
      error * features[i] -
      CONFIG.logisticL2 * weights[i]
    );
  }

  // Giới hạn trọng số để tránh mô hình bị bùng.
  for (let i = 0; i < weights.length; i++) {
    weights[i] = clamp(weights[i], -3, 3);
  }
}

// =====================================================
// 11. MODEL ENSEMBLE
// =====================================================

function getModelPredictions(state) {
  return [
    predictBayesian(state),
    predictMarkov1(state),
    predictMarkov2(state),
    predictLogistic(state)
  ].map(p => clamp(p, 0.001, 0.999));
}

function combinePredictions(predictions, weights) {
  let total = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    const weight = clamp(
      safeNumber(weights[i], 1),
      CONFIG.minWeight,
      CONFIG.maxWeight
    );

    total += predictions[i] * weight;
    totalWeight += weight;
  }

  return totalWeight > 0
    ? total / totalWeight
    : 0.5;
}

function updateMetaWeights(predictions, actual, state) {
  const baselineLoss = Math.log(2);

  for (let i = 0; i < predictions.length; i++) {
    const loss = logLoss(predictions[i], actual);

    const adjustment = Math.exp(
      -CONFIG.metaRate * (loss - baselineLoss)
    );

    state.metaWeights[i] = clamp(
      state.metaWeights[i] * adjustment,
      CONFIG.minWeight,
      CONFIG.maxWeight
    );
  }
}

function getEnsembleProbability(state) {
  const predictions = getModelPredictions(state);

  const rawProbability = combinePredictions(
    predictions,
    state.metaWeights
  );

  // Khi chưa có nhiều mẫu, không cho mô hình quá tự tin.
  const samples = state.stats.trainedSamples;

  const learningFactor = clamp(
    samples / 150,
    0,
    1
  );

  const shrinkage =
    CONFIG.probabilityShrinkage *
    (1 - learningFactor);

  const calibrated = (
    rawProbability * (1 - shrinkage) +
    0.5 * shrinkage
  );

  return {
    pTai: clamp(calibrated, 0.05, 0.95),
    modelPredictions: predictions
  };
}

// =====================================================
// 12. STATS
// =====================================================

function updateBacktestStats(state, probability, actual) {
  const stats = state.stats;

  stats.backtestTotal++;

  if (
    (probability >= 0.5 ? 1 : 0) === actual
  ) {
    stats.backtestHits++;
  }

  stats.backtestAccuracy =
    stats.backtestHits / stats.backtestTotal;

  const loss = logLoss(probability, actual);

  stats.backtestLogLoss += (
    loss - stats.backtestLogLoss
  ) / stats.backtestTotal;
}

function updateLiveStats(state, probability, actual) {
  const stats = state.stats;

  stats.liveTotal++;

  if (
    (probability >= 0.5 ? 1 : 0) === actual
  ) {
    stats.liveHits++;
  }

  stats.liveAccuracy =
    stats.liveHits / stats.liveTotal;

  const loss = logLoss(probability, actual);

  stats.liveLogLoss += (
    loss - stats.liveLogLoss
  ) / stats.liveTotal;
}

// =====================================================
// 13. LEARN ONE RESULT
// =====================================================

function learnOneResult(actual) {
  const state = learningState;

  // Phải lấy feature và dự đoán trước khi cập nhật kết quả.
  const features = getFeatures(state);
  const predictions = getModelPredictions(state);

  const rawProbability = combinePredictions(
    predictions,
    state.metaWeights
  );

  const samples = state.stats.trainedSamples;

  const learningFactor = clamp(samples / 150, 0, 1);

  const shrinkage =
    CONFIG.probabilityShrinkage *
    (1 - learningFactor);

  const probability =
    rawProbability * (1 - shrinkage) +
    0.5 * shrinkage;

  // Backtest tuần tự: dự đoán trước, biết kết quả sau.
  updateBacktestStats(state, probability, actual);

  // Điều chỉnh trọng số theo kết quả vừa biết.
  updateMetaWeights(predictions, actual, state);

  // Cập nhật các mô hình.
  updateBayesian(state, actual);
  updateMarkov1(state, actual);
  updateMarkov2(state, actual);
  updateLogistic(state, actual, features);

  state.recent.push(actual);

  if (state.recent.length > MAX_CONTEXT) {
    state.recent.shift();
  }

  state.stats.trainedSamples++;
}

// =====================================================
// 14. FORECAST LEDGER
// =====================================================

function findForecast(sessionId) {
  return learningState.forecasts.find(
    item => Number(item.targetId) === Number(sessionId)
  );
}

function saveForecast(targetId, prediction) {
  const state = learningState;

  // Không ghi đè dự đoán cũ của cùng một phiên.
  if (findForecast(targetId)) return;

  state.forecasts.push({
    targetId: Number(targetId),
    pTai: prediction.pTai,
    pick: prediction.pick,
    createdAt: formatTimeVN(),
    evaluated: false
  });

  if (state.forecasts.length > MAX_FORECASTS) {
    state.forecasts = state.forecasts.slice(-MAX_FORECASTS);
  }

  atomicSave(LEARNING_FILE, state);
}

function evaluateForecast(session) {
  const forecast = findForecast(session.id);

  if (!forecast || forecast.evaluated) return;

  updateLiveStats(
    learningState,
    forecast.pTai,
    session.result
  );

  forecast.evaluated = true;
  forecast.actual = session.result;
  forecast.correct = forecast.pick === session.result;
  forecast.evaluatedAt = formatTimeVN();

  console.log(
    `[LIVE SCORE] Session=${session.id} ` +
    `Pred=${resultName(forecast.pick)} ` +
    `Actual=${resultName(session.result)} ` +
    `Correct=${forecast.correct}`
  );
}

// =====================================================
// 15. HISTORY
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
// 16. QUEUE
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
      processedIds.has(session.id) ||
      pendingIds.has(session.id)
    ) {
      continue;
    }

    state.pendingSessions.push(session);
    pendingIds.add(session.id);
  }

  state.pendingSessions.sort((a, b) => a.id - b.id);

  if (state.pendingSessions.length > MAX_PENDING) {
    state.pendingSessions =
      state.pendingSessions.slice(-MAX_PENDING);
  }

  atomicSave(LEARNING_FILE, state);
}

function processPendingSessions() {
  const state = learningState;

  state.pendingSessions.sort((a, b) => a.id - b.id);

  while (state.pendingSessions.length) {
    const session = state.pendingSessions[0];

    if (
      state.processedIds.includes(Number(session.id))
    ) {
      state.pendingSessions.shift();
      atomicSave(LEARNING_FILE, state);
      continue;
    }

    // Đánh giá dự đoán thực tế trước khi học kết quả.
    evaluateForecast(session);

    // Học kết quả mới theo thứ tự thời gian.
    learnOneResult(session.result);

    state.processedIds.push(Number(session.id));

    if (state.processedIds.length > MAX_PROCESSED_IDS) {
      state.processedIds =
        state.processedIds.slice(-MAX_PROCESSED_IDS);
    }

    state.lastProcessedId = Number(session.id);

    state.pendingSessions.shift();

    atomicSave(LEARNING_FILE, state);
  }
}

// =====================================================
// 17. WARMUP FROM HISTORY
// =====================================================

function warmupFromHistory(sessions) {
  const state = learningState;

  console.log(
    `[WARMUP] Training from ${sessions.length} historical sessions`
  );

  // Reset model trước khi học lại.
  const fresh = createLearningState();

  Object.assign(state, fresh);

  const sorted = [...sessions].sort(
    (a, b) => a.id - b.id
  );

  for (const session of sorted) {
    learnOneResult(session.result);

    state.processedIds.push(Number(session.id));
    state.lastProcessedId = Number(session.id);
  }

  state.initialized = true;

  state.baselineId = sorted.length
    ? sorted[0].id
    : null;

  state.pendingSessions = [];

  atomicSave(LEARNING_FILE, state);

  console.log(
    `[WARMUP DONE] Samples=${state.stats.trainedSamples}`
  );
}

// =====================================================
// 18. SOURCE API
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
    throw new Error(
      `Source API HTTP ${response.status}`
    );
  }

  return response.json();
}

function extractSourceList(raw) {
  if (Array.isArray(raw)) return raw;

  if (Array.isArray(raw.data)) return raw.data;
  if (Array.isArray(raw.sessions)) return raw.sessions;
  if (Array.isArray(raw.list)) return raw.list;
  if (Array.isArray(raw.items)) return raw.items;

  if (raw.data && Array.isArray(raw.data.sessions)) {
    return raw.data.sessions;
  }

  return [];
}

async function syncData() {
  if (syncInProgress) return;

  syncInProgress = true;

  try {
    const raw = await fetchSource();
    const list = extractSourceList(raw);

    if (!list.length) {
      throw new Error('Source API không trả về danh sách phiên');
    }

    const sessions = list
      .map(normalizeSession)
      .filter(Boolean)
      .sort((a, b) => a.id - b.id);

    sourceSessionCount = sessions.length;

    if (!sessions.length) {
      throw new Error('Không có phiên hợp lệ sau khi chuẩn hóa');
    }

    const history = readHistory();
    const merged = mergeHistory(history, sessions);

    // Lần đầu hoặc model mới: học lại từ lịch sử đã có.
    if (!learningState.initialized) {
      writeHistory(merged);
      warmupFromHistory(merged);

      lastSyncTime = formatTimeVN();
      lastSyncError = null;
      return;
    }

    // Lưu hàng chờ trước khi xử lý.
    addPendingSessions(sessions);

    // Lưu lịch sử.
    writeHistory(merged);

    // Học các phiên mới.
    processPendingSessions();

    lastSyncTime = formatTimeVN();
    lastSyncError = null;

    console.log(
      `[SYNC] History=${merged.length} ` +
      `Trained=${learningState.stats.trainedSamples} ` +
      `Pending=${learningState.pendingSessions.length}`
    );

  } catch (err) {
    lastSyncError = err.message;
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 19. PREDICTION
// =====================================================

function ensemblePredict() {
  const state = learningState;

  const result = getEnsembleProbability(state);

  const pTai = result.pTai;
  const pXiu = 1 - pTai;

  const pick = pTai >= 0.5 ? 1 : 0;

  const predictions = result.modelPredictions;

  const agreement = predictions.filter(
    p => (p >= 0.5) === (pick === 1)
  ).length;

  const probabilityGap = Math.abs(pTai - 0.5);

  const cautious =
    probabilityGap < 0.05 ||
    agreement <= 2 ||
    state.stats.trainedSamples < 50;

  let level = 'THAP';

  if (
    !cautious &&
    probabilityGap >= 0.10 &&
    state.stats.trainedSamples >= 150
  ) {
    level = 'TRUNG_BINH';
  }

  return {
    pick,
    pTai,
    pXiu,
    agreement,
    cautious,
    level,

    algorithms: ALGO_NAMES.map((name, index) => ({
      name,
      pTai: predictions[index],
      pick: predictions[index] >= 0.5 ? 1 : 0,
      weight: state.metaWeights[index]
    }))
  };
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

  const targetId = Number(current.id) + 1;

  // Lưu dự đoán đầu tiên cho phiên kế tiếp.
  saveForecast(targetId, prediction);

  return {
    id: 'noname09',

    phienhientai: current.id,
    ketqua: resultName(current.result),

    xucxac: current.dices,
    tong: current.sum,

    dudoanphien: targetId,
    dudoan: resultName(prediction.pick),

    xacsuattai:
      `${(prediction.pTai * 100).toFixed(2)}%`,

    xacsuatxiu:
      `${(prediction.pXiu * 100).toFixed(2)}%`,

    // Đây là xác suất mô hình ước lượng,
    // không phải tỷ lệ thắng được đảm bảo.
    dotincay:
      `${(Math.max(prediction.pTai, prediction.pXiu) * 100).toFixed(2)}%`,

    mucdo: prediction.level,

    chedo: prediction.cautious
      ? 'THAN_TRONG'
      : 'BINH_THUONG',

    dongthuan: `${prediction.agreement}/4`,

    dochinhxacbacktest:
      `${(learningState.stats.backtestAccuracy * 100).toFixed(2)}%`,

    dochinhxacthucte:
      `${(learningState.stats.liveAccuracy * 100).toFixed(2)}%`,

    solanhoc:
      learningState.stats.trainedSamples,

    soLanDuDoanDaCham:
      learningState.stats.liveTotal,

    logLossBacktest:
      Number(
        learningState.stats.backtestLogLoss.toFixed(5)
      ),

    logLossThucTe:
      Number(
        learningState.stats.liveLogLoss.toFixed(5)
      ),

    trongso: prediction.algorithms.map(a => ({
      thuattoan: a.name,
      du_doan: resultName(a.pick),
      xac_suat_tai: Number(a.pTai.toFixed(4)),
      trong_so: Number(a.weight.toFixed(4))
    })),

    luuy:
      'Xác suất chỉ mang tính tham khảo; kết quả xúc xắc có thể độc lập và ngẫu nhiên.',

    uptime: formatTimeVN()
  };
}

// =====================================================
// 21. STATUS
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
      version: learningState.version,
      initialized: learningState.initialized,
      baselineId: learningState.baselineId,
      lastProcessedId: learningState.lastProcessedId,

      pendingCount:
        learningState.pendingSessions.length,

      trainedSamples:
        learningState.stats.trainedSamples,

      backtest: {
        total: learningState.stats.backtestTotal,
        hits: learningState.stats.backtestHits,
        accuracy: Number(
          (learningState.stats.backtestAccuracy * 100).toFixed(2)
        ),
        logLoss: Number(
          learningState.stats.backtestLogLoss.toFixed(5)
        )
      },

      live: {
        total: learningState.stats.liveTotal,
        hits: learningState.stats.liveHits,
        accuracy: Number(
          (learningState.stats.liveAccuracy * 100).toFixed(2)
        ),
        logLoss: Number(
          learningState.stats.liveLogLoss.toFixed(5)
        )
      },

      forecasts: {
        total: learningState.forecasts.length,
        evaluated: learningState.forecasts.filter(
          f => f.evaluated
        ).length
      },

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
    res.status(500).json({
      error: 'Internal server error'
    });
  }
});

app.get('/predict', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[PREDICT]', err.message);
    res.status(500).json({
      error: 'Prediction error'
    });
  }
});

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

app.get('/forecast-history', (req, res) => {
  res.json({
    count: learningState.forecasts.length,
    forecasts: learningState.forecasts.slice(-100)
  });
});

// =====================================================
// 23. MANUAL RESET
// =====================================================

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
      message: 'Đã tạo lại lịch sử và mô hình',
      time: formatTimeVN()
    });

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

// Khôi phục hàng chờ còn dang dở.
try {
  processPendingSessions();
} catch (err) {
  console.error('[PENDING RECOVERY]', err.message);
}

app.listen(PORT, () => {
  console.log(`[BOOT] Server running on port ${PORT}`);
  console.log(`[BOOT] Time: ${formatTimeVN()}`);
  console.log(`[BOOT] Models: ${ALGO_NAMES.join(', ')}`);
  console.log('[BOOT] /status enabled');
  console.log('[BOOT] /api/status enabled');
  console.log('[BOOT] Forecast ledger enabled');

  syncData();

  const timer = setInterval(syncData, POLL_INTERVAL);

  if (typeof timer.unref === 'function') {
    timer.unref();
  }
});
