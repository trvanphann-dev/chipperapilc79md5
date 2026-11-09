
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const LEARNING_FILE = path.join(__dirname, 'learning_state.json');

const SOURCE_API = 'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const MAX_CONTEXT = 64;
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

function writeHistory(arr) {
  const tempFile = DATA_FILE + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(arr, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, DATA_FILE);
}

function atomicSave(file, data) {
  const tempFile = file + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, file);
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

// =====================================================
// 3. LEARNING CONFIGURATION
// =====================================================

const LEARNING = {
  initialWeight: 1,
  minWeight: 0.1,
  maxWeight: 10,

  metaLearningRate: 0.08,
  logisticLearningRate: 0.035,
  neuralLearningRate: 0.025,

  ewmaAlpha: 0.15,
  monteCarloSamples: 256
};

const ALGO_NAMES = [
  'bayesian_online',
  'monte_carlo',
  'online_logistic',
  'adaptive_neural_network',
  'ewma_online'
];

// =====================================================
// 4. INITIAL LEARNING STATE
// =====================================================

function createLearningState() {
  return {
    initialized: false,
    lastProcessedId: null,

    version: 1,

    // Bayesian model
    bayesian: {
      alpha: 2,
      beta: 2
    },

    // Online logistic model
    logistic: {
      weights: [0, 0, 0, 0]
    },

    // Adaptive neural network
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

    // Ensemble meta-learning
    metaWeights: [
      1, 1, 1, 1, 1
    ],

    // Compact context from NEW observations only
    recent: [],

    stats: {
      total: 0,
      hits: 0,
      lastAccuracy: 0
    }
  };
}

function loadLearningState() {
  try {
    if (!fs.existsSync(LEARNING_FILE)) {
      return createLearningState();
    }

    const parsed = JSON.parse(
      fs.readFileSync(LEARNING_FILE, 'utf8')
    );

    if (
      parsed.version !== 1 ||
      !Array.isArray(parsed.metaWeights) ||
      !Array.isArray(parsed.recent)
    ) {
      console.error('[LEARNING] Invalid state, initializing');
      return createLearningState();
    }

    return parsed;

  } catch (err) {
    console.error('[LEARNING LOAD]', err.message);
    return createLearningState();
  }
}

function saveLearningState() {
  atomicSave(LEARNING_FILE, learningState);
}

let learningState = null;

// =====================================================
// 5. COMMON FUNCTIONS
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
// 6. FEATURE ENGINEERING
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

  return [
    1,
    last * 2 - 1,
    (state.ewma.value - 0.5) * 2,
    clamp(streak / 10, -1, 1)
  ];
}

// =====================================================
// 7. ALGORITHM 1: BAYESIAN ONLINE
// =====================================================

function algoBayesian(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  return makePrediction(
    a / (a + b)
  );
}

// =====================================================
// 8. ALGORITHM 2: MONTE CARLO
// =====================================================

function randomGamma(shape) {
  if (shape < 1) {
    return randomGamma(shape + 1) *
      Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x, v;

    do {
      const u1 = Math.random();
      const u2 = Math.random();

      x = Math.sqrt(-2 * Math.log(u1)) *
        Math.cos(2 * Math.PI * u2);

      v = 1 + c * x;

    } while (v <= 0);

    v = v * v * v;

    const u = Math.random();

    if (
      u < 1 - 0.0331 * x * x * x * x ||
      Math.log(u) <
      0.5 * x * x + d * (1 - v + Math.log(v))
    ) {
      return d * v;
    }
  }
}

function algoMonteCarlo(state) {
  const a = state.bayesian.alpha;
  const b = state.bayesian.beta;

  let total = 0;

  for (
    let i = 0;
    i < LEARNING.monteCarloSamples;
    i++
  ) {
    const x = randomGamma(a);
    const y = randomGamma(b);

    total += x / (x + y);
  }

  return makePrediction(
    total / LEARNING.monteCarloSamples
  );
}

// =====================================================
// 9. ALGORITHM 3: ONLINE LOGISTIC LEARNING
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
// 10. ALGORITHM 4: ADAPTIVE NEURAL NETWORK
// =====================================================

function algoNeural(state) {
  const features = getFeatures(state);
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
    output +=
      hiddenValues[i] * neural.output[i];
  }

  return makePrediction(sigmoid(output));
}

function updateNeural(state, actual, features) {
  const neural = state.neural;

  const hiddenValues = [];
  const hiddenDerivatives = [];

  for (const row of neural.hidden) {
    let sum = 0;

    for (let i = 0; i < features.length; i++) {
      sum += row[i] * features[i];
    }

    const h = Math.tanh(sum);

    hiddenValues.push(h);
    hiddenDerivatives.push(1 - h * h);
  }

  let output = neural.bias;

  for (let i = 0; i < hiddenValues.length; i++) {
    output +=
      hiddenValues[i] * neural.output[i];
  }

  const prediction = sigmoid(output);
  const error = actual - prediction;

  const oldOutput = [...neural.output];

  // Update output layer
  for (let i = 0; i < neural.output.length; i++) {
    neural.output[i] +=
      LEARNING.neuralLearningRate *
      error *
      hiddenValues[i];
  }

  neural.bias +=
    LEARNING.neuralLearningRate * error;

  // Update hidden layer
  for (let h = 0; h < neural.hidden.length; h++) {
    const hiddenError =
      error *
      oldOutput[h] *
      hiddenDerivatives[h];

    for (
      let i = 0;
      i < features.length;
      i++
    ) {
      neural.hidden[h][i] +=
        LEARNING.neuralLearningRate *
        hiddenError *
        features[i];
    }
  }
}

// =====================================================
// 11. ALGORITHM 5: EWMA ONLINE
// =====================================================

function algoEWMA(state) {
  return makePrediction(
    state.ewma.value
  );
}

function updateEWMA(state, actual) {
  const alpha = LEARNING.ewmaAlpha;

  state.ewma.value =
    alpha * actual +
    (1 - alpha) * state.ewma.value;
}

// =====================================================
// 12. REGISTER FIVE ALGORITHMS
// =====================================================

const ALGOS = [
  { name: ALGO_NAMES[0], fn: algoBayesian },
  { name: ALGO_NAMES[1], fn: algoMonteCarlo },
  { name: ALGO_NAMES[2], fn: algoLogistic },
  { name: ALGO_NAMES[3], fn: algoNeural },
  { name: ALGO_NAMES[4], fn: algoEWMA }
];

function getAlgoPredictions(state) {
  return ALGOS.map(algo => {
    try {
      return algo.fn(state);
    } catch (err) {
      console.error(
        '[ALGO ERROR]',
        algo.name,
        err.message
      );

      return makePrediction(0.5);
    }
  });
}

// =====================================================
// 13. ENSEMBLE META-LEARNING
// =====================================================

function combinePredictions(predictions, weights) {
  let total = 0;
  let weightTotal = 0;

  for (let i = 0; i < predictions.length; i++) {
    const weight = weights[i];

    total += predictions[i].pTai * weight;
    weightTotal += weight;
  }

  return weightTotal
    ? total / weightTotal
    : 0.5;
}

function updateMetaWeights(predictions, actual, state) {
  const weights = state.metaWeights;

  for (let i = 0; i < predictions.length; i++) {
    const error =
      predictions[i].pTai - actual;

    const loss = error * error;

    weights[i] = clamp(
      weights[i] * Math.exp(
        -LEARNING.metaLearningRate * loss
      ),
      LEARNING.minWeight,
      LEARNING.maxWeight
    );
  }
}

// =====================================================
// 14. ONLINE LEARNING ENGINE
// =====================================================

// This function processes each NEW result exactly once.
// It never reads or loops through old history.

function learnFromNewResult(actual) {
  const state = learningState;

  const features = getFeatures(state);

  // Prediction made BEFORE observing actual result.
  const predictions = getAlgoPredictions(state);

  const ensembleP = combinePredictions(
    predictions,
    state.metaWeights
  );

  const predicted = ensembleP >= 0.5 ? 1 : 0;

  // Online performance tracking.
  state.stats.total++;

  if (predicted === actual) {
    state.stats.hits++;
  }

  state.stats.lastAccuracy =
    state.stats.total
      ? state.stats.hits / state.stats.total
      : 0;

  // Update ensemble weights.
  updateMetaWeights(
    predictions,
    actual,
    state
  );

  // Update Bayesian posterior.
  if (actual === 1) {
    state.bayesian.alpha++;
  } else {
    state.bayesian.beta++;
  }

  // Update logistic model.
  updateLogistic(
    state,
    actual,
    features
  );

  // Update neural network.
  updateNeural(
    state,
    actual,
    features
  );

  // Update EWMA.
  updateEWMA(
    state,
    actual
  );

  // Store compact online context.
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
// 15. MAIN PREDICTION ENGINE
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

  const accuracy = state.stats.lastAccuracy;

  // Conservative display metric.
  // This is not a guaranteed win probability.
  const confidence = clamp(
    50 + Math.max(0, accuracy - 0.5) * 50,
    50,
    75
  );

  return {
    pick,

    confidence:
      Math.round(confidence * 100) / 100,

    pTai:
      Math.round(pTai * 10000) / 10000,

    agreement:
      `${agreement}/${ALGOS.length}`,

    historicalAccuracy:
      Math.round(accuracy * 10000) / 100,

    trainedSamples: state.stats.total,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,

      pick: predictions[index].pick,

      probabilityTai:
        Math.round(
          predictions[index].pTai * 10000
        ) / 10000,

      weight:
        Math.round(
          state.metaWeights[index] * 10000
        ) / 10000
    }))
  };
}

// =====================================================
// 16. VIETNAM TIME
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
// 17. SOURCE DATA SYNC + ONLINE LEARNING
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
      history.map(h => h.id)
    );

    const newSessions = [];

    for (const item of list) {
      const s = normalizeSession(item);

      if (
        !Number.isFinite(s.id) ||
        !Number.isFinite(s.sum) ||
        ![0, 1].includes(s.result) ||
        knownIds.has(s.id)
      ) {
        continue;
      }

      history.push(s);
      knownIds.add(s.id);

      newSessions.push(s);
    }

    if (newSessions.length > 0) {
      history.sort((a, b) => a.id - b.id);

      const saved = history.slice(-MAX_HISTORY);

      writeHistory(saved);

      console.log(
        `[SYNC] +${newSessions.length} sessions | Total ${saved.length}`
      );
    }

    // First startup:
    // Establish a baseline without training from old history.
    if (!learningState.initialized) {
      const latest = history.length
        ? Math.max(...history.map(h => h.id))
        : null;

      learningState.initialized = true;
      learningState.lastProcessedId = latest;

      saveLearningState();

      console.log(
        '[LEARNING] Fresh online learning initialized'
      );

      console.log(
        '[LEARNING] Existing history was NOT used for training'
      );

      return;
    }

    // Process only newly received sessions.
    const pending = newSessions
      .filter(s =>
        learningState.lastProcessedId === null ||
        s.id > learningState.lastProcessedId
      )
      .sort((a, b) => a.id - b.id);

    for (const session of pending) {
      learnFromNewResult(session.result);

      learningState.lastProcessedId = session.id;
    }

    if (pending.length > 0) {
      saveLearningState();

      console.log(
        `[LEARNING] Processed ${pending.length} new results`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 18. API RESPONSE
// =====================================================

function buildResponse() {
  const history = readHistory();

  if (history.length === 0) {
    return {
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime: formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  // Prediction uses persistent online state.
  // No historical backtesting.
  const ens = ensemblePredict();

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: current.id + 1,

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
// 19. API ROUTES
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
// 20. BOOT SERVER
// =====================================================

initDataFile();

learningState = loadLearningState();

// Initial synchronization.
syncData();

// Periodic synchronization.
const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

app.listen(PORT, () => {
  console.log(
    `[BOOT] Port ${PORT} — ${formatTimeVN()}`
  );

  console.log(
    `[BOOT] ${ALGOS.length} algorithms loaded`
  );

  console.log(
    '[BOOT] Persistent online learning enabled'
  );

  console.log(
    `[BOOT] Learning state: ${LEARNING_FILE}`
  );
});
