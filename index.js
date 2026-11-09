
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const SOURCE_API = 'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

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

function writeHistory(arr) {
  const tempFile = DATA_FILE + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(arr, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, DATA_FILE);
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
    const knownIds = new Set(history.map(h => h.id));

    let added = 0;

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
      added++;
    }

    if (added > 0) {
      history.sort((a, b) => a.id - b.id);

      const saved = history.slice(-MAX_HISTORY);

      writeHistory(saved);

      console.log(
        `[SYNC] +${added} sessions | Total ${saved.length}`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 3. COMMON STATISTICS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mean(seq, fallback = 0.5) {
  if (!seq.length) return fallback;

  return seq.reduce(
    (sum, value) => sum + value,
    0
  ) / seq.length;
}

function smoothedRate(seq, alpha = 2, beta = 2) {
  const tai = seq.reduce(
    (sum, value) => sum + value,
    0
  );

  return (tai + alpha) /
    (seq.length + alpha + beta);
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

function weightedRate(values, weights) {
  let total = 0;
  let totalWeight = 0;

  for (let i = 0; i < values.length; i++) {
    total += values[i] * weights[i];
    totalWeight += weights[i];
  }

  return totalWeight
    ? total / totalWeight
    : 0.5;
}

// =====================================================
// 4. ALGORITHM 1: BAYESIAN ADAPTIVE
// =====================================================

function algoBayesian(seq) {
  return makePrediction(
    smoothedRate(seq, 2, 2)
  );
}

// =====================================================
// 5. ALGORITHM 2: MARKOV CHAIN
// =====================================================

function algoMarkov(seq) {
  if (seq.length < 2) {
    return makePrediction(0.5);
  }

  const current = seq[seq.length - 1];
  const next = [];

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] === current) {
      next.push(seq[i]);
    }
  }

  return makePrediction(
    smoothedRate(next, 2, 2)
  );
}

// =====================================================
// 6. ALGORITHM 3: HIGHER ORDER MARKOV
// =====================================================

function algoHigherMarkov(seq) {
  if (seq.length < 3) {
    return makePrediction(0.5);
  }

  const a = seq[seq.length - 2];
  const b = seq[seq.length - 1];

  const next = [];

  for (let i = 2; i < seq.length; i++) {
    if (
      seq[i - 2] === a &&
      seq[i - 1] === b
    ) {
      next.push(seq[i]);
    }
  }

  return makePrediction(
    smoothedRate(next, 2, 2)
  );
}

// =====================================================
// 7. ALGORITHM 4: RUN LENGTH
// =====================================================

function algoRunLength(seq) {
  if (seq.length < 3) {
    return makePrediction(0.5);
  }

  const current = seq[seq.length - 1];

  let run = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] !== current) break;
    run++;
  }

  const continuations = [];

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] !== current) continue;

    let priorRun = 1;

    for (let j = i - 2; j >= 0; j--) {
      if (seq[j] !== current) break;
      priorRun++;
    }

    if (priorRun === run) {
      continuations.push(
        seq[i] === current ? 1 : 0
      );
    }
  }

  const pSame = smoothedRate(
    continuations,
    2,
    2
  );

  return makePrediction(
    current === 1 ? pSame : 1 - pSame
  );
}

// =====================================================
// 8. ALGORITHM 5: EWMA
// =====================================================

function algoEWMA(seq) {
  if (!seq.length) {
    return makePrediction(0.5);
  }

  const alpha = 0.22;

  let value = 0.5;

  for (const result of seq) {
    value =
      alpha * result +
      (1 - alpha) * value;
  }

  return makePrediction(value);
}

// =====================================================
// 9. ALGORITHM 6: MOMENTUM
// =====================================================

function algoMomentum(seq) {
  const recent = seq.slice(-30);

  if (recent.length < 3) {
    return makePrediction(0.5);
  }

  const weights = recent.map(
    (_, i) => i + 1
  );

  return makePrediction(
    weightedRate(recent, weights)
  );
}

// =====================================================
// 10. ALGORITHM 7: FREQUENCY WINDOW
// =====================================================

function algoFrequencyWindow(seq) {
  if (!seq.length) {
    return makePrediction(0.5);
  }

  const p7 = smoothedRate(seq.slice(-7));
  const p20 = smoothedRate(seq.slice(-20));
  const p50 = smoothedRate(seq.slice(-50));

  return makePrediction(
    p7 * 0.5 +
    p20 * 0.3 +
    p50 * 0.2
  );
}

// =====================================================
// 11. ALGORITHM 8: TRANSITION PATTERN
// =====================================================

function algoTransitionPattern(seq) {
  if (seq.length < 4) {
    return makePrediction(0.5);
  }

  const pattern = seq.slice(-3).join('');
  const matches = [];

  for (let i = 3; i < seq.length; i++) {
    if (
      seq.slice(i - 3, i).join('') === pattern
    ) {
      matches.push(seq[i]);
    }
  }

  return makePrediction(
    smoothedRate(matches, 2, 2)
  );
}

// =====================================================
// 12. ALGORITHM 9: STREAK HAZARD
// =====================================================

function algoStreakHazard(seq) {
  if (seq.length < 3) {
    return makePrediction(0.5);
  }

  const current = seq[seq.length - 1];

  let run = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] !== current) break;
    run++;
  }

  let switches = 0;
  let observations = 0;

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] !== current) continue;

    let length = 1;

    for (let j = i - 2; j >= 0; j--) {
      if (seq[j] !== current) break;
      length++;
    }

    if (length === run) {
      observations++;

      if (seq[i] !== current) {
        switches++;
      }
    }
  }

  const pSwitch =
    (switches + 1) / (observations + 2);

  const pSame = 1 - pSwitch;

  return makePrediction(
    current === 1 ? pSame : 1 - pSame
  );
}

// =====================================================
// 13. ALGORITHM 10: PATTERN SIMILARITY
// =====================================================

function algoPatternSimilarity(seq) {
  const size = 4;

  if (seq.length <= size) {
    return makePrediction(0.5);
  }

  const pattern = seq.slice(-size).join('');
  const matches = [];

  for (let i = size; i < seq.length; i++) {
    if (
      seq.slice(i - size, i).join('') === pattern
    ) {
      matches.push(seq[i]);
    }
  }

  return makePrediction(
    smoothedRate(matches, 2, 2)
  );
}

// =====================================================
// 14. REGISTER ALGORITHMS
// =====================================================

const ALGOS = [
  { name: 'bayesian_adaptive', fn: algoBayesian },
  { name: 'markov_order_1', fn: algoMarkov },
  { name: 'higher_order_markov', fn: algoHigherMarkov },
  { name: 'run_length', fn: algoRunLength },
  { name: 'ewma', fn: algoEWMA },
  { name: 'momentum', fn: algoMomentum },
  { name: 'frequency_window', fn: algoFrequencyWindow },
  { name: 'transition_pattern', fn: algoTransitionPattern },
  { name: 'streak_hazard', fn: algoStreakHazard },
  { name: 'pattern_similarity', fn: algoPatternSimilarity }
];

// =====================================================
// 15. ONLINE LEARNING CONFIGURATION
// =====================================================

const LEARNING = {
  initialWeight: 1,
  minWeight: 0.15,
  maxWeight: 5,
  learningRate: 0.12,
  warmup: 12
};

// =====================================================
// 16. RUN ALL ALGORITHMS
// =====================================================

function getAlgoPredictions(seq) {
  return ALGOS.map(algo => {
    try {
      return algo.fn(seq);
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
// 17. WEIGHTED ENSEMBLE
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

// =====================================================
// 18. ONLINE WEIGHT LEARNING
// =====================================================

function updateWeights(predictions, actual, weights) {
  for (let i = 0; i < predictions.length; i++) {
    const error =
      predictions[i].pTai - actual;

    const loss = error * error;

    // Dự đoán tốt: tăng tương đối trọng số.
    // Dự đoán kém: giảm tương đối trọng số.
    weights[i] = clamp(
      weights[i] * Math.exp(
        -LEARNING.learningRate * (loss - 0.25)
      ),
      LEARNING.minWeight,
      LEARNING.maxWeight
    );
  }
}

// =====================================================
// 19. MAIN PREDICTION ENGINE
// =====================================================

function ensemblePredict(seq) {
  const weights = ALGOS.map(
    () => LEARNING.initialWeight
  );

  const historicalHits = [];

  const firstTrainIndex = Math.max(
    LEARNING.warmup,
    3
  );

  // Backtest tuần tự và cập nhật trọng số.
  for (
    let i = firstTrainIndex;
    i < seq.length;
    i++
  ) {
    const past = seq.slice(0, i);

    const predictions = getAlgoPredictions(past);

    const ensembleP = combinePredictions(
      predictions,
      weights
    );

    const actual = seq[i];

    historicalHits.push(
      (ensembleP >= 0.5 ? 1 : 0) === actual
        ? 1
        : 0
    );

    updateWeights(
      predictions,
      actual,
      weights
    );
  }

  // Dự đoán phiên kế tiếp.
  const predictions = getAlgoPredictions(seq);

  const pTai = combinePredictions(
    predictions,
    weights
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  const recentHits = historicalHits.slice(-100);

  const historicalAccuracy = recentHits.length
    ? mean(recentHits)
    : 0.5;

  // Độ tin cậy thận trọng, không phải xác suất
  // thắng đã được bảo đảm.
  const confidence = clamp(
    50 + Math.max(
      0,
      historicalAccuracy - 0.5
    ) * 100,
    50,
    75
  );

  const agreement = predictions.filter(
    p => (p.pTai >= 0.5) === (pTai >= 0.5)
  ).length;

  return {
    pick,

    confidence:
      Math.round(confidence * 100) / 100,

    pTai:
      Math.round(pTai * 10000) / 10000,

    agreement: `${agreement}/${ALGOS.length}`,

    historicalAccuracy:
      Math.round(historicalAccuracy * 10000) / 100,

    trainedSamples: historicalHits.length,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,

      pick: predictions[index].pick,

      probabilityTai:
        Math.round(
          predictions[index].pTai * 10000
        ) / 10000,

      weight:
        Math.round(weights[index] * 10000) / 10000
    }))
  };
}

// =====================================================
// 20. VIETNAM TIME
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
// 21. API RESPONSE
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

  const seq = history.map(
    h => h.result
  );

  const ens = ensemblePredict(seq);

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

    solanhoctap: ens.trainedSamples,

    dongthuan: ens.agreement,

    trongso: ens.algorithms.map(a => ({
      thuattoan: a.name,

      du_doan: a.pick === 1
        ? 'tai'
        : 'xiu',

      xac_suat_tai: a.probabilityTai,

      trong_so: a.weight
    })),

    uptime: formatTimeVN()
  };
}

// =====================================================
// 22. API ROUTES
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
// 23. BOOT SERVER
// =====================================================

initDataFile();

// Đồng bộ lần đầu.
syncData();

// Đồng bộ định kỳ.
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
    '[BOOT] Online learning enabled'
  );
});
