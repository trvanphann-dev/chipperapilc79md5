
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// =====================================================
// 0. CONFIGURATION
// =====================================================

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');

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

function resetHistory() {
  try {
    fs.writeFileSync(DATA_FILE, '[]', 'utf8');
    CAUTION.data = [];
    console.log('[RESET] History cleared');
  } catch (err) {
    console.error('[RESET ERROR]', err.message);
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

  const validDices = dices.filter(
    d => Number.isInteger(d) && d >= 1 && d <= 6
  );

  const sum = validDices.length
    ? validDices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? 0);

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    (Array.isArray(raw.result) ? '' : raw.result) ??
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
    _id: String(raw._id ?? id),
    dices: validDices,
    sum,
    result
  };
}

// =====================================================
// 3. BASIC STATISTICS
// =====================================================

function clampProbability(value) {
  if (!Number.isFinite(value)) return 0.5;

  return Math.max(0.02, Math.min(0.98, value));
}

function countTai(seq) {
  return seq.reduce(
    (total, value) => total + (value === 1 ? 1 : 0),
    0
  );
}

function smoothedProbability(seq, windowSize, prior = 12) {
  const data = seq.slice(-windowSize);

  if (!data.length) return 0.5;

  const tai = countTai(data);

  return (tai + prior * 0.5) / (data.length + prior);
}

// =====================================================
// 4. EXPERT MODEL 1 - GLOBAL
// =====================================================

function globalModel(seq) {
  if (!seq.length) return 0.5;

  return smoothedProbability(seq, seq.length, 12);
}

// =====================================================
// 5. EXPERT MODEL 2 - RECENT TREND
// =====================================================

function recentModel(seq) {
  const p10 = smoothedProbability(seq, 10, 8);
  const p25 = smoothedProbability(seq, 25, 10);
  const p50 = smoothedProbability(seq, 50, 12);
  const p100 = smoothedProbability(seq, 100, 16);

  return (
    p10 * 0.40 +
    p25 * 0.30 +
    p50 * 0.20 +
    p100 * 0.10
  );
}

// =====================================================
// 6. EXPERT MODEL 3 - MARKOV
// =====================================================

function markovModel(seq) {
  if (seq.length < 3) return 0.5;

  const last = seq[seq.length - 1];

  let total = 0;
  let tai = 0;

  for (let i = 0; i < seq.length - 1; i++) {
    if (seq[i] === last) {
      total++;

      if (seq[i + 1] === 1) {
        tai++;
      }
    }
  }

  if (total === 0) return 0.5;

  const observed = (tai + 3) / (total + 6);
  const reliability = total / (total + 12);

  return 0.5 + (observed - 0.5) * reliability;
}

// =====================================================
// 7. EXPERT MODEL 4 - PATTERN
// =====================================================

function patternModel(seq) {
  if (seq.length < 5) return 0.5;

  let weighted = 0;
  let totalWeight = 0;

  for (let length = 1; length <= 3; length++) {
    if (seq.length <= length) continue;

    const context = seq.slice(-length).join('');

    let matches = 0;
    let tai = 0;

    for (let i = 0; i <= seq.length - length - 1; i++) {
      const oldContext = seq
        .slice(i, i + length)
        .join('');

      if (oldContext === context) {
        matches++;

        if (seq[i + length] === 1) {
          tai++;
        }
      }
    }

    // Bỏ qua mẫu có quá ít dữ liệu.
    if (matches < 3) continue;

    const observed = (tai + 3) / (matches + 6);
    const reliability = matches / (matches + 10);

    const probability =
      0.5 + (observed - 0.5) * reliability;

    const weight = Math.min(matches, 12);

    weighted += probability * weight;
    totalWeight += weight;
  }

  if (totalWeight === 0) return 0.5;

  return weighted / totalWeight;
}

// =====================================================
// 8. COLLECT EXPERT PREDICTIONS
// =====================================================

function getExpertProbabilities(seq) {
  return [
    clampProbability(globalModel(seq)),
    clampProbability(recentModel(seq)),
    clampProbability(markovModel(seq)),
    clampProbability(patternModel(seq))
  ];
}

// =====================================================
// 9. ADAPTIVE WEIGHT SYSTEM
// =====================================================

function normalizeWeights(weights) {
  const adjusted = weights.map(w =>
    Math.max(0.02, Number.isFinite(w) ? w : 0.25)
  );

  const total = adjusted.reduce((a, b) => a + b, 0);

  return adjusted.map(w => w / total);
}

function weightedPrediction(probabilities, weights) {
  return probabilities.reduce(
    (sum, p, i) => sum + p * weights[i],
    0
  );
}

// Cập nhật trọng số sau khi có kết quả thật.
// Dùng log-loss để phạt dự đoán sai tự tin quá mức.
function updateExpertWeights(weights, probabilities, actual) {
  const learningRate = 0.15;

  const updated = weights.map((weight, i) => {
    const p = clampProbability(probabilities[i]);

    const loss = actual === 1
      ? -Math.log(p)
      : -Math.log(1 - p);

    return weight * Math.exp(-learningRate * loss);
  });

  return normalizeWeights(updated);
}

// =====================================================
// 10. WALK-FORWARD ADAPTIVE LEARNING
// =====================================================

function adaptiveEvaluation(seq) {
  const minimumTraining = 25;

  let weights = [0.25, 0.25, 0.25, 0.25];

  const results = [];

  if (seq.length <= minimumTraining) {
    return {
      weights,
      results,
      probability: 0.5,
      confidence: 50,
      components: {}
    };
  }

  // Chỉ dùng dữ liệu trước phiên đang kiểm tra.
  for (let i = minimumTraining; i < seq.length; i++) {
    const training = seq.slice(0, i);

    const expertProbabilities =
      getExpertProbabilities(training);

    const prediction = weightedPrediction(
      expertProbabilities,
      weights
    );

    const actual = seq[i];

    const predictedSide = prediction >= 0.5 ? 1 : 0;

    results.push({
      predicted: predictedSide,
      actual,
      probability: prediction,
      correct: predictedSide === actual
    });

    // Biết kết quả rồi mới cập nhật trọng số.
    weights = updateExpertWeights(
      weights,
      expertProbabilities,
      actual
    );
  }

  // Trọng số đã được học từ dữ liệu lịch sử.
  const currentExperts = getExpertProbabilities(seq);

  const finalProbability = weightedPrediction(
    currentExperts,
    weights
  );

  // Đánh giá độ chính xác trên tối đa 80 phiên gần nhất.
  const evaluationResults = results.slice(-80);

  const correct = evaluationResults.filter(
    r => r.correct
  ).length;

  // Prior cân bằng, tránh kết luận quá mạnh khi ít mẫu.
  const priorSamples = 20;
  const priorCorrect = 10;

  const accuracy =
    (correct + priorCorrect) /
    (evaluationResults.length + priorSamples);

  const confidence = Math.round(accuracy * 100);

  return {
    weights,
    results: evaluationResults,
    probability: clampProbability(finalProbability),
    confidence,
    components: {
      global: currentExperts[0],
      recent: currentExperts[1],
      markov: currentExperts[2],
      pattern: currentExperts[3],
      weights: {
        global: weights[0],
        recent: weights[1],
        markov: weights[2],
        pattern: weights[3]
      }
    }
  };
}

// =====================================================
// 11. PREDICTION ENGINE
// =====================================================

function predictProbability(seq) {
  if (!seq || seq.length < 25) {
    return {
      probability: 0.5,
      confidence: 50,
      components: {}
    };
  }

  const evaluation = adaptiveEvaluation(seq);

  return {
    probability: evaluation.probability,
    confidence: evaluation.confidence,
    components: evaluation.components
  };
}

// =====================================================
// 12. ENTROPY / CAUTION
// =====================================================

function entropy(seq, windowSize = 30) {
  const data = seq.slice(-windowSize);

  if (data.length < windowSize) return 0;

  const tai = countTai(data);
  const p1 = tai / data.length;
  const p0 = 1 - p1;

  if (p1 === 0 || p1 === 1) return 0;

  return -(
    p1 * Math.log2(p1) +
    p0 * Math.log2(p0)
  );
}

function chaosBlock(seq) {
  if (seq.length < 30) return false;

  return entropy(seq, 30) > 0.92;
}

// =====================================================
// 13. FINAL PREDICTION
// =====================================================

function calculateFinal(seq) {
  const result = predictProbability(seq);

  const probability = result.probability;

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  return {
    final,
    confidence: result.confidence,
    caution: chaosBlock(seq),
    probability,
    components: result.components
  };
}

// =====================================================
// 14. STATE: PREDICTION AND STATISTICS
// =====================================================

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

function processLatest(session) {
  const seq = readHistory().map(h => h.result);

  const {
    final,
    confidence,
    caution
  } = calculateFinal(seq);

  if (lastPredict) {
    STATS.total++;

    const real = session.result === 1
      ? 'TÀI'
      : 'XỈU';

    const win = lastPredict.predict === real;

    if (win) {
      STATS.win++;
    } else {
      STATS.lose++;
    }

    COMPARE.push({
      phien: session.id,
      predict: lastPredict.predict,
      real,
      win
    });

    if (COMPARE.length > 10) {
      COMPARE.shift();
    }

    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);

      if (CAUTION.data.length > 10) {
        CAUTION.data.shift();
      }
    }
  }

  lastPredict = {
    predict: final,
    caution
  };

  return {
    final,
    confidence,
    caution
  };
}

// =====================================================
// 15. SYNC
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

    const fresh = list
      .map(normalizeSession)
      .filter(s =>
        Number.isFinite(s.id) &&
        Number.isFinite(s.sum) &&
        [0, 1].includes(s.result) &&
        s.id > lastId
      )
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    const history = readHistory();

    const knownIds = new Set(
      history.map(h => h.id)
    );

    const added = [];

    for (const session of fresh) {
      if (knownIds.has(session.id)) continue;

      history.push(session);
      knownIds.add(session.id);
      added.push(session);
    }

    lastId = Math.max(
      lastId,
      ...fresh.map(s => s.id)
    );

    if (added.length > 0) {
      history.sort((a, b) => a.id - b.id);

      const saved = history.slice(-MAX_HISTORY);

      writeHistory(saved);

      console.log(
        `[SYNC] +${added.length} sessions | Total ${saved.length}`
      );

      processLatest(added[added.length - 1]);
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 16. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace('Z', '+07:00');
}

// =====================================================
// 17. API RESPONSE
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

  const seq = history.map(h => h.result);

  const {
    final,
    confidence
  } = calculateFinal(seq);

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua: current.result === 1
      ? 'tai'
      : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: current.id + 1,

    dudoan: final === 'TÀI'
      ? 'tai'
      : 'xiu',

    dotincay: `${confidence}%`,

    uptime: formatTimeVN()
  };
}

// =====================================================
// 18. API ROUTES
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
// 19. DEBUG ROUTE
// Xem xác suất và trọng số từng mô hình.
// =====================================================

app.get('/debug', (req, res) => {
  try {
    const history = readHistory();
    const seq = history.map(h => h.result);

    const result = adaptiveEvaluation(seq);

    res.json({
      totalSessions: seq.length,

      taiCount: countTai(seq),

      xiuCount: seq.length - countTai(seq),

      taiRate: seq.length
        ? Number((countTai(seq) / seq.length).toFixed(4))
        : 0,

      prediction: result.probability >= 0.5
        ? 'tai'
        : 'xiu',

      taiProbability: Number(
        (result.probability * 100).toFixed(2)
      ),

      xiuProbability: Number(
        ((1 - result.probability) * 100).toFixed(2)
      ),

      confidence: result.confidence,

      components: result.components,

      backtestSamples: result.results.length,

      backtestCorrect: result.results.filter(
        r => r.correct
      ).length
    });
  } catch (err) {
    console.error('[DEBUG ERROR]', err.message);

    res.status(500).json({
      error: 'Debug error'
    });
  }
});

// =====================================================
// 20. BOOT SERVER
// =====================================================

initDataFile();

{
  const history = readHistory();

  lastId = history.length
    ? Number(history[history.length - 1].id)
    : 0;
}

// Đồng bộ ngay khi khởi động.
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
    '[BOOT] Adaptive Global + Recent + Markov + Pattern loaded'
  );

  console.log(
    '[BOOT] Walk-forward learning enabled'
  );
});
