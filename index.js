
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// =====================================================
// 1. CONFIG
// =====================================================

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const DATA_DIR = process.env.DATA_DIR || __dirname;

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

const DATA_FILE = path.join(DATA_DIR, 'lichsulc79md5.json');
const AI_FILE = path.join(DATA_DIR, 'ai_learning_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const PORT = process.env.PORT || 3000;
const POLL_INTERVAL = 15000;

const ANALYSIS_WINDOW = 200;
const LEARNING_WINDOW = 40;

const MODEL_NAMES = [
  'alternation',
  'markov',
  'pattern',
  'regime',
  'trend'
];

// =====================================================
// 2. STORAGE
// =====================================================

function atomicWrite(file, content) {
  const temp = file + '.tmp';
  fs.writeFileSync(temp, content, 'utf8');
  fs.renameSync(temp, file);
}

function readHistory() {
  try {
    if (!fs.existsSync(DATA_FILE)) {
      atomicWrite(DATA_FILE, '[]');
      return [];
    }

    const data = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8')
    );

    return Array.isArray(data) ? data : [];
  } catch (err) {
    console.error('[HISTORY READ]', err.message);
    return [];
  }
}

function writeHistory(history) {
  atomicWrite(
    DATA_FILE,
    JSON.stringify(history.slice(-ANALYSIS_WINDOW), null, 2)
  );
}

function defaultState() {
  const weights = {};
  const modelStats = {};

  for (const name of MODEL_NAMES) {
    weights[name] = 1;
    modelStats[name] = {
      total: 0,
      correct: 0,
      brier: 0,
      recentLosses: []
    };
  }

  return {
    version: 'AI-ADAPTIVE-V4',
    weights,
    modelStats,
    stats: {
      total: 0,
      win: 0,
      lose: 0
    },
    compare: [],
    pending: null,
    lastLearnedId: null
  };
}

function loadState() {
  const defaults = defaultState();

  try {
    if (!fs.existsSync(AI_FILE)) return defaults;

    const saved = JSON.parse(
      fs.readFileSync(AI_FILE, 'utf8')
    );

    const state = {
      ...defaults,
      ...saved,
      version: 'AI-ADAPTIVE-V4',
      weights: {
        ...defaults.weights,
        ...(saved.weights || {})
      },
      modelStats: {
        ...defaults.modelStats,
        ...(saved.modelStats || {})
      },
      stats: {
        ...defaults.stats,
        ...(saved.stats || {})
      },
      compare: Array.isArray(saved.compare)
        ? saved.compare
        : [],
      pending: saved.pending || null
    };

    // Bảo đảm trạng thái cũ tương thích.
    for (const name of MODEL_NAMES) {
      const old = state.modelStats[name] || {};

      state.modelStats[name] = {
        total: Number(old.total) || 0,
        correct: Number(old.correct) || 0,
        brier: Number(old.brier) || 0,
        recentLosses: Array.isArray(old.recentLosses)
          ? old.recentLosses.slice(-LEARNING_WINDOW)
          : []
      };

      state.weights[name] = clamp(
        Number(state.weights[name]) || 1,
        0.2,
        4
      );
    }

    return state;
  } catch (err) {
    console.error('[AI READ]', err.message);
    return defaults;
  }
}

let AI = loadState();

function saveState() {
  atomicWrite(AI_FILE, JSON.stringify(AI, null, 2));
}

// =====================================================
// 3. COMMON FUNCTIONS
// =====================================================

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

function mean(arr) {
  if (!arr.length) return 0.5;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function entropy(p) {
  p = clamp(p, 0.000001, 0.999999);
  return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
}

function toSequence(history) {
  return history
    .slice(-ANALYSIS_WINDOW)
    .map(x => Number(x.result))
    .filter(x => x === 0 || x === 1);
}

// =====================================================
// 4. MODEL 1: ALTERNATION INTELLIGENCE
// Phân tích cầu 1-1: TXT / XTX
// =====================================================

function getAlternationLength(seq) {
  if (seq.length < 2) return 1;

  let length = 1;

  for (let i = seq.length - 1; i > 0; i--) {
    if (seq[i] !== seq[i - 1]) {
      length++;
    } else {
      break;
    }
  }

  return length;
}

function predictAlternation(seq) {
  if (seq.length < 6) return 0.5;

  const run = getAlternationLength(seq);
  const last = seq[seq.length - 1];

  let continueCount = 1;
  let breakCount = 1;
  let matches = 0;

  // Tìm các chuỗi luân phiên tương tự trong 200 phiên.
  for (let end = 1; end < seq.length - 1; end++) {
    let oldRun = 1;

    for (let j = end; j > 0; j--) {
      if (seq[j] !== seq[j - 1]) {
        oldRun++;
      } else {
        break;
      }
    }

    if (oldRun !== run) continue;
    if (seq[end] !== last) continue;

    const next = seq[end + 1];

    if (next !== last) continueCount++;
    else breakCount++;

    matches++;
  }

  // Khi chuỗi hiện tại đang luân phiên mạnh,
  // ưu tiên kết quả đối diện nhưng phải có dữ liệu hỗ trợ.
  if (run >= 3 && matches >= 2) {
    const pOpposite =
      continueCount / (continueCount + breakCount);

    return last === 1
      ? 1 - pOpposite
      : pOpposite;
  }

  // Không đủ mẫu thì trở về trung tính.
  return 0.5;
}

// =====================================================
// 5. MODEL 2: MULTI-ORDER MARKOV
// =====================================================

function predictMarkov(seq) {
  if (seq.length < 8) return 0.5;

  let numerator = 0;
  let denominator = 0;

  for (let order = 1; order <= 6; order++) {
    if (seq.length <= order + 3) continue;

    const key = seq.slice(-order).join('');

    let tai = 1;
    let xiu = 1;
    let matches = 0;

    for (let i = order; i < seq.length; i++) {
      const previous = seq
        .slice(i - order, i)
        .join('');

      if (previous !== key) continue;

      if (seq[i] === 1) tai++;
      else xiu++;

      matches++;
    }

    if (matches < 2) continue;

    const probability = tai / (tai + xiu);
    const weight = Math.sqrt(matches) * order;

    numerator += probability * weight;
    denominator += weight;
  }

  return denominator
    ? numerator / denominator
    : 0.5;
}

// =====================================================
// 6. MODEL 3: PATTERN INTELLIGENCE
// =====================================================

function predictPattern(seq) {
  if (seq.length < 10) return 0.5;

  const candidates = [];

  for (let length = 3; length <= 10; length++) {
    if (seq.length <= length + 2) continue;

    const target = seq.slice(-length);

    let tai = 1;
    let xiu = 1;
    let matches = 0;

    for (let i = length; i < seq.length; i++) {
      let equal = true;

      for (let j = 0; j < length; j++) {
        if (seq[i - length + j] !== target[j]) {
          equal = false;
          break;
        }
      }

      if (!equal) continue;

      if (seq[i] === 1) tai++;
      else xiu++;

      matches++;
    }

    if (matches >= 2) {
      const probability = tai / (tai + xiu);

      // Ưu tiên mẫu dài và có số lần xuất hiện đủ.
      const weight = Math.sqrt(matches) * Math.log2(length + 1);

      candidates.push({ probability, weight, matches });
    }
  }

  if (!candidates.length) return 0.5;

  let numerator = 0;
  let denominator = 0;

  for (const item of candidates) {
    numerator += item.probability * item.weight;
    denominator += item.weight;
  }

  return numerator / denominator;
}

// =====================================================
// 7. MODEL 4: REGIME DETECTION
// Phân loại: BỆT / ĐẢO / HỖN HỢP
// =====================================================

function detectRegime(seq) {
  const recent = seq.slice(-20);

  if (recent.length < 8) {
    return {
      type: 'INSUFFICIENT_DATA',
      alternation: 0,
      sameRate: 0.5,
      streak: getAlternationLength(seq)
    };
  }

  let switches = 0;

  for (let i = 1; i < recent.length; i++) {
    if (recent[i] !== recent[i - 1]) switches++;
  }

  const switchRate = switches / (recent.length - 1);
  const sameRate = 1 - switchRate;

  let streak = 1;

  for (let i = recent.length - 1; i > 0; i--) {
    if (recent[i] === recent[i - 1]) streak++;
    else break;
  }

  let type = 'MIXED';

  if (switchRate >= 0.75) {
    type = 'ALTERNATING';
  } else if (switchRate <= 0.35) {
    type = 'STREAK';
  }

  return {
    type,
    alternation: switchRate,
    sameRate,
    streak
  };
}

function predictRegime(seq) {
  const regime = detectRegime(seq);

  if (regime.type === 'INSUFFICIENT_DATA') {
    return 0.5;
  }

  // Tìm lịch sử có trạng thái tương tự.
  const currentType = regime.type;
  let tai = 1;
  let xiu = 1;
  let matches = 0;

  for (let end = 8; end < seq.length; end++) {
    const sample = seq.slice(Math.max(0, end - 20), end);

    if (sample.length < 8) continue;

    let switches = 0;

    for (let i = 1; i < sample.length; i++) {
      if (sample[i] !== sample[i - 1]) switches++;
    }

    const rate = switches / (sample.length - 1);

    let oldType = 'MIXED';

    if (rate >= 0.75) oldType = 'ALTERNATING';
    else if (rate <= 0.35) oldType = 'STREAK';

    if (oldType !== currentType) continue;

    if (seq[end] === 1) tai++;
    else xiu++;

    matches++;
  }

  if (matches < 3) return 0.5;

  return tai / (tai + xiu);
}

// =====================================================
// 8. MODEL 5: RECENT TREND
// =====================================================

function predictTrend(seq) {
  if (seq.length < 5) return 0.5;

  const windows = [8, 15, 30, 60, 100, 200];

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < windows.length; i++) {
    const size = windows[i];

    if (seq.length < Math.min(size, 5)) continue;

    const sample = seq.slice(-size);
    const p = mean(sample);

    // Cửa sổ gần hiện tại có trọng số cao hơn.
    const weight = Math.exp(-i * 0.38);

    numerator += p * weight;
    denominator += weight;
  }

  return denominator
    ? numerator / denominator
    : 0.5;
}

// =====================================================
// 9. RUN FIVE MODELS
// =====================================================

function getModelPredictions(seq) {
  return {
    alternation: clamp(predictAlternation(seq), 0.01, 0.99),
    markov: clamp(predictMarkov(seq), 0.01, 0.99),
    pattern: clamp(predictPattern(seq), 0.01, 0.99),
    regime: clamp(predictRegime(seq), 0.01, 0.99),
    trend: clamp(predictTrend(seq), 0.01, 0.99)
  };
}

// =====================================================
// 10. ADAPTIVE META LEARNER
// =====================================================

function calculatePrediction(history) {
  const seq = toSequence(history);
  const models = getModelPredictions(seq);

  let numerator = 0;
  let denominator = 0;

  for (const name of MODEL_NAMES) {
    const weight = clamp(AI.weights[name], 0.2, 4);

    numerator += models[name] * weight;
    denominator += weight;
  }

  let probability = denominator
    ? numerator / denominator
    : 0.5;

  // Ít dữ liệu kiểm định thì giảm độ cực đoan.
  const sampleFactor = Math.min(
    1,
    Math.sqrt(AI.stats.total / 50)
  );

  probability =
    0.5 + (probability - 0.5) * sampleFactor;

  // Khi các mô hình bất đồng mạnh, giảm độ tự tin.
  const values = Object.values(models);
  const disagreement = Math.sqrt(
    mean(values.map(p => Math.pow(p - mean(values), 2)))
  );

  if (disagreement > 0.2) {
    probability =
      0.5 + (probability - 0.5) * 0.75;
  }

  probability = clamp(probability, 0.02, 0.98);

  const final = probability >= 0.5 ? 'TÀI' : 'XỈU';

  const confidence = Math.round(
    clamp(50 + Math.abs(probability - 0.5) * 100, 50, 75)
  );

  return {
    final,
    probability,
    confidence,
    models,
    disagreement
  };
}

// =====================================================
// 11. ONLINE LEARNING
// =====================================================

function learnFromPrediction(pending, actual) {
  if (!pending || !pending.models) return;

  const actualValue = actual.result;
  const predicted = pending.probability >= 0.5 ? 1 : 0;
  const win = predicted === actualValue;

  AI.stats.total++;

  if (win) AI.stats.win++;
  else AI.stats.lose++;

  for (const name of MODEL_NAMES) {
    const p = clamp(
      Number(pending.models[name] ?? 0.5),
      0.01,
      0.99
    );

    const loss = Math.pow(p - actualValue, 2);
    const stats = AI.modelStats[name];

    stats.total++;

    if ((p >= 0.5 ? 1 : 0) === actualValue) {
      stats.correct++;
    }

    stats.brier += loss;

    stats.recentLosses.push(loss);

    if (stats.recentLosses.length > LEARNING_WINDOW) {
      stats.recentLosses.shift();
    }

    // Tự điều chỉnh theo lỗi gần đây.
    const recentLoss = mean(stats.recentLosses);

    const adjustment = Math.exp(
      0.08 * (0.25 - recentLoss)
    );

    AI.weights[name] = clamp(
      AI.weights[name] * adjustment,
      0.2,
      4
    );
  }

  AI.compare.push({
    phien: actual.id,
    predict: pending.final,
    real: actualValue === 1 ? 'TÀI' : 'XỈU',
    win,
    probability: Number(pending.probability.toFixed(4)),
    time: new Date().toISOString()
  });

  if (AI.compare.length > 200) {
    AI.compare = AI.compare.slice(-200);
  }

  AI.lastLearnedId = actual.id;

  console.log(
    `[LEARN] ${actual.id} | ${win ? 'WIN' : 'LOSE'} | ` +
    `Accuracy ${(AI.stats.win / AI.stats.total * 100).toFixed(2)}%`
  );
}

// =====================================================
// 12. PENDING PREDICTION
// =====================================================

function createPending(history, targetId) {
  const result = calculatePrediction(history);

  return {
    targetId,
    final: result.final,
    probability: result.probability,
    confidence: result.confidence,
    models: result.models,
    createdAt: new Date().toISOString()
  };
}

// =====================================================
// 13. PROCESS EACH SESSION IN ORDER
// =====================================================

function processSession(session, history) {
  if (AI.pending) {
    if (AI.pending.targetId === session.id) {
      learnFromPrediction(AI.pending, session);
    } else if (AI.pending.targetId < session.id) {
      console.log(
        `[LEARN SKIP] Missing session ${AI.pending.targetId}`
      );
    } else {
      console.log(
        `[LEARN SKIP] Unexpected session ${session.id}`
      );
    }

    AI.pending = null;
  }

  history.push(session);

  if (history.length > ANALYSIS_WINDOW) {
    history.splice(0, history.length - ANALYSIS_WINDOW);
  }

  AI.pending = createPending(history, session.id + 1);
}

// =====================================================
// 14. SOURCE NORMALIZATION
// =====================================================

async function fetchSource() {
  const response = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(10000)
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json();
}

function normalizeSession(raw) {
  const id = Number(raw.id ?? raw.session ?? raw.issue);

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
    (typeof raw.result === 'string' ? raw.result : '')
  ).trim().toUpperCase();

  let result;

  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) {
    result = 1;
  } else if (['XIU', 'XỈU', 'SMALL'].includes(sourceResult)) {
    result = 0;
  } else if (sum >= 11 && sum <= 18) {
    result = 1;
  } else if (sum >= 3 && sum <= 10) {
    result = 0;
  } else {
    return null;
  }

  if (!Number.isFinite(id)) return null;

  return {
    id,
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
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

    if (!Array.isArray(list) || !list.length) return;

    const history = readHistory();
    const knownIds = new Set(
      history.map(x => Number(x.id))
    );

    const fresh = list
      .map(normalizeSession)
      .filter(s => s && !knownIds.has(s.id))
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    let added = 0;

    for (const session of fresh) {
      if (knownIds.has(session.id)) continue;

      processSession(session, history);
      knownIds.add(session.id);
      added++;
    }

    if (added > 0) {
      history.sort((a, b) => a.id - b.id);

      writeHistory(history);
      saveState();

      console.log(
        `[SYNC] +${added} | History ${history.length} | ` +
        `Learned ${AI.stats.total}`
      );
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 16. TIME
// =====================================================

function formatTimeVN() {
  return new Date(
    Date.now() + 7 * 3600 * 1000
  ).toISOString().replace('Z', '+07:00');
}

// =====================================================
// 17. MAIN RESPONSE
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
  const prediction = calculatePrediction(history);

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,

    dudoanphien: current.id + 1,
    dudoan: prediction.final === 'TÀI' ? 'tai' : 'xiu',
    dotincay: `${prediction.confidence}%`,
    uptime: formatTimeVN(),

    ai: {
      version: AI.version,
      window: Math.min(history.length, ANALYSIS_WINDOW),
      probability: Number(prediction.probability.toFixed(4)),
      disagreement: Number(prediction.disagreement.toFixed(4)),
      models: prediction.models,
      weights: AI.weights,
      total: AI.stats.total,
      win: AI.stats.win,
      lose: AI.stats.lose,
      accuracy: AI.stats.total
        ? Number((AI.stats.win / AI.stats.total * 100).toFixed(2))
        : 0
    }
  };
}

// =====================================================
// 18. ROUTES
// =====================================================

app.get('/', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[API ERROR]', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/predict', (req, res) => {
  try {
    res.json(buildResponse());
  } catch (err) {
    console.error('[PREDICT ERROR]', err.message);
    res.status(500).json({ error: 'Prediction error' });
  }
});

app.get('/ping', (req, res) => {
  res.send('pong');
});

app.get('/history', (req, res) => {
  const history = readHistory();

  res.json({
    count: history.length,
    window: ANALYSIS_WINDOW,
    last20: history.slice(-20)
  });
});

app.get('/ai-status', (req, res) => {
  const total = AI.stats.total;

  const models = {};

  for (const name of MODEL_NAMES) {
    const stats = AI.modelStats[name];

    models[name] = {
      weight: Number(AI.weights[name].toFixed(4)),
      total: stats.total,
      accuracy: stats.total
        ? Number((stats.correct / stats.total * 100).toFixed(2))
        : 0,
      averageBrier: stats.total
        ? Number((stats.brier / stats.total).toFixed(4))
        : null,
      recentBrier: stats.recentLosses.length
        ? Number(mean(stats.recentLosses).toFixed(4))
        : null
    };
  }

  res.json({
    version: AI.version,
    total,
    win: AI.stats.win,
    lose: AI.stats.lose,
    accuracy: total
      ? Number((AI.stats.win / total * 100).toFixed(2))
      : 0,
    analysisWindow: ANALYSIS_WINDOW,
    weights: AI.weights,
    models,
    pending: AI.pending,
    lastLearnedId: AI.lastLearnedId
  });
});

// =====================================================
// 19. BOOT
// =====================================================

if (!fs.existsSync(DATA_FILE)) {
  atomicWrite(DATA_FILE, '[]');
}

const bootHistory = readHistory().slice(-ANALYSIS_WINDOW);

if (bootHistory.length) {
  const current = bootHistory[bootHistory.length - 1];

  // Chỉ giữ pending nếu nó đúng phiên kế tiếp.
  if (
    !AI.pending ||
    Number(AI.pending.targetId) !== Number(current.id) + 1
  ) {
    AI.pending = createPending(
      bootHistory,
      Number(current.id) + 1
    );

    saveState();
  }
}

console.log('[BOOT] AI-ADAPTIVE-V4');
console.log('[BOOT] Analysis window:', ANALYSIS_WINDOW);
console.log('[BOOT] History:', bootHistory.length);
console.log('[BOOT] Learned:', AI.stats.total);

syncData();

const timer = setInterval(syncData, POLL_INTERVAL);

if (typeof timer.unref === 'function') {
  timer.unref();
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[SERVER] Running on ${PORT}`);
  console.log(`[SERVER] ${formatTimeVN()}`);
});
