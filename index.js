
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
// 1. STATE
// =====================================================

let lastPredict = null;
let lastId = 0;
let syncInProgress = false;

const STATS = {
  total: 0,
  win: 0,
  lose: 0
};

const CAUTION = {
  data: []
};

const COMPARE = [];

// =====================================================
// 2. DATA STORAGE
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
    lastPredict = null;

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
// 3. SOURCE API
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

// =====================================================
// 4. NORMALIZE SESSION
// =====================================================

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
    (typeof raw.result === 'string' ? raw.result : '')
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
    id: Number(id),
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// =====================================================
// 5. ADVANCED ALGORITHM ENGINE
// NO MD5 - NO RANDOM
// =====================================================

// -----------------------------------------------------
// 5.1 BAYESIAN ADAPTIVE
// -----------------------------------------------------

class BayesianModel {
  probability(seq) {
    if (!seq.length) return 0.5;

    const recent = seq.slice(-120);

    const ones = recent.reduce(
      (a, b) => a + b,
      0
    );

    // Laplace smoothing
    return (ones + 1) / (recent.length + 2);
  }
}

// -----------------------------------------------------
// 5.2 MARKOV CHAIN
// -----------------------------------------------------

class HMMModel {
  predict(seq) {
    if (seq.length < 3) return 0.5;

    const last = seq[seq.length - 1];

    let total = 0;
    let ones = 0;

    // Thống kê kết quả sau trạng thái hiện tại
    for (let i = 0; i < seq.length - 1; i++) {
      if (seq[i] === last) {
        total++;
        ones += seq[i + 1];
      }
    }

    if (total === 0) return 0.5;

    // Bayesian smoothing
    return (ones + 2) / (total + 4);
  }
}

// -----------------------------------------------------
// 5.3 PATTERN RECOGNITION
// -----------------------------------------------------

function patternAnalysis(seq) {
  if (seq.length < 8) {
    return {
      probability: 0.5,
      support: 0
    };
  }

  const recent = seq.slice(-120);

  const base =
    recent.reduce((a, b) => a + b, 0) /
    recent.length;

  const maxLength = Math.min(6, seq.length - 1);

  let weightedProbability = 0;
  let totalWeight = 0;
  let totalSupport = 0;

  for (let len = 2; len <= maxLength; len++) {
    const pattern = seq.slice(-len);

    let count = 0;
    let ones = 0;

    // Tìm chuỗi giống nhau trong quá khứ
    for (
      let end = len - 1;
      end < seq.length - 1;
      end++
    ) {
      let matched = true;

      for (let j = 0; j < len; j++) {
        if (
          seq[end - len + 1 + j] !== pattern[j]
        ) {
          matched = false;
          break;
        }
      }

      if (matched) {
        count++;

        // Kết quả xuất hiện ngay sau pattern
        ones += seq[end + 1];
      }
    }

    if (count === 0) continue;

    const probability =
      (ones + base * 4) / (count + 4);

    // Mẫu có nhiều lần xuất hiện được tin cậy hơn
    const weight = count / (count + 4);

    weightedProbability += probability * weight;
    totalWeight += weight;
    totalSupport += count;
  }

  if (totalWeight === 0) {
    return {
      probability: 0.5,
      support: 0
    };
  }

  return {
    probability: weightedProbability / totalWeight,
    support: totalSupport
  };
}

// -----------------------------------------------------
// 5.4 STREAK & REVERSAL ANALYSIS
// -----------------------------------------------------

function streakAnalysis(seq) {
  if (seq.length < 10) {
    return {
      probability: 0.5,
      support: 0,
      streak: 0
    };
  }

  const last = seq[seq.length - 1];

  let streak = 1;

  // Tính cầu hiện tại
  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === last) {
      streak++;
    } else {
      break;
    }
  }

  // Giới hạn mẫu để tránh phụ thuộc chuỗi quá hiếm
  const targetStreak = Math.min(streak, 6);

  const recent = seq.slice(-120);

  const base =
    recent.reduce((a, b) => a + b, 0) /
    recent.length;

  let total = 0;
  let ones = 0;

  // Tìm các chuỗi cầu tương tự trong lịch sử
  for (let end = 1; end < seq.length - 1; end++) {
    if (seq[end] !== last) continue;

    // Chỉ xét điểm kết thúc của một chuỗi
    if (seq[end + 1] === seq[end]) continue;

    let previousStreak = 1;

    for (let j = end - 1; j >= 0; j--) {
      if (seq[j] === seq[end]) {
        previousStreak++;
      } else {
        break;
      }
    }

    if (previousStreak >= targetStreak) {
      total++;
      ones += seq[end + 1];
    }
  }

  return {
    probability: (ones + base * 4) / (total + 4),
    support: total,
    streak
  };
}

// -----------------------------------------------------
// 5.5 ENTROPY DETECTOR
// -----------------------------------------------------

function entropy(seq, windowSize = 30) {
  const recent = seq.slice(-windowSize);

  if (recent.length < windowSize) return 0;

  const p1 =
    recent.reduce((a, b) => a + b, 0) /
    recent.length;

  const p0 = 1 - p1;

  if (p0 === 0 || p1 === 0) return 0;

  return -(
    p0 * Math.log2(p0) +
    p1 * Math.log2(p1)
  );
}

function chaosBlock(seq) {
  if (seq.length < 30) return false;

  return entropy(seq, 30) > 0.92;
}

// -----------------------------------------------------
// 5.6 ADAPTIVE BANDIT
// -----------------------------------------------------

class Bandit {
  constructor() {
    this.score = {
      bayes: 1,
      hmm: 1,
      pattern: 1,
      streak: 1
    };
  }

  weight(name) {
    const value = this.score[name] ?? 1;

    return Math.max(
      0.4,
      Math.min(1.6, value)
    );
  }

  reward(name, predicted, actual) {
    const old = this.score[name] ?? 1;

    // Brier Score: đánh giá xác suất dự đoán
    const error = Math.pow(
      predicted - actual,
      2
    );

    const adjustment =
      1 - 4 * error;

    const updated =
      old * 0.98 + adjustment * 0.02;

    this.score[name] = Math.max(
      0.4,
      Math.min(1.6, updated)
    );
  }
}

// =====================================================
// 6. INITIALIZE MODELS
// =====================================================

const bayes = new BayesianModel();
const hmm = new HMMModel();
const bandit = new Bandit();

// =====================================================
// 7. FINAL PREDICTION ENGINE
// =====================================================

function calculateFinal(seq, id) {
  const n = seq.length;

  const caution = chaosBlock(seq);

  const pattern = patternAnalysis(seq);
  const streak = streakAnalysis(seq);

  const components = {
    bayes: bayes.probability(seq),
    hmm: hmm.predict(seq),
    pattern: pattern.probability,
    streak: streak.probability
  };

  // Adaptive ensemble
  let numerator = 0;
  let denominator = 0;

  for (const name of Object.keys(components)) {
    const weight = bandit.weight(name);

    numerator += components[name] * weight;
    denominator += weight;
  }

  let probability = denominator > 0
    ? numerator / denominator
    : 0.5;

  probability = Math.max(
    0.01,
    Math.min(0.99, probability)
  );

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  // Model agreement
  const agreement = Object.values(components)
    .filter(p =>
      (probability >= 0.5 && p >= 0.5) ||
      (probability < 0.5 && p < 0.5)
    ).length / 4;

  // Data support factor
  const dataFactor = Math.min(1, n / 100);

  const strength = Math.abs(probability - 0.5);

  // Confidence heuristic, not guaranteed win probability
  const confidence = Math.round(
    Math.min(
      75,
      50 + strength * 100 * agreement * dataFactor
    ) * 100
  ) / 100;

  return {
    final,
    confidence,
    caution,
    probability,
    components,
    agreement
  };
}

// =====================================================
// 8. PROCESS LATEST SESSION
// =====================================================

function processLatest(session) {
  const seq = readHistory().map(h => h.result);

  if (lastPredict) {
    STATS.total++;

    const actual = session.result;

    const real = actual === 1
      ? 'TÀI'
      : 'XỈU';

    const win = lastPredict.predict === real;

    // Update each model independently
    if (lastPredict.components) {
      for (const name of [
        'bayes',
        'hmm',
        'pattern',
        'streak'
      ]) {
        const predictedProbability =
          lastPredict.components[name];

        if (Number.isFinite(predictedProbability)) {
          bandit.reward(
            name,
            predictedProbability,
            actual
          );
        }
      }
    }

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

    // CAUTION tracking
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);

      if (CAUTION.data.length > 10) {
        CAUTION.data.shift();
      }

      if (CAUTION.data.length === 10) {
        const winCount = CAUTION.data.reduce(
          (a, b) => a + b,
          0
        );

        const loseCount = 10 - winCount;

        // Preserve original reset logic
        if (loseCount >= winCount) {
          resetHistory();

          console.log(
            '[CAUTION] History reset after 10 caution predictions'
          );

          return null;
        }
      }
    }
  }

  // Calculate prediction for next session
  const result = calculateFinal(
    seq,
    session._id ?? session.id
  );

  lastPredict = {
    predict: result.final,
    caution: result.caution,
    components: result.components
  };

  return result;
}

// =====================================================
// 9. SOURCE DATA SYNCHRONIZATION
// =====================================================

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

    let history = readHistory();

    const knownIds = new Set(
      history.map(h => h.id)
    );

    let added = 0;

    for (const s of fresh) {
      if (knownIds.has(s.id)) continue;

      history.push(s);
      knownIds.add(s.id);

      history.sort((a, b) => a.id - b.id);

      history = history.slice(-MAX_HISTORY);

      // Save new result
      writeHistory(history);

      lastId = s.id;

      // Evaluate old prediction and learn
      processLatest(s);

      // Reload in case the original reset logic ran
      history = readHistory();

      added++;
    }

    if (added > 0) {
      console.log(
        `[SYNC] +${added} sessions | Total ${history.length}`
      );

      console.log(
        `[LEARNING] Bayes: ${bandit.weight('bayes').toFixed(3)} | ` +
        `Markov: ${bandit.weight('hmm').toFixed(3)} | ` +
        `Pattern: ${bandit.weight('pattern').toFixed(3)} | ` +
        `Streak: ${bandit.weight('streak').toFixed(3)}`
      );

      if (STATS.total > 0) {
        const accuracy =
          STATS.win / STATS.total * 100;

        console.log(
          `[STATS] ${STATS.win}/${STATS.total} | ` +
          `Accuracy: ${accuracy.toFixed(2)}%`
        );
      }
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 10. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace('Z', '+07:00');
}

// =====================================================
// 11. API RESPONSE
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
  } = calculateFinal(
    seq,
    current._id ?? current.id
  );

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

    dotincay: `${Math.round(confidence)}%`,

    uptime: formatTimeVN()
  };
}

// =====================================================
// 12. API ROUTES
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
// 13. BOOT SERVER
// =====================================================

initDataFile();

// Restore existing history
{
  const h = readHistory();

  lastId = h.length
    ? Math.max(...h.map(s => s.id))
    : 0;

  // Restore a prediction for the next unseen session
  if (h.length > 0) {
    const seq = h.map(s => s.result);

    const current = h[h.length - 1];

    const result = calculateFinal(
      seq,
      current._id ?? current.id
    );

    lastPredict = {
      predict: result.final,
      caution: result.caution,
      components: result.components
    };
  }

  console.log(
    `[BOOT] Restored ${h.length} historical sessions`
  );
}

// Initial synchronization
syncData();

// Automatic synchronization
const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

// Start server
app.listen(PORT, () => {
  console.log(
    `[BOOT] Port ${PORT} — ${formatTimeVN()}`
  );

  console.log(
    '[BOOT] Bayesian + Markov + Pattern + Streak + Bandit + Entropy'
  );

  console.log('[BOOT] MD5 removed');
  console.log('[BOOT] Random prediction disabled');
  console.log('[BOOT] Adaptive learning enabled');
});
