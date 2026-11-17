
const express = require('express');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const app = express();
app.use(express.json());

// Cho phép trang HTML ở domain khác gọi API này (CORS).
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

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

// Xoá file lịch sử rồi tạo lại file rỗng (dùng khi reset).
function resetHistory() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      fs.unlinkSync(DATA_FILE);
    }
  } catch (err) {
    console.error('[RESET ERROR]', err.message);
  }

  fs.writeFileSync(DATA_FILE, '[]', 'utf8');
  CAUTION.data = [];

  console.log('[RESET] History file deleted and recreated');
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
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}


// =====================================================
// 3. ALGORITHM ENGINE - ENSEMBLE + WALK-FORWARD CALIBRATION
// =====================================================

// Giới hạn xác suất.
function clampProbability(value) {
  return Math.max(0.01, Math.min(0.99, value));
}

// 1. BAYESIAN ADAPTIVE
// Dữ liệu gần hiện tại có trọng số cao hơn.
function bayesianProbability(seq, windowSize = 120) {
  const data = seq.slice(-windowSize);
  let alpha = 1;
  let beta = 1;

  for (let i = 0; i < data.length; i++) {
    const age = data.length - 1 - i;
    const weight = Math.pow(0.985, age);

    if (data[i] === 1) {
      alpha += weight;
    } else {
      beta += weight;
    }
  }

  return clampProbability(
    alpha / (alpha + beta)
  );
}

// 2. MARKOV MODEL
// Học xác suất chuyển trạng thái.
function markovProbability(seq) {
  if (seq.length < 2) {
    return bayesianProbability(seq);
  }

  const last = seq[seq.length - 1];

  let tai = 1;
  let xiu = 1;
  let samples = 0;

  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] !== last) continue;

    const age = seq.length - 1 - i;
    const weight = Math.pow(0.99, age);

    if (seq[i] === 1) {
      tai += weight;
    } else {
      xiu += weight;
    }

    samples++;
  }

  const markov = tai / (tai + xiu);
  const base = bayesianProbability(seq);

  const reliability = Math.min(
    0.75,
    samples / 16
  );

  return clampProbability(
    base * (1 - reliability) +
    markov * reliability
  );
}

// 3. PATTERN RECOGNITION
// Nhận diện chuỗi tương đồng trong lịch sử.
function patternProbability(seq) {
  const base = bayesianProbability(seq);

  const maxPatternLength = Math.min(
    5,
    seq.length - 1
  );

  if (maxPatternLength < 2) {
    return base;
  }

  let totalTai = 0;
  let totalXiu = 0;
  let totalMatches = 0;

  for (let length = 2; length <= maxPatternLength; length++) {
    const pattern = seq.slice(-length).join('');

    let tai = 0;
    let xiu = 0;

    for (let i = 0; i + length < seq.length; i++) {
      if (
        seq.slice(i, i + length).join('') !== pattern
      ) {
        continue;
      }

      const next = seq[i + length];

      const age = seq.length - 1 - (i + length);
      const weight = Math.pow(0.99, age);

      if (next === 1) {
        tai += weight;
      } else {
        xiu += weight;
      }
    }

    const matches = tai + xiu;

    if (matches > 0) {
      const patternWeight = Math.min(
        1,
        matches / 5
      );

      totalTai += tai * patternWeight;
      totalXiu += xiu * patternWeight;
      totalMatches += matches * patternWeight;
    }
  }

  if (totalMatches < 1) {
    return base;
  }

  const learned =
    (totalTai + 1) /
    (totalTai + totalXiu + 2);

  const reliability = Math.min(
    0.65,
    totalMatches / 12
  );

  return clampProbability(
    base * (1 - reliability) +
    learned * reliability
  );
}

// 4. RECENT STATISTICS
// Thống kê 20 phiên gần nhất.
function recentProbability(seq, windowSize = 20) {
  const data = seq.slice(-windowSize);

  if (!data.length) {
    return 0.5;
  }

  const tai = data.reduce(
    (sum, value) => sum + value,
    0
  );

  return clampProbability(
    (tai + 1) / (data.length + 2)
  );
}

// 5. ENSEMBLE MODEL
// Kết hợp nhiều mô hình dự đoán.
function predictProbability(seq) {
  if (!seq.length) {
    return 0.5;
  }

  const pBayes = bayesianProbability(seq);
  const pMarkov = markovProbability(seq);
  const pPattern = patternProbability(seq);
  const pRecent = recentProbability(seq, 20);

  const sampleFactor = Math.min(
    1,
    seq.length / 40
  );

  const weights = {
    bayes: 0.35,
    markov: 0.30 * sampleFactor,
    pattern: 0.20 * sampleFactor,
    recent: 0.15
  };

  const totalWeight = Object.values(weights)
    .reduce((sum, value) => sum + value, 0);

  const probability = (
    pBayes * weights.bayes +
    pMarkov * weights.markov +
    pPattern * weights.pattern +
    pRecent * weights.recent
  ) / totalWeight;

  return clampProbability(probability);
}

// 6. ENTROPY DETECTOR
// Phát hiện chuỗi có độ nhiễu cao.
function entropy(seq, windowSize = 20) {
  if (seq.length < windowSize) {
    return 0;
  }

  const data = seq.slice(-windowSize);

  const p1 = data.reduce(
    (sum, value) => sum + value,
    0
  ) / data.length;

  const p0 = 1 - p1;

  if (p1 === 0 || p1 === 1) {
    return 0;
  }

  return -(
    p1 * Math.log2(p1) +
    p0 * Math.log2(p0)
  );
}

function chaosBlock(seq) {
  return (
    seq.length >= 30 &&
    entropy(seq, 30) > 0.92
  );
}

// 7. WALK-FORWARD VALIDATION
// Kiểm định từng dự đoán bằng dữ liệu có trước thời điểm đó.
function buildWalkForwardResults(seq) {
  const results = [];
  const minimumTraining = 25;

  for (
    let i = minimumTraining;
    i < seq.length;
    i++
  ) {
    const training = seq.slice(0, i);
    const actual = seq[i];

    const probability = predictProbability(training);

    const predicted = probability >= 0.5 ? 1 : 0;

    results.push({
      probability,
      strength: Math.max(
        probability,
        1 - probability
      ),
      correct: predicted === actual
    });
  }

  return results;
}

// 8. CONFIDENCE CALIBRATION
// Tính độ tin cậy dựa trên tỷ lệ dự đoán đúng trong lịch sử.
function calculateConfidence(seq, currentProbability) {
  const historyResults = buildWalkForwardResults(seq);

  if (!historyResults.length) {
    return 50;
  }

  const currentStrength = Math.max(
    currentProbability,
    1 - currentProbability
  );

  let comparable = historyResults.filter(item =>
    Math.abs(
      item.strength - currentStrength
    ) <= 0.05
  );

  // Mở rộng phạm vi nếu số mẫu quá ít.
  if (comparable.length < 8) {
    comparable = historyResults.filter(item =>
      Math.abs(
        item.strength - currentStrength
      ) <= 0.10
    );
  }

  // Không đủ mẫu thì dùng toàn bộ lịch sử kiểm định.
  if (comparable.length < 5) {
    comparable = historyResults;
  }

  const wins = comparable.reduce(
    (sum, item) =>
      sum + (item.correct ? 1 : 0),
    0
  );

  const sampleCount = comparable.length;
  const observedAccuracy = wins / sampleCount;

  // Prior 50% để hạn chế dao động khi ít dữ liệu.
  const priorSamples = 12;

  const calibrated = (
    observedAccuracy * sampleCount +
    0.5 * priorSamples
  ) / (sampleCount + priorSamples);

  return Math.round(
    Math.max(
      1,
      Math.min(99, calibrated * 100)
    ) * 100
  ) / 100;
}

// 9. FINAL DECISION
function calculateFinal(seq, id) {
  const probability = predictProbability(seq);

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  const confidence = calculateConfidence(
    seq,
    probability
  );

  const caution =
    chaosBlock(seq) ||
    confidence < 55;

  return {
    final,
    confidence,
    caution,
    probability: Math.round(
      probability * 10000
    ) / 10000
  };
}


// =====================================================
// 4. STATE: DỰ ĐOÁN, THỐNG KÊ, CAUTION, RESET
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

// Xử lý 1 phiên mới.
function processLatest(session) {
  const seq = readHistory().map(h => h.result);

  const {
    final,
    confidence,
    caution
  } = calculateFinal(
    seq,
    session._id ?? session.id
  );

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

    // Reset ẩn theo 10 lần Lưu Ý.
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

        if (loseCount >= winCount) {
          resetHistory();
        }
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
// 5. SYNC
// =====================================================

let syncInProgress = false;

async function syncData() {
  if (syncInProgress) {
    return;
  }

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

    if (!fresh.length) {
      return;
    }

    const history = readHistory();

    const knownIds = new Set(
      history.map(h => h.id)
    );

    const added = [];

    for (const s of fresh) {
      if (knownIds.has(s.id)) {
        continue;
      }

      history.push(s);
      knownIds.add(s.id);
      added.push(s);
    }

    lastId = fresh[fresh.length - 1].id;

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
// 6. VIETNAM TIME
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
// 7. API RESPONSE
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

// Khôi phục trạng thái từ lịch sử đang có.
{
  const h = readHistory();

  lastId = h.length
    ? h[h.length - 1].id
    : 0;
}

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
    '[BOOT] Ensemble + Walk-Forward Calibration engine loaded'
  );
});
