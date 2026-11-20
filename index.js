
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '1mb' }));

// =====================================================
// 1. CONFIGURATION
// =====================================================

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 200;
const PORT = process.env.PORT || 3000;

const clamp = (v, min, max) =>
  Math.max(min, Math.min(max, v));

// =====================================================
// 2. CORS
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

// =====================================================
// 3. DATA STORAGE
// =====================================================

let historyData = [];

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, '[]', 'utf8');
    console.log('[INIT] Created history file');
  }

  try {
    const data = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    historyData = Array.isArray(data)
      ? data
          .filter(s =>
            Number.isFinite(Number(s.id)) &&
            [0, 1].includes(Number(s.result))
          )
          .sort((a, b) => Number(a.id) - Number(b.id))
          .slice(-MAX_HISTORY)
      : [];

    saveHistory();

    console.log(
      `[INIT] Restored ${historyData.length} sessions`
    );
  } catch (err) {
    console.error('[INIT ERROR]', err.message);
    historyData = [];
  }
}

function saveHistory() {
  const tempFile = DATA_FILE + '.tmp';

  fs.writeFileSync(
    tempFile,
    JSON.stringify(historyData, null, 2),
    'utf8'
  );

  fs.renameSync(tempFile, DATA_FILE);
}

function readHistory() {
  return historyData;
}

// =====================================================
// 4. SOURCE API
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
  const id = Number(
    raw.id ?? raw.session ?? raw.issue
  );

  let rawDices =
    raw.dices ??
    raw.dice ??
    (Array.isArray(raw.result) ? raw.result : []);

  if (!Array.isArray(rawDices)) {
    rawDices = [];
  }

  const dices = rawDices.map(Number);

  const validDices =
    dices.length > 0 &&
    dices.every(d => Number.isInteger(d) && d >= 1 && d <= 6);

  const sum = validDices
    ? dices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? 0);

  const sourceResult = String(
    raw.resultTruyenThong ??
    raw.bigSmall ??
    (typeof raw.result === 'string' ? raw.result : '') ??
    ''
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
  } else if (validDices || Number.isFinite(sum)) {
    result = sum >= 11 ? 1 : 0;
  } else {
    throw new Error('Invalid session result');
  }

  return {
    id,
    _id: String(raw._id ?? id),
    dices: validDices ? dices : [],
    sum,
    result
  };
}

// =====================================================
// 5. AI ADAPTIVE V4 ENGINE
// =====================================================

// Các mô hình được đánh giá độc lập.
// Loss càng thấp thì mô hình càng được ưu tiên.

const MODEL_NAMES = [
  'bayes',
  'pattern',
  'markov',
  'streak',
  'trend'
];

const MODEL_PERFORMANCE = {};

for (const name of MODEL_NAMES) {
  MODEL_PERFORMANCE[name] = {
    loss: 0.25,
    samples: 0
  };
}

// =====================================================
// MODEL 1: ADAPTIVE BAYESIAN
// =====================================================

function adaptiveBayes(seq) {
  if (!seq.length) return 0.5;

  let tai = 2;
  let xiu = 2;

  const halfLife = 40;

  for (let i = 0; i < seq.length; i++) {
    const age = seq.length - 1 - i;

    const weight = Math.pow(
      0.5,
      age / halfLife
    );

    if (seq[i] === 1) {
      tai += weight;
    } else {
      xiu += weight;
    }
  }

  return tai / (tai + xiu);
}

// =====================================================
// MODEL 2: PATTERN RECOGNITION
// =====================================================

function patternRecognition(seq) {
  if (seq.length < 8) return 0.5;

  let weightedTai = 0;
  let weightedTotal = 0;

  const maxPattern = Math.min(5, seq.length - 1);

  for (let k = 1; k <= maxPattern; k++) {
    const context = seq.slice(-k).join('');

    let tai = 0;
    let xiu = 0;

    for (let i = k; i < seq.length; i++) {
      const previous = seq
        .slice(i - k, i)
        .join('');

      if (previous !== context) continue;

      if (seq[i] === 1) tai++;
      else xiu++;
    }

    const total = tai + xiu;

    if (total === 0) continue;

    const probability =
      (tai + 2) / (total + 4);

    const reliability =
      total / (total + 8);

    const weight = reliability * (1 + k * 0.15);

    weightedTai += probability * weight;
    weightedTotal += weight;
  }

  if (weightedTotal === 0) return 0.5;

  return weightedTai / weightedTotal;
}

// =====================================================
// MODEL 3: MARKOV TRANSITION
// =====================================================

function markovProbability(seq, order) {
  if (seq.length <= order) return 0.5;

  const context = seq
    .slice(-order)
    .join('');

  let tai = 2;
  let xiu = 2;

  for (let i = order; i < seq.length; i++) {
    const previous = seq
      .slice(i - order, i)
      .join('');

    if (previous !== context) continue;

    if (seq[i] === 1) tai++;
    else xiu++;
  }

  return tai / (tai + xiu);
}

function markovModel(seq) {
  const p1 = markovProbability(seq, 1);
  const p2 = markovProbability(seq, 2);
  const p3 = markovProbability(seq, 3);

  return (
    p1 * 0.25 +
    p2 * 0.35 +
    p3 * 0.40
  );
}

// =====================================================
// MODEL 4: STREAK / ALTERNATION
// =====================================================

function getCurrentStreak(seq) {
  if (!seq.length) {
    return { value: 0, length: 0 };
  }

  const value = seq[seq.length - 1];
  let length = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] !== value) break;
    length++;
  }

  return { value, length };
}

function streakModel(seq) {
  if (seq.length < 8) return 0.5;

  const current = getCurrentStreak(seq);

  const targetBucket = Math.min(
    current.length,
    6
  );

  let tai = 2;
  let xiu = 2;

  for (let i = 1; i < seq.length; i++) {
    let length = 1;

    for (let j = i - 2; j >= 0; j--) {
      if (seq[j] !== seq[i - 1]) break;
      length++;
    }

    const bucket = Math.min(length, 6);

    if (
      bucket !== targetBucket ||
      seq[i - 1] !== current.value
    ) {
      continue;
    }

    if (seq[i] === 1) tai++;
    else xiu++;
  }

  const empirical = tai / (tai + xiu);

  const recent = adaptiveBayes(
    seq.slice(-40)
  );

  const support = (tai + xiu - 4);

  const reliability = clamp(
    support / 12,
    0,
    0.65
  );

  return (
    empirical * reliability +
    recent * (1 - reliability)
  );
}

// =====================================================
// MODEL 5: TREND ADAPTIVE
// =====================================================

function trendModel(seq) {
  if (seq.length < 10) return 0.5;

  const recent = seq.slice(-20);

  const medium = seq.slice(-60);

  const long = seq.slice(-120);

  const pRecent = adaptiveBayes(recent);
  const pMedium = adaptiveBayes(medium);
  const pLong = adaptiveBayes(long);

  return (
    pRecent * 0.55 +
    pMedium * 0.30 +
    pLong * 0.15
  );
}

// =====================================================
// 6. REGIME / ENTROPY DETECTOR
// =====================================================

function calculateEntropy(seq) {
  if (seq.length < 10) return 1;

  const sample = seq.slice(-40);

  let transitions = [
    [0, 0],
    [0, 0]
  ];

  for (let i = 1; i < sample.length; i++) {
    transitions[sample[i - 1]][sample[i]]++;
  }

  let totalEntropy = 0;
  let totalCount = 0;

  for (let i = 0; i < 2; i++) {
    const count =
      transitions[i][0] + transitions[i][1];

    if (!count) continue;

    const p0 = transitions[i][0] / count;
    const p1 = transitions[i][1] / count;

    let h = 0;

    if (p0 > 0) h -= p0 * Math.log2(p0);
    if (p1 > 0) h -= p1 * Math.log2(p1);

    totalEntropy += h * count;
    totalCount += count;
  }

  return totalCount
    ? totalEntropy / totalCount
    : 1;
}

function detectRegime(seq) {
  const entropy = calculateEntropy(seq);

  const short = seq.slice(-20);
  const medium = seq.slice(-60);

  const shortP = adaptiveBayes(short);
  const mediumP = adaptiveBayes(medium);

  const drift = Math.abs(shortP - mediumP);

  const streak = getCurrentStreak(seq);

  const chaotic =
    entropy > 0.90 ||
    (drift > 0.25 && entropy > 0.75);

  return {
    entropy,
    drift,
    streak: streak.length,
    chaotic
  };
}

// =====================================================
// 7. ADAPTIVE BANDIT
// =====================================================

// Mỗi mô hình được thưởng/phạt riêng dựa trên
// xác suất mà nó đưa ra trước khi biết kết quả.

function updateModelPerformance(prediction, actual) {
  for (const name of MODEL_NAMES) {
    const model = MODEL_PERFORMANCE[name];

    const p = clamp(
      prediction[name],
      0,
      1
    );

    const loss = Math.pow(
      p - actual,
      2
    );

    model.loss =
      model.samples === 0
        ? loss
        : model.loss * 0.95 + loss * 0.05;

    model.samples++;
  }
}

function getAdaptiveWeights() {
  const raw = {};
  let total = 0;

  for (const name of MODEL_NAMES) {
    const model = MODEL_PERFORMANCE[name];

    // Khi chưa có dữ liệu đánh giá,
    // các mô hình được đối xử công bằng.

    const performance =
      model.samples < 5
        ? 1
        : Math.exp(-3 * model.loss);

    raw[name] = Math.max(
      0.05,
      performance
    );

    total += raw[name];
  }

  const weights = {};

  for (const name of MODEL_NAMES) {
    weights[name] = raw[name] / total;
  }

  return weights;
}

// =====================================================
// 8. FINAL AI DECISION
// =====================================================

function calculateFinal(seq) {
  const predictions = {
    bayes: adaptiveBayes(seq),
    pattern: patternRecognition(seq),
    markov: markovModel(seq),
    streak: streakModel(seq),
    trend: trendModel(seq)
  };

  const weights = getAdaptiveWeights();

  let probability = 0;
  let disagreement = 0;

  for (const name of MODEL_NAMES) {
    probability +=
      predictions[name] * weights[name];
  }

  for (const name of MODEL_NAMES) {
    disagreement +=
      Math.abs(
        predictions[name] - probability
      ) * weights[name];
  }

  const regime = detectRegime(seq);

  // Khi thiếu dữ liệu, giảm độ tự tin.
  if (seq.length < 30) {
    probability =
      0.5 + (probability - 0.5) *
      (seq.length / 30);
  }

  // Khi biến động mạnh, kéo dự đoán về trung tính.
  const cautionFactor = regime.chaotic
    ? 0.55
    : 1;

  probability =
    0.5 + (probability - 0.5) * cautionFactor;

  probability = clamp(
    probability,
    0.01,
    0.99
  );

  const final =
    probability >= 0.5 ? 'TÀI' : 'XỈU';

  // Confidence là chỉ số nội bộ tham khảo,
  // không phải xác suất thắng đã được chứng minh.
  const strength = Math.abs(
    probability - 0.5
  );

  const confidence = clamp(
    50 +
    strength * 100 -
    disagreement * 30 -
    (regime.chaotic ? 8 : 0),
    50,
    80
  );

  const caution =
    seq.length < 30 ||
    regime.chaotic ||
    disagreement > 0.20 ||
    strength < 0.04;

  return {
    final,
    confidence: Math.round(confidence * 100) / 100,
    caution,
    probability,
    predictions,
    weights,
    regime,
    disagreement
  };
}

// =====================================================
// 9. PREDICTION STATE
// =====================================================

let lastPredict = null;
let lastId = 0;

const STATS = {
  total: 0,
  win: 0,
  lose: 0
};

const COMPARE = [];

const CAUTION = {
  data: []
};

// Tạo dự đoán cho phiên tiếp theo.
// Chỉ dùng dữ liệu đã có tại thời điểm dự đoán.

function createNextPrediction() {
  if (!historyData.length) {
    lastPredict = null;
    return null;
  }

  const seq = historyData.map(
    s => Number(s.result)
  );

  const result = calculateFinal(seq);

  lastPredict = {
    predict: result.final,
    caution: result.caution,
    confidence: result.confidence,
    probability: result.probability,
    predictions: result.predictions,
    targetId: Number(
      historyData[historyData.length - 1].id
    ) + 1
  };

  return lastPredict;
}

// =====================================================
// 10. PROCESS EACH NEW SESSION
// =====================================================

function processSession(session) {
  // Chấm dự đoán cũ trước khi đưa kết quả mới
  // vào lịch sử. Điều này tránh rò rỉ dữ liệu.

  if (lastPredict) {
    const actual =
      session.result === 1 ? 1 : 0;

    const real =
      actual === 1 ? 'TÀI' : 'XỈU';

    const win =
      lastPredict.predict === real;

    // Học riêng từng thuật toán.
    updateModelPerformance(
      lastPredict.predictions,
      actual
    );

    STATS.total++;

    if (win) STATS.win++;
    else STATS.lose++;

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

    console.log(
      `[LEARN] Session ${session.id} | ` +
      `Predict: ${lastPredict.predict} | ` +
      `Actual: ${real} | ` +
      `${win ? 'WIN' : 'LOSE'}`
    );
  }

  // Sau khi chấm điểm mới thêm kết quả vào lịch sử.

  historyData.push(session);

  historyData.sort(
    (a, b) => Number(a.id) - Number(b.id)
  );

  if (historyData.length > MAX_HISTORY) {
    historyData = historyData.slice(-MAX_HISTORY);
  }

  saveHistory();

  // Tạo dự đoán cho phiên tiếp theo.
  createNextPrediction();
}

// =====================================================
// 11. SYNC ENGINE
// =====================================================

let syncInProgress = false;

async function syncData() {
  if (syncInProgress) return;

  syncInProgress = true;

  try {
    const raw = await fetchSource();

    const list = Array.isArray(raw)
      ? raw
      : raw.data ??
        raw.sessions ??
        raw.list ??
        [];

    if (!Array.isArray(list) || !list.length) {
      return;
    }

    const fresh = list
      .map(rawSession => {
        try {
          return normalizeSession(rawSession);
        } catch (err) {
          console.error(
            '[NORMALIZE ERROR]',
            err.message
          );

          return null;
        }
      })
      .filter(s =>
        s &&
        Number.isSafeInteger(s.id) &&
        s.id > lastId &&
        [0, 1].includes(s.result)
      )
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    const knownIds = new Set(
      historyData.map(s => Number(s.id))
    );

    let added = 0;

    for (const session of fresh) {
      if (knownIds.has(session.id)) continue;

      processSession(session);

      knownIds.add(session.id);

      lastId = Math.max(
        lastId,
        session.id
      );

      added++;
    }

    if (added > 0) {
      console.log(
        `[SYNC] +${added} sessions | ` +
        `History ${historyData.length}/${MAX_HISTORY}`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 12. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  return new Date().toLocaleString(
    'sv-SE',
    {
      timeZone: 'Asia/Ho_Chi_Minh'
    }
  ).replace(' ', 'T') + '+07:00';
}

// =====================================================
// 13. API RESPONSE
// =====================================================

// Giữ nguyên cấu trúc JSON cũ để frontend
// không phải sửa code.

function buildResponse() {
  const history = readHistory();

  if (!history.length) {
    return {
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime: formatTimeVN()
    };
  }

  const current =
    history[history.length - 1];

  // Dự đoán đã được tạo sau khi cập nhật
  // phiên hiện tại, không tính lại nhiều lần.

  const prediction =
    lastPredict || createNextPrediction();

  return {
    id: 'noname09',

    phienhientai: current.id,

    ketqua:
      current.result === 1 ? 'tai' : 'xiu',

    xucxac: current.dices,

    tong: current.sum,

    dudoanphien: current.id + 1,

    dudoan:
      prediction.predict === 'TÀI'
        ? 'tai'
        : 'xiu',

    dotincay:
      `${Math.round(prediction.confidence)}%`,

    uptime: formatTimeVN()
  };
}

// =====================================================
// 14. API ROUTES
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
// 15. OPTIONAL AI STATUS
// =====================================================

app.get('/ai-status', (req, res) => {
  try {
    const weights = getAdaptiveWeights();

    res.json({
      version: 'AI-ADAPTIVE-V4',
      history: historyData.length,
      maxHistory: MAX_HISTORY,

      stats: {
        ...STATS,
        accuracy: STATS.total
          ? Number(
              (STATS.win / STATS.total * 100).toFixed(2)
            )
          : 0
      },

      models: MODEL_NAMES.map(name => ({
        name,
        weight: Number(
          (weights[name] * 100).toFixed(2)
        ),
        loss: Number(
          MODEL_PERFORMANCE[name].loss.toFixed(4)
        ),
        samples: MODEL_PERFORMANCE[name].samples
      })),

      lastPrediction: lastPredict
        ? {
            predict: lastPredict.predict,
            confidence: lastPredict.confidence,
            targetId: lastPredict.targetId
          }
        : null,

      uptime: formatTimeVN()
    });
  } catch (err) {
    res.status(500).json({
      error: 'AI status error'
    });
  }
});

// =====================================================
// 16. BOOT SERVER
// =====================================================

initDataFile();

lastId = historyData.length
  ? Math.max(
      ...historyData.map(s => Number(s.id))
    )
  : 0;

// Khôi phục dự đoán từ lịch sử hiện tại.
// Trọng số học trực tuyến được khởi tạo lại
// vì chưa lưu trạng thái hiệu suất ra file.

if (historyData.length > 0) {
  createNextPrediction();
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
  console.log('====================================');
  console.log(' AI ADAPTIVE V4');
  console.log('====================================');

  console.log(`[BOOT] Port: ${PORT}`);
  console.log(`[BOOT] History: ${MAX_HISTORY}`);
  console.log(`[BOOT] Poll: ${POLL_INTERVAL}ms`);

  console.log(
    '[BOOT] Bayesian + Pattern + Markov + Streak + Trend'
  );

  console.log(
    '[BOOT] Adaptive Bandit + Regime Detection'
  );

  console.log(`[BOOT] Time: ${formatTimeVN()}`);

  console.log('====================================');
});
