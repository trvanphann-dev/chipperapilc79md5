
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// Cho phép trang HTML ở domain khác gọi API này (CORS).
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
// 3. ALGORITHM ENGINE - DATA BASED, NO FIXED SIDE
// =====================================================

// Giới hạn xác suất để tránh giá trị bất thường.
function clampProbability(value) {
  return Math.max(0.01, Math.min(0.99, value));
}

// Tính tỷ lệ Tài có làm mượt.
// Prior luôn cân bằng 50/50, không thiên vị bên nào.
function smoothedProbability(seq, windowSize, priorWeight = 8) {
  const data = seq.slice(-windowSize);

  if (data.length === 0) {
    return 0.5;
  }

  const taiCount = data.reduce(
    (total, value) => total + (value === 1 ? 1 : 0),
    0
  );

  return (
    (taiCount + priorWeight * 0.5) /
    (data.length + priorWeight)
  );
}

// -----------------------------------------------------
// 3.1 BAYESIAN PROBABILITY
// -----------------------------------------------------

function bayesianProbability(seq) {
  if (!seq.length) return 0.5;

  const taiCount = seq.filter(v => v === 1).length;
  const xiuCount = seq.length - taiCount;

  // Prior Beta(1, 1): hai phía hoàn toàn cân bằng.
  return (taiCount + 1) / (seq.length + 2);
}

// -----------------------------------------------------
// 3.2 RECENT DATA PROBABILITY
// -----------------------------------------------------

function recentProbability(seq) {
  const p12 = smoothedProbability(seq, 12, 8);
  const p30 = smoothedProbability(seq, 30, 8);
  const p80 = smoothedProbability(seq, 80, 12);

  // Dữ liệu gần đây có trọng số lớn hơn,
  // nhưng vẫn giữ thông tin từ lịch sử dài hơn.
  return (
    p12 * 0.50 +
    p30 * 0.30 +
    p80 * 0.20
  );
}

// -----------------------------------------------------
// 3.3 MARKOV TRANSITION PROBABILITY
// -----------------------------------------------------

function markovProbability(seq) {
  if (seq.length < 2) return 0.5;

  const last = seq[seq.length - 1];

  let total = 0;
  let taiCount = 0;

  // Chỉ xét các lần chuyển đổi có trạng thái trước
  // giống trạng thái gần nhất hiện tại.
  for (let i = 0; i < seq.length - 1; i++) {
    if (seq[i] === last) {
      total++;

      if (seq[i + 1] === 1) {
        taiCount++;
      }
    }
  }

  // Làm mượt cân bằng để tránh mẫu quá nhỏ.
  return (taiCount + 2) / (total + 4);
}

// -----------------------------------------------------
// 3.4 PATTERN PROBABILITY
// -----------------------------------------------------

function patternProbability(seq) {
  if (seq.length < 3) return 0.5;

  let weightedProbability = 0;
  let totalWeight = 0;

  // Kiểm tra các mẫu có độ dài 1 đến 4.
  for (let contextLength = 1; contextLength <= 4; contextLength++) {
    if (seq.length <= contextLength) continue;

    const context = seq
      .slice(-contextLength)
      .join('');

    let matches = 0;
    let taiAfterMatch = 0;

    for (
      let i = 0;
      i <= seq.length - contextLength - 1;
      i++
    ) {
      const oldContext = seq
        .slice(i, i + contextLength)
        .join('');

      if (oldContext === context) {
        matches++;

        if (seq[i + contextLength] === 1) {
          taiAfterMatch++;
        }
      }
    }

    if (matches === 0) continue;

    // Prior cân bằng, giảm ảnh hưởng mẫu ít xuất hiện.
    const probability =
      (taiAfterMatch + 2) / (matches + 4);

    // Mẫu dài hơn được ưu tiên nhẹ,
    // nhưng không được áp đảo mẫu ngắn.
    const weight = Math.min(matches, 10) *
      (1 + contextLength * 0.15);

    weightedProbability += probability * weight;
    totalWeight += weight;
  }

  if (totalWeight === 0) return 0.5;

  return weightedProbability / totalWeight;
}

// -----------------------------------------------------
// 3.5 ENSEMBLE MODEL
// -----------------------------------------------------

function predictProbability(seq) {
  if (!seq.length) {
    return {
      probability: 0.5,
      components: {}
    };
  }

  const pBayes = bayesianProbability(seq);
  const pRecent = recentProbability(seq);
  const pMarkov = markovProbability(seq);
  const pPattern = patternProbability(seq);

  // Kết hợp các mô hình theo trọng số cố định,
  // áp dụng đối xứng cho Tài và Xỉu.
  const modelProbability =
    pBayes * 0.15 +
    pRecent * 0.35 +
    pMarkov * 0.25 +
    pPattern * 0.25;

  // Phương án B:
  // kết hợp xác suất mô hình với dữ liệu gần đây.
  // Không cộng điểm cố định cho Tài hoặc Xỉu.
  const finalProbability =
    modelProbability * 0.70 +
    pRecent * 0.30;

  return {
    probability: clampProbability(finalProbability),

    components: {
      bayesian: pBayes,
      recent: pRecent,
      markov: pMarkov,
      pattern: pPattern,
      ensemble: modelProbability
    }
  };
}

// =====================================================
// 4. WALK-FORWARD BACKTEST
// =====================================================

// Mỗi dự đoán lịch sử chỉ được sử dụng dữ liệu xuất hiện
// trước phiên cần dự đoán.
// Không được dùng kết quả tương lai để dự đoán quá khứ.

function buildWalkForwardResults(seq, maxSamples = 100) {
  const results = [];

  // Cần tối thiểu 20 phiên trước khi bắt đầu kiểm tra.
  const minimumTraining = 20;

  if (seq.length <= minimumTraining) {
    return results;
  }

  const startIndex = Math.max(
    minimumTraining,
    seq.length - maxSamples
  );

  for (let i = startIndex; i < seq.length; i++) {
    const trainingData = seq.slice(0, i);
    const actual = seq[i];

    const prediction = predictProbability(trainingData);

    const predictedSide =
      prediction.probability >= 0.5 ? 1 : 0;

    results.push({
      predicted: predictedSide,
      actual,
      correct: predictedSide === actual
    });
  }

  return results;
}

// =====================================================
// 5. CONFIDENCE CALIBRATION
// =====================================================

// Độ tin cậy được tính từ kết quả kiểm tra lịch sử.
// Không lấy độ lệch xác suất làm độ chính xác.
// Có prior 50/50 để hạn chế dao động khi số mẫu ít.

function calculateConfidence(seq) {
  const results = buildWalkForwardResults(seq, 100);

  if (results.length === 0) {
    return 50;
  }

  const correctCount = results.filter(
    item => item.correct
  ).length;

  // Prior tương đương 12 mẫu, trong đó 6 mẫu đúng.
  const priorSamples = 12;
  const priorCorrect = 6;

  const accuracy =
    (correctCount + priorCorrect) /
    (results.length + priorSamples);

  return Math.round(
    Math.max(0, Math.min(100, accuracy * 100))
  );
}

// =====================================================
// 6. ENTROPY / CAUTION
// =====================================================

function entropy(seq, windowSize = 30) {
  const data = seq.slice(-windowSize);

  if (data.length < windowSize) {
    return 0;
  }

  const taiCount = data.filter(v => v === 1).length;
  const p1 = taiCount / data.length;
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
  if (seq.length < 30) {
    return false;
  }

  // Entropy cao thể hiện chuỗi cân bằng và khó phân biệt.
  return entropy(seq, 30) > 0.92;
}

// =====================================================
// 7. FINAL PREDICTION
// =====================================================

function calculateFinal(seq) {
  const result = predictProbability(seq);

  const probability = result.probability;

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  const confidence = calculateConfidence(seq);
  const caution = chaosBlock(seq);

  return {
    final,
    confidence,
    caution,
    probability,
    components: result.components
  };
}

// =====================================================
// 8. STATE: DỰ ĐOÁN, THỐNG KÊ, CAUTION
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

// Xử lý phiên mới và chấm điểm dự đoán phiên trước.
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
// 9. SYNC
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

    for (const s of fresh) {
      if (knownIds.has(s.id)) continue;

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

// Khôi phục ID phiên mới nhất từ lịch sử.
{
  const history = readHistory();

  lastId = history.length
    ? Number(history[history.length - 1].id)
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
    '[BOOT] Bayesian + Recent + Markov + Pattern + Walk-forward loaded'
  );
});
