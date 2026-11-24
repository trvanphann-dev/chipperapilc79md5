
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

// =====================================================
// 1. CONFIGURATION
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
const MODEL_FILE = path.join(__dirname, 'algorithm_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const PORT = process.env.PORT || 3000;

// =====================================================
// 2. GLOBAL STATE
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
// 3. STORAGE
// =====================================================

function atomicWrite(file, data) {
  const temp = file + '.tmp';

  fs.writeFileSync(
    temp,
    JSON.stringify(data, null, 2),
    'utf8'
  );

  fs.renameSync(temp, file);
}

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, '[]', 'utf8');
    console.log('[INIT] Created history');
  } else {
    console.log('[INIT] Existing history preserved');
  }
}

function readHistory() {
  try {
    const parsed = JSON.parse(
      fs.readFileSync(DATA_FILE, 'utf8') || '[]'
    );

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[READ ERROR]', err.message);
    return [];
  }
}

function writeHistory(history) {
  atomicWrite(DATA_FILE, history);
}

// Giữ chức năng reset thủ công.
// Không tự động xóa lịch sử khi dự đoán sai.
function resetHistory() {
  writeHistory([]);

  lastPredict = null;
  lastId = 0;

  CAUTION.data = [];

  console.log('[RESET] History cleared');
}

// =====================================================
// 4. SOURCE API
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

// =====================================================
// 5. NORMALIZE SESSION
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

  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) {
    result = 1;
  } else if (['XIU', 'XỈU', 'SMALL'].includes(sourceResult)) {
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
// 6. BAYESIAN ENGINE
// =====================================================

class BayesianModel {
  predict(seq) {
    if (!seq.length) return 0.5;

    const recent = seq.slice(-100);

    const ones = recent.reduce(
      (a, b) => a + b,
      0
    );

    return (ones + 1) / (recent.length + 2);
  }
}

// =====================================================
// 7. MARKOV CHAIN ENGINE
// =====================================================

class HMMModel {
  predict(seq) {
    if (seq.length < 5) return 0.5;

    let numerator = 0;
    let denominator = 0;

    const maxOrder = Math.min(3, seq.length - 1);

    for (let order = 1; order <= maxOrder; order++) {
      const context = seq.slice(-order);

      let total = 0;
      let ones = 0;

      for (
        let i = order;
        i < seq.length;
        i++
      ) {
        let matched = true;

        for (let j = 0; j < order; j++) {
          if (seq[i - order + j] !== context[j]) {
            matched = false;
            break;
          }
        }

        if (matched) {
          total++;
          ones += seq[i];
        }
      }

      if (!total) continue;

      const p = (ones + 2) / (total + 4);

      const weight = total / (total + 5);

      numerator += p * weight;
      denominator += weight;
    }

    return denominator
      ? numerator / denominator
      : 0.5;
  }
}

// =====================================================
// 8. PATTERN RECOGNITION ENGINE
// =====================================================

function patternAnalysis(seq) {
  if (seq.length < 8) {
    return {
      probability: 0.5,
      support: 0
    };
  }

  const recent = seq.slice(-100);

  const base =
    recent.reduce((a, b) => a + b, 0) /
    recent.length;

  const maxLength = Math.min(6, seq.length - 1);

  let numerator = 0;
  let denominator = 0;
  let support = 0;

  for (let length = 2; length <= maxLength; length++) {
    const pattern = seq.slice(-length);

    let count = 0;
    let ones = 0;

    for (
      let end = length - 1;
      end < seq.length - 1;
      end++
    ) {
      let matched = true;

      for (let j = 0; j < length; j++) {
        if (
          seq[end - length + 1 + j] !== pattern[j]
        ) {
          matched = false;
          break;
        }
      }

      if (matched) {
        count++;
        ones += seq[end + 1];
      }
    }

    if (!count) continue;

    const p = (ones + base * 6) / (count + 6);

    const weight = count / (count + 6);

    numerator += p * weight;
    denominator += weight;
    support += count;
  }

  return {
    probability: denominator
      ? numerator / denominator
      : 0.5,

    support
  };
}

// =====================================================
// 9. STREAK DETECTOR
// =====================================================

function getCurrentStreak(seq) {
  if (!seq.length) {
    return {
      side: null,
      length: 0
    };
  }

  const side = seq[seq.length - 1];

  let length = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === side) {
      length++;
    } else {
      break;
    }
  }

  return {
    side,
    length
  };
}

function streakBucket(length) {
  if (length <= 1) return 1;
  if (length === 2) return 2;
  if (length === 3) return 3;
  if (length === 4) return 4;

  return 5;
}

// =====================================================
// 10. ADVANCED REVERSAL ENGINE
// =====================================================

class ReversalEngine {

  analyze(seq) {
    if (seq.length < 10) {
      return {
        probability: 0.5,
        reversalProbability: 0.5,
        continuationProbability: 0.5,
        support: 0,
        streak: 0,
        side: null,
        signal: 'INSUFFICIENT_DATA'
      };
    }

    const current = getCurrentStreak(seq);

    const side = current.side;
    const streak = current.length;

    const bucket = streakBucket(streak);

    let total = 0;
    let reverseCount = 0;
    let continueCount = 0;

    // Tìm các trường hợp lịch sử có:
    // - Cùng phía Tài/Xỉu
    // - Cùng nhóm độ dài cầu
    // - Có kết quả tiếp theo để kiểm chứng

    for (let i = 1; i < seq.length - 1; i++) {
      if (seq[i] !== side) continue;

      let historicalStreak = 1;

      for (let j = i - 1; j >= 0; j--) {
        if (seq[j] === seq[i]) {
          historicalStreak++;
        } else {
          break;
        }
      }

      if (
        streakBucket(historicalStreak) !== bucket
      ) {
        continue;
      }

      const next = seq[i + 1];

      total++;

      if (next !== side) {
        reverseCount++;
      } else {
        continueCount++;
      }
    }

    // Chỉ sử dụng dữ liệu lịch sử gần đây
    // để hạn chế ảnh hưởng của mẫu quá cũ.
    // Mỗi điểm dữ liệu là một vị trí kết thúc cầu.
    //
    // Làm mượt bằng 6 mẫu giả định cân bằng.
    // Không sử dụng random.

    const reversalProbability =
      (reverseCount + 3) / (total + 6);

    const continuationProbability =
      (continueCount + 3) / (total + 6);

    // Chuyển xác suất đảo chiều thành xác suất Tài.
    const probability = side === 1
      ? 1 - reversalProbability
      : reversalProbability;

    let signal = 'NEUTRAL';

    if (total >= 8) {
      if (reversalProbability >= 0.60) {
        signal = 'REVERSAL';
      } else if (reversalProbability <= 0.40) {
        signal = 'CONTINUATION';
      }
    }

    return {
      probability,
      reversalProbability,
      continuationProbability,
      support: total,
      streak,
      side,
      bucket,
      signal
    };
  }
}

// =====================================================
// 11. ENTROPY DETECTOR
// =====================================================

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

  // Entropy cao không tự động đồng nghĩa với cầu đảo.
  // Chỉ sử dụng như một chỉ báo bất định.
  return entropy(seq, 30) > 0.92;
}

// =====================================================
// 12. ADAPTIVE BANDIT
// =====================================================

class Bandit {
  constructor() {
    this.score = {
      bayes: 1,
      markov: 1,
      pattern: 1,
      reversal: 1
    };

    this.load();
  }

  load() {
    try {
      if (!fs.existsSync(MODEL_FILE)) return;

      const data = JSON.parse(
        fs.readFileSync(MODEL_FILE, 'utf8')
      );

      if (data.score) {
        for (const name of Object.keys(this.score)) {
          const value = Number(data.score[name]);

          if (
            Number.isFinite(value) &&
            value >= 0.4 &&
            value <= 2
          ) {
            this.score[name] = value;
          }
        }
      }

      console.log('[LEARNING] Model weights restored');

    } catch (err) {
      console.error('[MODEL LOAD ERROR]', err.message);
    }
  }

  save() {
    try {
      atomicWrite(MODEL_FILE, {
        score: this.score,
        updatedAt: new Date().toISOString()
      });
    } catch (err) {
      console.error('[MODEL SAVE ERROR]', err.message);
    }
  }

  weight(name) {
    return this.score[name] ?? 1;
  }

  reward(name, predicted, actual) {
    if (
      !Number.isFinite(predicted) ||
      !Number.isFinite(actual)
    ) {
      return;
    }

    const old = this.weight(name);

    const error = Math.pow(
      predicted - actual,
      2
    );

    // Brier error:
    // 0 = dự đoán xác suất hoàn hảo
    // 0.25 = dự đoán 50/50
    // 1 = dự đoán xác suất sai hoàn toàn

    const updated =
      old * Math.exp(0.12 * (0.25 - error));

    this.score[name] = Math.max(
      0.4,
      Math.min(2, updated)
    );

    this.save();
  }
}

// =====================================================
// 13. INITIALIZE MODELS
// =====================================================

const bayes = new BayesianModel();
const hmm = new HMMModel();
const reversalEngine = new ReversalEngine();
const bandit = new Bandit();

// =====================================================
// 14. ENSEMBLE PREDICTION
// =====================================================

function calculateFinal(seq, id) {
  const n = seq.length;

  const reversal = reversalEngine.analyze(seq);

  const pattern = patternAnalysis(seq);

  const components = {
    bayes: bayes.predict(seq),
    markov: hmm.predict(seq),
    pattern: pattern.probability,
    reversal: reversal.probability
  };

  let numerator = 0;
  let denominator = 0;

  for (const name of Object.keys(components)) {
    const weight = bandit.weight(name);

    numerator += components[name] * weight;
    denominator += weight;
  }

  const probability = denominator
    ? numerator / denominator
    : 0.5;

  const final = probability >= 0.5
    ? 'TÀI'
    : 'XỈU';

  // Mức độ đồng thuận giữa các mô hình
  const agreement = Object.values(components)
    .filter(p =>
      (probability >= 0.5 && p >= 0.5) ||
      (probability < 0.5 && p < 0.5)
    ).length / 4;

  // Hệ số dữ liệu
  const dataFactor = Math.min(1, n / 100);

  // Bằng chứng lịch sử của mô hình đảo cầu
  const reversalFactor = Math.min(
    1,
    reversal.support / 20
  );

  const strength = Math.abs(probability - 0.5);

  // Chỉ số tham khảo, không phải xác suất thắng đã hiệu chuẩn
  const confidence = Math.round(
    Math.min(
      75,
      50 +
      strength * 100 *
      agreement *
      dataFactor *
      (0.7 + 0.3 * reversalFactor)
    ) * 100
  ) / 100;

  const caution =
    n < 30 ||
    agreement < 0.5 ||
    Math.abs(probability - 0.5) < 0.06 ||
    reversal.support < 4;

  return {
    final,
    confidence,
    caution,
    probability,
    components,
    agreement,
    reversal,
    pattern,
    entropy: entropy(seq, 30)
  };
}

// =====================================================
// 15. PROCESS LATEST SESSION
// =====================================================

function processLatest(session) {
  const seq = readHistory().map(h => h.result);

  // Chấm dự đoán của phiên trước
  if (lastPredict) {
    const actual = session.result;

    const real = actual === 1
      ? 'TÀI'
      : 'XỈU';

    const win = lastPredict.predict === real;

    STATS.total++;

    if (win) {
      STATS.win++;
    } else {
      STATS.lose++;
    }

    // Mỗi mô hình được đánh giá độc lập
    if (lastPredict.components) {
      for (const name of [
        'bayes',
        'markov',
        'pattern',
        'reversal'
      ]) {
        bandit.reward(
          name,
          lastPredict.components[name],
          actual
        );
      }
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

    // Giữ thống kê CAUTION cũ
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);

      if (CAUTION.data.length > 10) {
        CAUTION.data.shift();
      }
    }

    console.log(
      `[RESULT] Session ${session.id} | ` +
      `${lastPredict.predict} | ${real} | ` +
      `${win ? 'WIN' : 'LOSE'}`
    );
  }

  // Dự đoán phiên tiếp theo
  const result = calculateFinal(
    seq,
    session._id ?? session.id
  );

  lastPredict = {
    predict: result.final,
    caution: result.caution,
    components: result.components
  };

  console.log(
    `[PREDICT] ${session.id + 1} | ${result.final} | ` +
    `Probability: ${(result.probability * 100).toFixed(2)}%`
  );

  console.log(
    `[REVERSAL] Streak: ${result.reversal.streak} | ` +
    `Support: ${result.reversal.support} | ` +
    `Reverse: ${(result.reversal.reversalProbability * 100).toFixed(1)}% | ` +
    `Signal: ${result.reversal.signal}`
  );

  return result;
}

// =====================================================
// 16. DATA SYNCHRONIZATION
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

    for (const session of fresh) {
      if (knownIds.has(session.id)) continue;

      history.push(session);

      knownIds.add(session.id);

      history.sort((a, b) => a.id - b.id);

      history = history.slice(-MAX_HISTORY);

      // Lưu dữ liệu mới trước khi học
      writeHistory(history);

      lastId = session.id;

      // Học kết quả vừa nhận và tạo dự đoán mới
      processLatest(session);

      added++;
    }

    if (added > 0) {
      console.log(
        `[SYNC] +${added} sessions | Total ${history.length}`
      );

      console.log(
        '[WEIGHTS]',
        JSON.stringify(bandit.score)
      );

      if (STATS.total > 0) {
        const accuracy =
          STATS.win / STATS.total * 100;

        console.log(
          `[STATS] ${STATS.win}/${STATS.total} | ` +
          `${accuracy.toFixed(2)}%`
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
// 17. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace('Z', '+07:00');
}

// =====================================================
// 18. API RESPONSE
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

  const seq = history.map(h => h.result);

  const result = calculateFinal(
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

    dudoan: result.final === 'TÀI'
      ? 'tai'
      : 'xiu',

    dotincay: `${Math.round(result.confidence)}%`,

    // Các trường bổ sung, không ảnh hưởng trường cũ
    xacsuatTai: Number(
      (result.probability * 100).toFixed(2)
    ),

    phanTichCau: {
      loai: result.reversal.signal,
      doDaiCau: result.reversal.streak,
      soMauLichSu: result.reversal.support,

      xacSuatBeCau: Number(
        (result.reversal.reversalProbability * 100).toFixed(2)
      ),

      xacSuatTiepTuc: Number(
        (result.reversal.continuationProbability * 100).toFixed(2)
      )
    },

    canhbao: result.caution,

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

// Thống kê mô hình
app.get('/stats', (req, res) => {
  const accuracy = STATS.total
    ? STATS.win / STATS.total * 100
    : 0;

  res.json({
    total: STATS.total,
    win: STATS.win,
    lose: STATS.lose,
    accuracy: Number(accuracy.toFixed(2)),
    weights: bandit.score,
    recent: COMPARE,
    caution: CAUTION.data
  });
});

// =====================================================
// 20. BOOT SERVER
// =====================================================

initDataFile();

// Restore historical data
{
  const history = readHistory();

  lastId = history.length
    ? Math.max(...history.map(s => s.id))
    : 0;

  if (history.length > 0) {
    const seq = history.map(s => s.result);

    const current = history[history.length - 1];

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
    `[BOOT] Restored ${history.length} sessions`
  );
}

// Initial sync
syncData();

// Automatic sync
const syncTimer = setInterval(
  syncData,
  POLL_INTERVAL
);

if (typeof syncTimer.unref === 'function') {
  syncTimer.unref();
}

// Start server
app.listen(PORT, () => {
  console.log(`[BOOT] Port ${PORT}`);
  console.log(`[BOOT] Time ${formatTimeVN()}`);

  console.log(
    '[BOOT] Bayesian + Markov + Pattern + Reversal + Bandit'
  );

  console.log('[BOOT] MD5 removed');
  console.log('[BOOT] Random disabled');
  console.log('[BOOT] Reversal learning enabled');
  console.log('[BOOT] Persistent model weights enabled');
});
