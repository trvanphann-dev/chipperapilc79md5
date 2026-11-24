
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

// Chỉ file này được dùng để phân tích và học.
const LEARNING_FILE = path.join(__dirname, 'learning_data.json');

const MODEL_FILE = path.join(__dirname, 'algorithm_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const POLL_INTERVAL = 15000;
const MAX_HISTORY = 500;
const MAX_LEARNING = 500;
const ANALYSIS_WINDOW = 300;
const PORT = process.env.PORT || 3000;

// =====================================================
// 2. GLOBAL STATE
// =====================================================

let lastPredict = null;
let lastId = 0;
let syncInProgress = false;
let bootstrapped = false;

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
  }

  if (!fs.existsSync(LEARNING_FILE)) {
    fs.writeFileSync(LEARNING_FILE, '[]', 'utf8');
  }

  console.log('[INIT] History and learning files ready');
}

function readJsonArray(file) {
  try {
    const parsed = JSON.parse(
      fs.readFileSync(file, 'utf8') || '[]'
    );

    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error('[READ ERROR]', path.basename(file), err.message);
    return [];
  }
}

function readHistory() {
  return readJsonArray(DATA_FILE);
}

function readLearning() {
  return readJsonArray(LEARNING_FILE);
}

function writeHistory(data) {
  atomicWrite(DATA_FILE, data);
}

function writeLearning(data) {
  atomicWrite(LEARNING_FILE, data);
}

// =====================================================
// 4. RESET 10 CAUTION SESSIONS
// =====================================================

function resetHistory(throughId = lastId) {
  const safeId = Number(throughId) || 0;

  writeHistory([]);
  writeLearning([]);

  lastPredict = null;
  lastId = safeId;

  CAUTION.data = [];

  STATS.total = 0;
  STATS.win = 0;
  STATS.lose = 0;

  COMPARE.length = 0;

  bandit.reset();

  console.log('[RESET] 10 caution sessions completed');
  console.log('[RESET] Old history cleared');
  console.log('[RESET] New learning cycle started');
}

// =====================================================
// 5. SOURCE API
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
// 6. NORMALIZE SESSION
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

  const validDices =
    dices.length === 3 &&
    dices.every(d => Number.isInteger(d) && d >= 1 && d <= 6);

  const sum = validDices
    ? dices.reduce((a, b) => a + b, 0)
    : Number(raw.sum ?? raw.total ?? 0);

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
    if (!Number.isFinite(sum) || sum < 3 || sum > 18) {
      return null;
    }

    result = sum >= 11 ? 1 : 0;
  }

  if (!Number.isFinite(Number(id))) {
    return null;
  }

  if (!Number.isFinite(sum) || sum < 3 || sum > 18) {
    return null;
  }

  return {
    id: Number(id),
    _id: String(raw._id ?? id),
    dices: validDices ? dices : [],
    sum,
    result
  };
}

// =====================================================
// 7. COMMON FUNCTIONS
// =====================================================

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getRecent(seq, size = ANALYSIS_WINDOW) {
  return seq.slice(-size);
}

function getBaseProbability(seq) {
  if (!seq.length) return 0.5;

  const recent = getRecent(seq, 100);

  const ones = recent.reduce((a, b) => a + b, 0);

  return (ones + 1) / (recent.length + 2);
}

// =====================================================
// 8. BAYESIAN ADAPTIVE MODEL
// =====================================================

class BayesianModel {
  predict(seq) {
    if (seq.length < 5) return 0.5;

    const recent = getRecent(seq, 100);

    let weightedOnes = 0;
    let totalWeight = 0;

    for (let i = 0; i < recent.length; i++) {
      const age = recent.length - 1 - i;

      const weight = Math.exp(-age / 50);

      weightedOnes += recent[i] * weight;
      totalWeight += weight;
    }

    // Prior cân bằng, tránh kết luận quá sớm.
    const probability =
      (weightedOnes + 2) /
      (totalWeight + 4);

    return clamp(probability, 0.05, 0.95);
  }
}

// =====================================================
// 9. MARKOV MODEL
// =====================================================

class HMMModel {
  predict(seq) {
    if (seq.length < 10) return 0.5;

    const recent = getRecent(seq, 250);
    const base = getBaseProbability(recent);

    let numerator = 0;
    let denominator = 0;

    const maxOrder = Math.min(3, recent.length - 1);

    for (let order = 1; order <= maxOrder; order++) {
      const context = recent.slice(-order);

      let weightedOnes = 0;
      let totalWeight = 0;
      let count = 0;

      for (let i = order; i < recent.length; i++) {
        let matched = true;

        for (let j = 0; j < order; j++) {
          if (recent[i - order + j] !== context[j]) {
            matched = false;
            break;
          }
        }

        if (!matched) continue;

        const age = recent.length - 1 - i;
        const weight = Math.exp(-age / 100);

        weightedOnes += recent[i] * weight;
        totalWeight += weight;
        count++;
      }

      if (!count) continue;

      const probability =
        (weightedOnes + base * 5) /
        (totalWeight + 5);

      const reliability =
        totalWeight / (totalWeight + 5);

      numerator += probability * reliability;
      denominator += reliability;
    }

    return denominator
      ? clamp(numerator / denominator, 0.05, 0.95)
      : 0.5;
  }
}

// =====================================================
// 10. PATTERN RECOGNITION
// =====================================================

function patternAnalysis(seq) {
  if (seq.length < 15) {
    return {
      probability: 0.5,
      support: 0
    };
  }

  const recent = getRecent(seq, 250);
  const base = getBaseProbability(recent);

  let numerator = 0;
  let denominator = 0;
  let support = 0;

  const maxLength = Math.min(5, recent.length - 1);

  for (let length = 2; length <= maxLength; length++) {
    const pattern = recent.slice(-length);

    let weightedOnes = 0;
    let totalWeight = 0;
    let count = 0;

    for (let end = length - 1; end < recent.length - 1; end++) {
      let matched = true;

      for (let j = 0; j < length; j++) {
        if (recent[end - length + 1 + j] !== pattern[j]) {
          matched = false;
          break;
        }
      }

      if (!matched) continue;

      const age = recent.length - 1 - end;
      const weight = Math.exp(-age / 100);

      weightedOnes += recent[end + 1] * weight;
      totalWeight += weight;
      count++;
    }

    if (!count) continue;

    const probability =
      (weightedOnes + base * 5) /
      (totalWeight + 5);

    const reliability =
      totalWeight / (totalWeight + 5);

    numerator += probability * reliability;
    denominator += reliability;
    support += count;
  }

  return {
    probability: denominator
      ? clamp(numerator / denominator, 0.05, 0.95)
      : 0.5,

    support
  };
}

// =====================================================
// 11. STREAK DETECTOR
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
// 12. COMPLETE RUN DETECTION
// =====================================================

function getRuns(seq) {
  if (!seq.length) return [];

  const runs = [];

  let side = seq[0];
  let length = 1;
  let start = 0;

  for (let i = 1; i < seq.length; i++) {
    if (seq[i] === side) {
      length++;
    } else {
      runs.push({
        side,
        length,
        start,
        end: i - 1,
        next: seq[i]
      });

      side = seq[i];
      length = 1;
      start = i;
    }
  }

  runs.push({
    side,
    length,
    start,
    end: seq.length - 1,
    next: null
  });

  return runs;
}

// =====================================================
// 13. REVERSAL ENGINE
// =====================================================

class ReversalEngine {
  analyze(seq) {
    if (seq.length < 20) {
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

    const recent = getRecent(seq, 300);
    const runs = getRuns(recent);

    let reverseWeight = 0;
    let continueWeight = 0;
    let totalWeight = 0;
    let support = 0;

    // Chỉ lấy các chuỗi đã hoàn tất.
    // Không tính từng vị trí trong cùng một chuỗi.

    for (let i = 0; i < runs.length - 1; i++) {
      const run = runs[i];

      if (run.side !== side) continue;

      if (streakBucket(run.length) !== bucket) {
        continue;
      }

      if (run.next === null) continue;

      const age = recent.length - 1 - run.end;
      const weight = Math.exp(-age / 150);

      totalWeight += weight;
      support++;

      if (run.next !== side) {
        reverseWeight += weight;
      } else {
        continueWeight += weight;
      }
    }

    // Prior cân bằng: 3 đảo, 3 tiếp tục.
    const reversalProbability =
      (reverseWeight + 3) / (totalWeight + 6);

    const continuationProbability =
      (continueWeight + 3) / (totalWeight + 6);

    // Xác suất Tài của phiên kế tiếp.
    const probability = side === 1
      ? continuationProbability
      : reversalProbability;

    let signal = 'NEUTRAL';

    if (support >= 8) {
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
      support,
      streak,
      side,
      bucket,
      signal
    };
  }
}

// =====================================================
// 14. ENTROPY
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

// =====================================================
// 15. ADAPTIVE BANDIT
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

    const error = Math.pow(predicted - actual, 2);

    // Cập nhật theo Brier Score.
    const updated =
      old * Math.exp(0.12 * (0.25 - error));

    this.score[name] = clamp(updated, 0.4, 2);
  }

  reset() {
    this.score = {
      bayes: 1,
      markov: 1,
      pattern: 1,
      reversal: 1
    };

    this.save();
  }
}

// =====================================================
// 16. INITIALIZE MODELS
// =====================================================

const bayes = new BayesianModel();
const hmm = new HMMModel();
const reversalEngine = new ReversalEngine();
const bandit = new Bandit();

// =====================================================
// 17. ENSEMBLE PREDICTION
// =====================================================

function calculateFinal(seq) {
  const n = seq.length;

  const insufficient = n < 10;

  if (insufficient) {
    return {
      final: 'CHƯA ĐỦ DỮ LIỆU',
      confidence: 0,
      caution: true,
      probability: 0.5,
      components: {
        bayes: 0.5,
        markov: 0.5,
        pattern: 0.5,
        reversal: 0.5
      },
      agreement: 0,
      reversal: reversalEngine.analyze(seq),
      pattern: {
        probability: 0.5,
        support: 0
      },
      entropy: 0
    };
  }

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

  // Đo mức độ đồng thuận.
  const agreement = Object.values(components)
    .filter(p =>
      (probability >= 0.5 && p >= 0.5) ||
      (probability < 0.5 && p < 0.5)
    ).length / 4;

  const strength = Math.abs(probability - 0.5);

  const dataFactor = Math.min(1, n / 100);

  const supportFactor = Math.min(
    1,
    reversal.support / 15
  );

  // Confidence là chỉ số tham khảo, không phải xác suất thắng.
  const confidence = Math.round(
    clamp(
      50 +
      strength * 100 *
      agreement *
      dataFactor *
      (0.7 + 0.3 * supportFactor),
      0,
      75
    ) * 100
  ) / 100;

  const caution =
    n < 30 ||
    agreement < 0.5 ||
    strength < 0.06 ||
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
// 18. PROCESS NEW SESSION
// =====================================================

function processLatest(session) {
  // Chỉ đọc dữ liệu học mới.
  const learning = readLearning();

  const seq = learning.map(h => h.result);

  // Chấm dự đoán trước đó.
  if (lastPredict) {
    const actual = session.result;

    const real = actual === 1 ? 'TÀI' : 'XỈU';

    const win = lastPredict.predict === real;

    STATS.total++;

    if (win) {
      STATS.win++;
    } else {
      STATS.lose++;
    }

    // Cập nhật từng mô hình bằng dự đoán đã lưu trước đó.
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

      bandit.save();
    }

    COMPARE.push({
      phien: session.id,
      predict: lastPredict.predict,
      real,
      win
    });

    if (COMPARE.length > 20) {
      COMPARE.shift();
    }

    console.log(
      `[RESULT] ${session.id} | ` +
      `${lastPredict.predict} | ${real} | ` +
      `${win ? 'WIN' : 'LOSE'}`
    );

    // Chỉ tính kết quả của những dự đoán cảnh báo.
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);

      console.log(
        `[CAUTION] ${CAUTION.data.length}/10 | ` +
        `Win: ${CAUTION.data.filter(x => x === 1).length} | ` +
        `Lose: ${CAUTION.data.filter(x => x === 0).length}`
      );

      // Đủ 10 dự đoán cảnh báo thì kiểm tra reset.
      if (CAUTION.data.length >= 10) {
        const wins = CAUTION.data.filter(x => x === 1).length;
        const loses = CAUTION.data.filter(x => x === 0).length;

        if (loses >= wins) {
          resetHistory(session.id);
          return null;
        }

        // Nếu thắng nhiều hơn thua, bắt đầu chu kỳ 10 mới.
        CAUTION.data = [];

        console.log('[CAUTION] 10-session check passed');
      }
    }
  }

  // Sau khi chấm dự đoán cũ, thêm phiên mới vào dữ liệu học.
  // Dữ liệu mới được lưu trước khi tính dự đoán tiếp theo.
  const currentLearning = readLearning();

  if (!currentLearning.some(h => h.id === session.id)) {
    currentLearning.push(session);

    currentLearning.sort((a, b) => a.id - b.id);

    const trimmed = currentLearning.slice(-MAX_LEARNING);

    writeLearning(trimmed);
  }

  const newSeq = readLearning().map(h => h.result);

  const result = calculateFinal(newSeq);

  if (result.final === 'CHƯA ĐỦ DỮ LIỆU') {
    lastPredict = null;

    console.log(
      `[LEARNING] ${session.id} | ` +
      `${newSeq.length} new sessions | Not enough data`
    );

    return result;
  }

  lastPredict = {
    predict: result.final,
    caution: result.caution,
    components: { ...result.components }
  };

  console.log(
    `[PREDICT] ${session.id + 1} | ${result.final} | ` +
    `Probability TAI: ${(result.probability * 100).toFixed(2)}%`
  );

  console.log(
    `[REVERSAL] Streak: ${result.reversal.streak} | ` +
    `Support: ${result.reversal.support} | ` +
    `Reverse: ${(result.reversal.reversalProbability * 100).toFixed(1)}% | ` +
    `Signal: ${result.reversal.signal}`
  );

  console.log(
    '[WEIGHTS]',
    JSON.stringify(bandit.score)
  );

  return result;
}

// =====================================================
// 19. DATA SYNCHRONIZATION
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

    const normalized = list
      .map(normalizeSession)
      .filter(s => s !== null)
      .sort((a, b) => a.id - b.id);

    if (!normalized.length) return;

    let history = readHistory();
    let learning = readLearning();

    // Lần đầu cài đặt:
    // Nếu chưa có lịch sử, chỉ dùng dữ liệu nguồn để
    // tạo mốc phiên hiện tại và không đưa dữ liệu cũ
    // vào learning_data.json.
    if (!bootstrapped) {
      if (history.length === 0 && learning.length === 0) {
        history = normalized.slice(-MAX_HISTORY);

        writeHistory(history);

        lastId = Math.max(...normalized.map(s => s.id));

        bootstrapped = true;

        console.log(
          `[BOOTSTRAP] Saved display history only. ` +
          `Learning starts after session ${lastId}`
        );

        return;
      }

      const historyMax = history.length
        ? Math.max(...history.map(s => Number(s.id) || 0))
        : 0;

      const learningMax = learning.length
        ? Math.max(...learning.map(s => Number(s.id) || 0))
        : 0;

      lastId = Math.max(lastId, historyMax, learningMax);

      bootstrapped = true;
    }

    const fresh = normalized.filter(s => s.id > lastId);

    if (!fresh.length) return;

    for (const session of fresh) {
      // Cập nhật mốc ID ngay, kể cả khi reset xảy ra.
      if (session.id <= lastId) continue;

      lastId = session.id;

      // Lưu lịch sử hiển thị.
      history = readHistory();

      if (!history.some(h => h.id === session.id)) {
        history.push(session);

        history.sort((a, b) => a.id - b.id);

        history = history.slice(-MAX_HISTORY);

        writeHistory(history);
      }

      // Chỉ phiên mới được đưa vào learning_data.
      const result = processLatest(session);

      // Nếu reset được kích hoạt, processLatest trả null.
      // Không xử lý lại dữ liệu cũ.
      if (result === null) {
        history = readHistory();
      }

      console.log(
        `[SYNC] Session ${session.id} processed`
      );
    }

    const currentLearning = readLearning();

    console.log(
      `[SYNC] History: ${readHistory().length} | ` +
      `New learning data: ${currentLearning.length}`
    );

    if (STATS.total > 0) {
      const accuracy =
        STATS.win / STATS.total * 100;

      console.log(
        `[STATS] ${STATS.win}/${STATS.total} | ` +
        `${accuracy.toFixed(2)}%`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 20. VIETNAM TIME
// =====================================================

function formatTimeVN() {
  const now = new Date(
    Date.now() + 7 * 3600 * 1000
  );

  return now.toISOString().replace('Z', '+07:00');
}

// =====================================================
// 21. API RESPONSE
// =====================================================

function buildResponse() {
  const history = readHistory();
  const learning = readLearning();

  if (!history.length) {
    return {
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime: formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  // Tuyệt đối không dùng history để tính dự đoán.
  const seq = learning.map(h => h.result);

  const result = calculateFinal(seq);

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
      : result.final === 'XỈU'
        ? 'xiu'
        : 'chua_du_du_lieu',

    dotincay: `${Math.round(result.confidence)}%`,

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

    soPhienHocMoi: seq.length,

    canhbao: result.caution,

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

// Chỉ hiển thị dữ liệu mới mà thuật toán đã học.
app.get('/learning', (req, res) => {
  try {
    const learning = readLearning();

    res.json({
      count: learning.length,
      last20: learning.slice(-20)
    });
  } catch (err) {
    res.status(500).json({
      error: 'Learning data error'
    });
  }
});

// Thống kê
app.get('/stats', (req, res) => {
  const accuracy = STATS.total
    ? STATS.win / STATS.total * 100
    : 0;

  res.json({
    total: STATS.total,
    win: STATS.win,
    lose: STATS.lose,
    accuracy: Number(accuracy.toFixed(2)),

    learningSessions: readLearning().length,

    weights: bandit.score,

    recent: COMPARE,

    caution: CAUTION.data
  });
});

// =====================================================
// 23. BOOT SERVER
// =====================================================

initDataFile();

{
  const history = readHistory();
  const learning = readLearning();

  const historyMax = history.length
    ? Math.max(...history.map(s => Number(s.id) || 0))
    : 0;

  const learningMax = learning.length
    ? Math.max(...learning.map(s => Number(s.id) || 0))
    : 0;

  lastId = Math.max(historyMax, learningMax);

  // Khôi phục dự đoán chỉ từ dữ liệu học mới.
  if (learning.length >= 10) {
    const seq = learning.map(h => h.result);

    const result = calculateFinal(seq);

    if (result.final !== 'CHƯA ĐỦ DỮ LIỆU') {
      lastPredict = {
        predict: result.final,
        caution: result.caution,
        components: { ...result.components }
      };
    }
  }

  bootstrapped = history.length > 0 || learning.length > 0;

  console.log(
    `[BOOT] History: ${history.length}`
  );

  console.log(
    `[BOOT] New learning sessions: ${learning.length}`
  );

  console.log(
    `[BOOT] Last session ID: ${lastId}`
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

  console.log('[BOOT] Bayesian + Markov + Pattern + Reversal + Bandit');

  console.log('[BOOT] Old history excluded from prediction');
  console.log('[BOOT] Learning uses learning_data.json only');
  console.log('[BOOT] 10 caution-session reset enabled');
  console.log('[BOOT] MD5 removed from algorithm');
  console.log('[BOOT] Random disabled');
});
