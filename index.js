const express = require('express');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

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

      // Học ngay từ các phiên mới
      learnNewSessions(saved);
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 3. USER ALGORITHM ENGINE (UPGRADED)
// =====================================================

// 1. BAYESIAN ADAPTIVE (có quên dần để theo kịp khi game đổi)

class BayesianModel {
  constructor(decay = 0.98) {
    this.alpha = 1;
    this.beta = 1;
    this.decay = decay;
  }

  update(result) {
    this.alpha *= this.decay;
    this.beta *= this.decay;

    if (result === 1) {
      this.alpha += 1;
    } else {
      this.beta += 1;
    }
  }

  probability() {
    return this.alpha / (this.alpha + this.beta);
  }
}

// 2. MARKOV – TỰ HỌC CẦU TỪ DỮ LIỆU (thay luật cầu cố định)

class HMMModel {
  constructor(orders = [2, 3, 4, 5], decay = 0.995, minN = 6) {
    this.orders = orders;
    this.decay = decay;
    this.minN = minN;
    this.counts = {};

    for (const k of orders) {
      this.counts[k] = new Map();
    }
  }

  // Trả về tín hiệu trong [-1, 1]: >0 nghiêng TÀI, <0 nghiêng XỈU
  predict(seq) {
    if (seq.length < 8) return 0;

    let num = 0;
    let den = 0;

    for (const k of this.orders) {
      if (seq.length < k) continue;

      const cell = this.counts[k].get(seq.slice(-k).join(''));

      if (!cell) continue;

      const n = cell[0] + cell[1];

      if (n < this.minN) continue;

      const p = (cell[1] + 1) / (n + 2); // Laplace

      num += (p - 0.5) * 2 * n; // ngữ cảnh nhiều mẫu → tin hơn
      den += n;
    }

    return den > 0 ? num / den : 0;
  }

  update(seq, result) {
    for (const k of this.orders) {
      if (seq.length < k) continue;

      const key = seq.slice(-k).join('');

      let cell = this.counts[k].get(key);

      if (!cell) {
        cell = [0, 0];
        this.counts[k].set(key, cell);
      }

      cell[0] *= this.decay; // quên dần cầu cũ
      cell[1] *= this.decay;
      cell[result] += 1;
    }
  }
}

// 3. BANDIT – ONLINE LEARNING

class Bandit {
  constructor() {
    this.score = Object.create(null);
  }

  get(name) {
    return this.score[name] === undefined ? 1 : this.score[name];
  }

  weight(name) {
    return Math.max(0.3, Math.min(1.5, this.get(name)));
  }

  reward(name, win) {
    // kéo dần về 1.0 (quên ký ức cũ)
    let s = 1 + (this.get(name) - 1) * 0.95;

    // thưởng/phạt đối xứng: đoán ngẫu nhiên thì trọng số đứng yên
    s += win ? 0.05 : -0.05;

    this.score[name] = s;
  }
}

// 4. ENTROPY DETECTOR

function entropy(seq, w = 20) {
  if (seq.length < w) return 0;

  const win = seq.slice(-w);

  const p1 = win.reduce((sum, value) => sum + value, 0) / w;
  const p0 = 1 - p1;

  if (p1 === 0 || p1 === 1) {
    return 0;
  }

  return -(p1 * Math.log2(p1) + p0 * Math.log2(p0));
}

function chaosBlock(seq) {
  if (seq.length < 25) return false;

  return entropy(seq, 30) > 0.97;
}

// 5. MD5 SIGNAL (bandit sẽ tự hạ trọng số nếu nó vô dụng)

function algoMD5(id, seqLen) {
  const hash = crypto
    .createHash('md5')
    .update(String(id))
    .digest('hex');

  let signal =
    parseInt(hash[hash.length - 1], 16) < 8
      ? 1
      : -1;

  if (seqLen > 40) {
    signal *= 0.4;
  }

  return signal;
}

// =====================================================
// INIT MODELS (giữ trạng thái giữa các lần dự đoán)
// =====================================================

const bayes = new BayesianModel();
const hmm = new HMMModel();
const bandit = new Bandit();

// Theo dõi độ chính xác thật (walk-forward)
const tracker = {
  total: 0,
  hits: 0,
  recent: []
};

const TRACKER_WINDOW = 300;
const TRACKER_MIN_SEQ = 10;

let learnedUpTo = -Infinity;

function sigmoid(x) {
  const c = Math.max(-30, Math.min(30, x));
  return 1 / (1 + Math.exp(-c));
}

// =====================================================
// FINAL DECISION (thuần dự đoán, không thay đổi mô hình)
// =====================================================

function calculateFinal(seq, id) {
  const n = seq.length;

  const caution = chaosBlock(seq);

  const pBayes = bayes.probability();

  const hmmSignal = hmm.predict(seq);

  const md5Signal = algoMD5(id, n);

  const wBayes = bandit.weight('bayes');
  const wHmm = bandit.weight('hmm');
  const wMd5 = bandit.weight('md5');

  let score;

  if (n < 25) {
    // Short-term ưu tiên pattern
    score =
      (pBayes - 0.5) * 4 * wBayes +
      hmmSignal * 4 * wHmm +
      md5Signal * 1 * wMd5;
  } else {
    // Long-term ưu tiên thống kê
    score =
      (pBayes - 0.5) * 6 * wBayes +
      hmmSignal * 3 * wHmm +
      md5Signal * 0.5 * wMd5;
  }

  const probabilityTai = sigmoid(score);

  const final =
    probabilityTai >= 0.5
      ? 'TÀI'
      : 'XỈU';

  // confidence = xác suất mô hình tự tính
  const confidence = Math.round(
    Math.max(probabilityTai, 1 - probabilityTai) * 10000
  ) / 100;

  return {
    final,
    confidence,
    caution,
    score,
    probabilityTai,
    bayesProbability: pBayes,
    hmmSignal,
    md5Signal,
    weights: {
      bayes: wBayes,
      hmm: wHmm,
      md5: wMd5
    }
  };
}

// =====================================================
// HỌC SAU MỖI PHIÊN (walk-forward: dự đoán trước, học sau)
// =====================================================

function learnNewSessions(history) {
  if (!Array.isArray(history) || history.length === 0) return;

  const results = history.map(h => h.result);

  for (let i = 0; i < history.length; i++) {
    const session = history[i];

    if (session.id <= learnedUpTo) continue;

    const actual = session.result;

    if (i === 0) {
      bayes.update(actual);
      learnedUpTo = session.id;
      continue;
    }

    const seqBefore = results.slice(0, i);
    const prevId = history[i - 1].id;

    // Dự đoán như lúc chưa biết kết quả
    const r = calculateFinal(seqBefore, prevId);

    if (seqBefore.length >= TRACKER_MIN_SEQ) {
      const hit = (r.final === 'TÀI' ? 1 : 0) === actual ? 1 : 0;

      tracker.total++;
      tracker.hits += hit;
      tracker.recent.push(hit);

      if (tracker.recent.length > TRACKER_WINDOW) {
        tracker.recent.shift();
      }
    }

    // Thưởng / phạt từng thuật toán
    const signals = [
      ['bayes', r.bayesProbability - 0.5],
      ['hmm', r.hmmSignal],
      ['md5', r.md5Signal]
    ];

    for (const [name, sig] of signals) {
      if (sig === 0) continue;

      bandit.reward(name, (sig > 0) === (actual === 1));
    }

    // Cập nhật mô hình bằng kết quả thật
    bayes.update(actual);
    hmm.update(seqBefore, actual);

    learnedUpTo = session.id;
  }
}

function historicalAccuracyPercent() {
  if (tracker.recent.length === 0) return 0;

  const hits = tracker.recent.reduce((a, b) => a + b, 0);

  return Math.round((hits / tracker.recent.length) * 10000) / 100;
}

// =====================================================
// API COMPATIBILITY
// =====================================================

function ensemblePredict(seq, currentId) {
  const result = calculateFinal(seq, currentId);

  const pick = result.final === 'TÀI' ? 1 : 0;

  const algorithms = [
    {
      name: 'bayesian_adaptive',

      pick: result.bayesProbability >= 0.5 ? 1 : 0,

      probabilityTai: result.bayesProbability,

      weight: result.weights.bayes
    },

    {
      name: 'hmm_simplified',

      pick:
        result.hmmSignal > 0
          ? 1
          : result.hmmSignal < 0
            ? 0
            : pick,

      probabilityTai: 0.5 + result.hmmSignal * 0.5,

      weight: result.weights.hmm
    },

    {
      name: 'md5_signal',

      pick: result.md5Signal > 0 ? 1 : 0,

      probabilityTai:
        result.md5Signal > 0
          ? 0.75
          : 0.25,

      weight: result.weights.md5
    }
  ];

  const agreement = algorithms.filter(
    algo => algo.pick === pick
  ).length;

  return {
    pick,

    confidence: result.confidence,

    pTai: result.probabilityTai,

    agreement: `${agreement}/${algorithms.length}`,

    historicalAccuracy: historicalAccuracyPercent(),

    trainedSamples: seq.length,

    caution: result.caution,

    algorithms
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

  // Đảm bảo mô hình đã học hết lịch sử hiện có (idempotent)
  learnNewSessions(history);

  const current = history[history.length - 1];

  const seq = history.map(
    h => h.result
  );

  const ens = ensemblePredict(
    seq,
    String(current.id)
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

// Học từ lịch sử đã lưu để mô hình sẵn sàng ngay khi khởi động.

learnNewSessions(readHistory());

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
    '[BOOT] Bayesian + Markov + Bandit + Entropy + MD5 engine loaded'
  );

  console.log(
    '[BOOT] Upgraded algorithm engine enabled'
  );
});
