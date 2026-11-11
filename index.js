
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
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 3. USER ALGORITHM ENGINE
// =====================================================

// 1. BAYESIAN ADAPTIVE

class BayesianModel {
  constructor() {
    this.alpha = 1;
    this.beta = 1;
  }

  update(result) {
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

// 2. HIDDEN MARKOV (SIMPLIFIED)

class HMMModel {
  predict(seq) {
    if (seq.length < 8) return 0;

    const last = seq[seq.length - 1];
    let score = 0;

    // CẦU BỆT – càng dài càng mạnh

    let streak = 1;

    for (let i = seq.length - 2; i >= 0; i--) {
      if (seq[i] === last) {
        streak++;
      } else {
        break;
      }
    }

    if (streak >= 4) {
      score += Math.min(1.2, streak * 0.25);
    }

    // CẦU ĐẢO ĐỀU

    const lastSix = seq.slice(-6);

    if (
      lastSix.join('') === '101010' ||
      lastSix.join('') === '010101'
    ) {
      score -= 1.0;
    }

    // CẦU GÃY (TXTXXT)

    const recent = seq.slice(-7, -1);

    if (recent.filter(value => value === last).length <= 2) {
      score -= 0.8;
    }

    // NHIỄU CAO → giảm tác động

    if (new Set(lastSix).size === 2 && streak <= 2) {
      score *= 0.5;
    }

    // CHUẨN HÓA

    if (score > 0.4) return 1;
    if (score < -0.4) return -1;

    return 0;
  }
}

// 3. BANDIT – ONLINE LEARNING

class Bandit {
  constructor() {
    this.score = Object.create(null);
  }

  weight(name) {
    const score = this.score[name] || 0;

    return Math.max(
      0.3,
      Math.min(1.5, score)
    );
  }

  reward(name, win) {
    this.score[name] = (this.score[name] || 0) * 0.95;

    if (win) {
      this.score[name] += 0.15;
    } else {
      this.score[name] -= 0.2;
    }
  }
}

// 4. ENTROPY DETECTOR

function entropy(seq, w = 20) {
  if (seq.length < w) return 0;

  const win = seq.slice(-w);

  const p1 = win.reduce(
    (sum, value) => sum + value,
    0
  ) / w;

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
  if (seq.length < 25) return false;

  return entropy(seq, 30) > 0.92;
}

// 5. MD5 SIGNAL

function algoMD5(id, seqLen) {
  const hash = crypto
    .createHash('md5')
    .update(String(id))
    .digest('hex');

  let signal =
    parseInt(hash[hash.length - 1], 16) < 8
      ? 1
      : -1;

  // Chuỗi càng dài → MD5 càng yếu

  if (seqLen > 40) {
    signal *= 0.4;
  }

  return signal;
}

// =====================================================
// INIT MODELS
// =====================================================

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// =====================================================
// FINAL DECISION
// =====================================================

function calculateFinal(seq, id) {
  const n = seq.length;

  const caution = chaosBlock(seq);

  // BAYESIAN

  const bayes = new BayesianModel();

  for (const result of seq) {
    bayes.update(result);
  }

  // HMM

  const hmm = new HMMModel();

  // BANDIT

  const bandit = new Bandit();

  // MD5

  const pBayes = bayes.probability();

  const hmmSignal = hmm.predict(seq);

  const md5Signal = algoMD5(id, n);

  const wBayes = bandit.weight('bayes');
  const wHmm = bandit.weight('hmm');
  const wMd5 = bandit.weight('md5');

  // SHORT-TERM / LONG-TERM

  let score;

  if (n < 25) {
    score =
      (pBayes - 0.5) * 4 +
      hmmSignal * 3 +
      md5Signal * 2;

  } else {
    score =
      (pBayes - 0.5) * 6 * wBayes +
      hmmSignal * 2 * wHmm +
      md5Signal * 1 * wMd5;
  }

  // FINAL PROBABILITY

  const probabilityTai = sigmoid(score);

  const final =
    probabilityTai >= 0.5
      ? 'TÀI'
      : 'XỈU';

  const strength = Math.abs(score);

  const confidence = Math.round(
    Math.min(95, 50 + strength * 18) * 100
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

      probabilityTai:
        result.hmmSignal > 0
          ? 0.75
          : result.hmmSignal < 0
            ? 0.25
            : 0.5,

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

    historicalAccuracy: 0.5,

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
    '[BOOT] Bayesian + HMM + Bandit + Entropy + MD5 engine loaded'
  );

  console.log(
    '[BOOT] User algorithm engine enabled'
  );
});
