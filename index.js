
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
// 3. ALGORITHM ENGINE (port từ bot.pylc)
// =====================================================

// 1) BAYESIAN ADAPTIVE
class BayesianModel {
  constructor() {
    this.alpha = 1;
    this.beta = 1;
  }

  update(result) {
    if (result === 1) this.alpha += 1;
    else this.beta += 1;
  }

  probability() {
    return this.alpha / (this.alpha + this.beta);
  }
}

// 2) HMM (SIMPLIFIED) - nhận diện cầu
class HMMModel {
  predict(seq) {
    if (seq.length < 8) return 0;

    const last = seq[seq.length - 1];
    let score = 0.0;

    // CẦU BỆT - càng dài càng mạnh
    let streak = 1;
    for (let i = seq.length - 2; i >= 0; i--) {
      if (seq[i] === last) streak++;
      else break;
    }

    if (streak >= 4) {
      score += Math.min(1.2, streak * 0.25);
    }

    // CẦU ĐẢO ĐỀU
    const last6 = seq.slice(-6).join('');
    if (last6 === '101010' || last6 === '010101') {
      score -= 1.0;
    }

    // CẦU GÃY
    const recent = seq.slice(-7, -1);
    if (recent.filter(v => v === last).length <= 2) {
      score -= 0.8;
    }

    // NHIỄU CAO -> giảm tác động
    if (new Set(seq.slice(-6)).size === 2 && streak <= 2) {
      score *= 0.5;
    }

    if (score > 0.4) return 1;
    if (score < -0.4) return -1;
    return 0;
  }
}

// 3) BANDIT - ONLINE LEARNING
class Bandit {
  constructor() {
    this.score = {};
  }

  weight(name) {
    const s = this.score[name] ?? 0;
    return Math.max(0.3, Math.min(1.5, s));
  }

  reward(name, win) {
    const s = (this.score[name] ?? 0) * 0.95;
    this.score[name] = win ? s + 0.15 : s - 0.2;
  }
}

// 4) ENTROPY DETECTOR
function entropy(seq, w = 20) {
  if (seq.length < w) return 0;

  const win = seq.slice(-w);
  const p1 = win.reduce((a, b) => a + b, 0) / w;
  const p0 = 1 - p1;

  if (p1 === 0 || p1 === 1) return 0;

  return -(p1 * Math.log2(p1) + p0 * Math.log2(p0));
}

function chaosBlock(seq) {
  if (seq.length < 25) return false;
  return entropy(seq, 30) > 0.92;
}

// 5) MD5 SIGNAL
function algoMd5(id, seqLen) {
  const md5 = crypto
    .createHash('md5')
    .update(String(id))
    .digest('hex');

  let sig = parseInt(md5[md5.length - 1], 16) < 8 ? 1 : -1;

  // chuỗi càng dài -> md5 càng yếu
  if (seqLen > 40) sig *= 0.4;

  return sig;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// INIT MODELS
const bayes = new BayesianModel();
const hmm = new HMMModel();
const bandit = new Bandit();

// FINAL DECISION
function calculateFinal(seq, id) {
  const n = seq.length;
  const caution = chaosBlock(seq);

  const pBayes = bayes.probability();
  const hmmSig = hmm.predict(seq);
  const md5Sig = algoMd5(id, n);

  const wBayes = bandit.weight('bayes');
  const wHmm = bandit.weight('hmm');
  const wMd5 = bandit.weight('md5');

  let score;

  if (n < 25) {
    // Short-term ưu tiên pattern
    score =
      (pBayes - 0.5) * 4 +
      hmmSig * 3 +
      md5Sig * 2;
  } else {
    // Long-term ưu tiên thống kê
    score =
      (pBayes - 0.5) * 6 * wBayes +
      hmmSig * 2 * wHmm +
      md5Sig * 1 * wMd5;
  }

  const prob = sigmoid(score);
  const final = prob >= 0.5 ? 'TÀI' : 'XỈU';

  const strength = Math.abs(score);
  const confidence =
    Math.round(Math.min(95, 50 + strength * 18) * 100) / 100;

  return { final, confidence, caution };
}

// =====================================================
// 4. STATE: DỰ ĐOÁN, THỐNG KÊ, CAUTION, RESET
// =====================================================

let lastPredict = null;           // { predict, caution }
let lastId = 0;                   // phiên mới nhất đã xử lý
const STATS = { total: 0, win: 0, lose: 0 };
const CAUTION = { data: [] };     // 10 lần "Lưu Ý" gần nhất
const COMPARE = [];               // log 10 phiên gần nhất

// Xử lý 1 phiên mới: dự đoán phiên kế tiếp + chấm điểm dự đoán trước.
function processLatest(session) {
  const seq = readHistory().map(h => h.result);
  const { final, confidence, caution } =
    calculateFinal(seq, session._id ?? session.id);

  if (lastPredict) {
    STATS.total++;

    const real = session.result === 1 ? 'TÀI' : 'XỈU';
    const win = lastPredict.predict === real;

    bandit.reward('bayes', win);
    bandit.reward('hmm', win);
    bandit.reward('md5', win);

    if (win) STATS.win++;
    else STATS.lose++;

    COMPARE.push({
      phien: session.id,
      predict: lastPredict.predict,
      real,
      win
    });
    if (COMPARE.length > 10) COMPARE.shift();

    // Reset ẩn theo 10 lần "Lưu Ý"
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);
      if (CAUTION.data.length > 10) CAUTION.data.shift();

      if (CAUTION.data.length === 10) {
        const winCount = CAUTION.data.reduce((a, b) => a + b, 0);
        const loseCount = 10 - winCount;

        // Reset nếu sai >= đúng: xoá + tạo lại file lịch sử
        if (loseCount >= winCount) {
          resetHistory();
        }
      }
    }
  }

  lastPredict = { predict: final, caution };

  return { final, confidence, caution };
}

// =====================================================
// 5. SYNC
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
    const knownIds = new Set(history.map(h => h.id));
    const added = [];

    for (const s of fresh) {
      if (knownIds.has(s.id)) continue;
      history.push(s);
      knownIds.add(s.id);
      bayes.update(s.result);
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
  const now = new Date(Date.now() + 7 * 3600 * 1000);
  return now.toISOString().replace('Z', '+07:00');
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

  const { final, confidence } =
    calculateFinal(seq, current._id ?? current.id);

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,
    dudoanphien: current.id + 1,
    dudoan: final === 'TÀI' ? 'tai' : 'xiu',
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
  h.forEach(s => bayes.update(s.result));
  lastId = h.length ? h[h.length - 1].id : 0;
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

  console.log('[BOOT] Bayes + HMM + Bandit + MD5 engine loaded');
});

