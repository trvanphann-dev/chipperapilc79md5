
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
// 3. ENGINE v2: ĐẶC TRƯNG + HỌC TRỰC TUYẾN (logistic) + BAYES CÓ TRỌNG SỐ THỜI GIAN
// =====================================================

const STATE_FILE = path.join(__dirname, 'state.json');

const DECAY = 0.97;      // phiên cũ giảm dần ảnh hưởng trong Bayes
const LR = 0.05;         // tốc độ học
const L2 = 0.002;        // kéo trọng số về 0 nếu đặc trưng không có ích
const W_CLIP = 2.5;
const FEATURES = ['bias', 'bayes', 'streak', 'alt', 'md5', 'short', 'long'];
const W_INIT = { bias: 0, bayes: 0.5, streak: 0.3, alt: 0.3, md5: 0.2, short: 0, long: 0 };

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0.5;
}

// Entropy cửa sổ gần nhất -> phát hiện chuỗi quá nhiễu
function entropy(seq, w = 20) {
  if (seq.length < w) return 0;
  const win = seq.slice(-w);
  const p1 = mean(win);
  const p0 = 1 - p1;
  if (p1 === 0 || p1 === 1) return 0;
  return -(p1 * Math.log2(p1) + p0 * Math.log2(p0));
}

function chaosBlock(seq) {
  if (seq.length < 25) return false;
  return entropy(seq, 30) > 0.92;
}

function md5Signal(id) {
  const h = crypto.createHash('md5').update(String(id)).digest('hex');
  return parseInt(h[h.length - 1], 16) < 8 ? 1 : -1;
}

class Engine {
  constructor() {
    this.a = 1;                 // Bayes alpha (Tài)
    this.b = 1;                 // Bayes beta (Xỉu)
    this.w = { ...W_INIT };     // trọng số logistic, tự học
  }

  // Gọi khi có kết quả mới
  observe(result) {
    this.a = 1 + (this.a - 1) * DECAY + (result === 1 ? 1 : 0);
    this.b = 1 + (this.b - 1) * DECAY + (result === 0 ? 1 : 0);
  }

  features(seq, id) {
    const n = seq.length;
    const last = n ? seq[n - 1] : 1;
    const dir = last === 1 ? 1 : -1;

    let streak = 0;
    if (n) {
      streak = 1;
      for (let i = n - 2; i >= 0 && seq[i] === last; i--) streak++;
    }

    let alternating = 0;
    if (n >= 6) {
      const t = seq.slice(-6);
      if (t.every((v, i) => i === 0 || v !== t[i - 1])) alternating = 1;
    }

    return {
      bias: 1,
      bayes: (this.a / (this.a + this.b) - 0.5) * 2,
      streak: streak >= 3 ? dir * Math.min(streak, 6) / 6 : 0,
      alt: alternating ? -dir : 0,
      md5: md5Signal(id),
      short: n >= 10 ? (mean(seq.slice(-10)) - 0.5) * 2 : 0,
      long: n >= 50 ? (mean(seq.slice(-50)) - 0.5) * 2 : 0
    };
  }

  dot(x) {
    return FEATURES.reduce((z, k) => z + this.w[k] * x[k], 0);
  }

  predict(seq, id) {
    const x = this.features(seq, id);
    return { x, p: sigmoid(this.dot(x)) };
  }

  // Học từ kết quả thật của phiên đã dự đoán
  learn(x, result) {
    const err = result - sigmoid(this.dot(x));
    for (const k of FEATURES) {
      this.w[k] = clamp(
        this.w[k] + LR * err * x[k] - L2 * this.w[k],
        -W_CLIP,
        W_CLIP
      );
    }
  }
}

let engine = new Engine();

function decide(seq, id) {
  const { x, p } = engine.predict(seq, id);
  return {
    x,
    p,
    final: p >= 0.5 ? 'TÀI' : 'XỈU',
    // Độ tin cậy = xác suất mô hình tự tính (không bơm số)
    confidence: Math.round(Math.max(p, 1 - p) * 100),
    caution: chaosBlock(seq)
  };
}

// =====================================================
// 4. STATE: DỰ ĐOÁN, THỐNG KÊ, CAUTION, LƯU TRẠNG THÁI
// =====================================================

let lastPredict = null;           // { forId, predict, caution, x }
let lastId = 0;
const STATS = { total: 0, win: 0, lose: 0 };
const CAUTION = { data: [] };     // 10 lần "Lưu Ý" gần nhất
let predLog = [];                 // nhật ký dự đoán đã chấm (tối đa 2000)

function hourVN() {
  return new Date(Date.now() + 7 * 3600 * 1000).getUTCHours();
}

function saveState() {
  const tmp = STATE_FILE + '.tmp';
  try {
    fs.writeFileSync(tmp, JSON.stringify({
      engine: { a: engine.a, b: engine.b, w: engine.w },
      lastId,
      stats: STATS,
      caution: CAUTION.data,
      lastPredict,
      predLog
    }), 'utf8');
    fs.renameSync(tmp, STATE_FILE);
  } catch (err) {
    console.error('[STATE SAVE ERROR]', err.message);
  }
}

function loadState() {
  try {
    if (!fs.existsSync(STATE_FILE)) return false;
    const st = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));

    engine.a = st.engine.a;
    engine.b = st.engine.b;
    engine.w = { ...W_INIT, ...st.engine.w };
    lastId = st.lastId || 0;
    Object.assign(STATS, st.stats || {});
    CAUTION.data = Array.isArray(st.caution) ? st.caution : [];
    lastPredict = st.lastPredict || null;
    predLog = Array.isArray(st.predLog) ? st.predLog : [];

    console.log('[INIT] Engine state restored');
    return true;
  } catch (err) {
    console.error('[STATE LOAD ERROR]', err.message);
    return false;
  }
}

// Xử lý 1 phiên mới: chấm dự đoán trước, học, rồi dự đoán phiên kế tiếp.
function processLatest(session) {
  const seq = readHistory().map(h => h.result);

  if (lastPredict && lastPredict.forId === session.id) {
    const real = session.result === 1 ? 'TÀI' : 'XỈU';
    const win = lastPredict.predict === real;

    engine.learn(lastPredict.x, session.result);

    STATS.total++;
    if (win) STATS.win++;
    else STATS.lose++;

    predLog.push({
      phien: session.id,
      predict: lastPredict.predict,
      real,
      win,
      hour: hourVN(),
      ts: Date.now()
    });
    if (predLog.length > 2000) predLog.shift();

    // Reset ẩn theo 10 lần "Lưu Ý"
    if (lastPredict.caution) {
      CAUTION.data.push(win ? 1 : 0);
      if (CAUTION.data.length > 10) CAUTION.data.shift();

      if (CAUTION.data.length === 10) {
        const winCount = CAUTION.data.reduce((a, b) => a + b, 0);
        if (10 - winCount >= winCount) {
          resetHistory();   // xoá + tạo lại file lịch sử
        }
      }
    }
  }

  const d = decide(seq, session._id ?? session.id);
  lastPredict = {
    forId: session.id + 1,
    predict: d.final,
    caution: d.caution,
    x: d.x
  };

  saveState();
  return d;
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
      engine.observe(s.result);
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
  const { final, confidence } = decide(seq, current._id ?? current.id);

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,
    dudoanphien: current.id + 1,
    dudoan: final === 'TÀI' ? 'tai' : 'xiu',
    dotincay: `${confidence}%`,
    uptime: formatTimeVN()
  };
}

// =====================================================
// 8. THỐNG KÊ & BACKTEST (để kiểm chứng độ chính xác thật)
// =====================================================

function rate(win, total) {
  return total ? Math.round((win / total) * 1000) / 10 : null;
}

function buildStats() {
  const byHour = Array.from({ length: 24 }, (_, h) => ({ hour: h, total: 0, win: 0 }));
  for (const e of predLog) {
    byHour[e.hour].total++;
    if (e.win) byHour[e.hour].win++;
  }

  const last50 = predLog.slice(-50);

  return {
    overall: { ...STATS, rate: rate(STATS.win, STATS.total) },
    last50: {
      total: last50.length,
      win: last50.filter(e => e.win).length,
      rate: rate(last50.filter(e => e.win).length, last50.length)
    },
    byHour: byHour.map(h => ({ ...h, rate: rate(h.win, h.total) })),
    weights: engine.w,
    note: 'Tỷ lệ theo giờ chỉ có ý nghĩa khi mỗi giờ có vài trăm phiên trở lên.'
  };
}

// Walk-forward: mỗi phiên chỉ dùng dữ liệu TRƯỚC nó để dự đoán.
function backtest(minTrain = 30) {
  const hist = readHistory();
  const e = new Engine();
  const seq = [];
  let total = 0;
  let win = 0;

  for (let i = 0; i < hist.length; i++) {
    if (i >= 1) {
      const { x, p } = e.predict(seq, hist[i - 1]._id ?? hist[i - 1].id);
      const ok = (p >= 0.5 ? 1 : 0) === hist[i].result;
      if (i >= minTrain) {
        total++;
        if (ok) win++;
      }
      e.learn(x, hist[i].result);
    }
    seq.push(hist[i].result);
    e.observe(hist[i].result);
  }

  const pr = total ? win / total : 0;
  const se = total ? Math.sqrt((pr * (1 - pr)) / total) : 0;

  return {
    sessions: hist.length,
    tested: total,
    win,
    rate: rate(win, total),
    ci95: total
      ? [Math.round((pr - 1.96 * se) * 1000) / 10, Math.round((pr + 1.96 * se) * 1000) / 10]
      : null,
    baseline: 50,
    note: 'Nếu khoảng ci95 chứa 50 thì chưa có bằng chứng thuật toán tốt hơn đoán ngẫu nhiên.'
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

app.get('/stats', (req, res) => {
  try {
    res.json(buildStats());
  } catch (err) {
    res.status(500).json({ error: 'Stats error' });
  }
});

app.get('/backtest', (req, res) => {
  try {
    res.json(backtest());
  } catch (err) {
    res.status(500).json({ error: 'Backtest error' });
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

// Khôi phục trạng thái đã lưu; nếu chưa có thì dựng lại từ lịch sử.
if (!loadState()) {
  const h = readHistory();
  h.forEach(s => engine.observe(s.result));
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

  console.log('[BOOT] Engine v2: online logistic + decayed Bayes + MD5 + entropy guard');
});
