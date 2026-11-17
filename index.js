
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
// 3. ENGINE v3: NHIỀU MÔ HÌNH BỎ PHIẾU + ĐỘ ĐỒNG THUẬN + HIỆU CHỈNH ĐỘ TIN CẬY
// =====================================================

const STATE_FILE = path.join(__dirname, 'state.json');
const STATE_VERSION = 4;

const DECAY = 0.97;       // phiên cũ giảm dần ảnh hưởng trong Bayes
const LR = 0.05;          // tốc độ học của mô hình logistic
const L2 = 0.002;
const W_CLIP = 2.5;
const EMA_RATE = 0.05;    // tốc độ cập nhật độ tin cậy của từng mô hình
const CALIB_PRIOR = 20;   // số "phiên ảo" kéo độ tin cậy về ước lượng ban đầu
// Không thiên vị: mọi trọng số khởi đầu = 0, không có hệ số bias.
// Chiều hướng (theo cầu hay bẻ cầu) hoàn toàn do dữ liệu quyết định.
const FEATURES = ['last', 'bayes', 'streak', 'alt', 'md5', 'short', 'long'];
const W_INIT = { last: 0, bayes: 0, streak: 0, alt: 0, md5: 0, short: 0, long: 0 };

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0.5;
}

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

// Markov bậc k: P(Tài | k phiên gần nhất), đếm trực tiếp trên lịch sử
function markovP(seq, order, minLen) {
  const n = seq.length;
  if (n < minLen) return null;

  const cnt = {};
  for (let i = order; i < n; i++) {
    const key = seq.slice(i - order, i).join('');
    const c = cnt[key] || (cnt[key] = [0, 0]);
    c[seq[i]]++;
  }

  const c = cnt[seq.slice(n - order).join('')];
  if (!c) return null;

  return (c[1] + 1) / (c[0] + c[1] + 2);
}

// ---- Mô hình logistic tự học + Bayes có phân rã thời gian ----
class Engine {
  constructor() {
    this.a = 1;
    this.b = 1;
    this.w = { ...W_INIT };
  }

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
      last: n ? dir : 0,
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

// ---- Brain: nhiều mô hình bỏ phiếu, tính đồng thuận, hiệu chỉnh độ tin cậy ----
// Chỉ dùng các mô hình trung lập về chiều hướng: logistic (học dấu từ 0),
// Markov các bậc (đếm thực tế), và tín hiệu MD5 (đối xứng 50/50).
const VOTER_NAMES = ['logit', 'md5', 'markov1', 'markov2', 'markov3'];

class Brain {
  constructor() {
    this.engine = new Engine();
    this.ema = {};                       // độ chính xác trượt của từng mô hình
    VOTER_NAMES.forEach(n => { this.ema[n] = 0.5; });
    this.calib = {};                     // "k/N" -> { n, win }
  }

  observe(result) {
    this.engine.observe(result);
  }

  decide(seq, id) {
    const n = seq.length;
    const eg = this.engine.predict(seq, id);

    // 1) Mỗi mô hình đưa ra xác suất Tài
    const raw = {
      logit: eg.p,
      md5: md5Signal(id) === 1 ? 0.6 : 0.4,
      markov1: markovP(seq, 1, 10),
      markov2: markovP(seq, 2, 20),
      markov3: markovP(seq, 3, 40)
    };

    // 2) Bỏ phiếu: mô hình thiếu dữ liệu hoặc không có ý kiến thì bỏ qua
    const votes = {};
    for (const name of VOTER_NAMES) {
      const v = raw[name];
      if (v !== null && v !== undefined && Math.abs(v - 0.5) >= 0.01) {
        votes[name] = v;
      }
    }

    const names = Object.keys(votes);
    const N = names.length;

    // 3) Trọng số mỗi mô hình theo độ chính xác trượt thực tế của nó
    const weight = name => 0.1 + Math.max(0, this.ema[name] - 0.5) * 10;

    let sumW = 0;
    let sumWP = 0;
    for (const name of names) {
      const w = weight(name);
      sumW += w;
      sumWP += w * votes[name];
    }

    const p = N ? sumWP / sumW : 0.5;
    // Hoà (p đúng 0.5): chọn theo MD5 của id phiên, không mặc định về Tài
    const side = p > 0.5 ? 1 : p < 0.5 ? 0 : (md5Signal(id) === 1 ? 1 : 0);

    // 4) Độ đồng thuận: tỷ lệ mô hình (có trọng số) cùng phía với kết luận
    let agreeW = 0;
    let k = 0;
    for (const name of names) {
      if ((votes[name] >= 0.5 ? 1 : 0) === side) {
        agreeW += weight(name);
        k++;
      }
    }
    const agree = N ? agreeW / sumW : 0.5;

    // 5) Độ tin cậy thô từ cường độ + đồng thuận
    const cRaw = clamp(0.5 + Math.abs(p - 0.5) * (0.5 + agree), 0.5, 0.95);

    // 6) Hiệu chỉnh: kéo dần về tỷ lệ đúng thật của đúng mức đồng thuận này
    const key = `${k}/${N}`;
    const b = this.calib[key] || { n: 0, win: 0 };
    const conf = (b.win + CALIB_PRIOR * cRaw) / (b.n + CALIB_PRIOR);

    return {
      votes,
      key,
      k,
      N,
      x: eg.x,
      p,
      final: side === 1 ? 'TÀI' : 'XỈU',
      confidence: Math.round(conf * 100),
      caution: chaosBlock(seq)
    };
  }

  // Học từ kết quả thật của phiên đã dự đoán
  learn(dec, result) {
    for (const name of Object.keys(dec.votes)) {
      const correct = (dec.votes[name] >= 0.5 ? 1 : 0) === result ? 1 : 0;
      this.ema[name] = (1 - EMA_RATE) * this.ema[name] + EMA_RATE * correct;
    }

    this.engine.learn(dec.x, result);

    const b = this.calib[dec.key] || (this.calib[dec.key] = { n: 0, win: 0 });
    b.n++;
    if ((dec.final === 'TÀI' ? 1 : 0) === result) b.win++;
  }

  toJSON() {
    return {
      a: this.engine.a,
      b: this.engine.b,
      w: this.engine.w,
      ema: this.ema,
      calib: this.calib
    };
  }

  load(o) {
    this.engine.a = o.a;
    this.engine.b = o.b;
    this.engine.w = { ...W_INIT, ...o.w };
    this.ema = { ...this.ema, ...(o.ema || {}) };
    this.calib = o.calib || {};
  }
}

let brain = new Brain();

// =====================================================
// 4. STATE: DỰ ĐOÁN, THỐNG KÊ, CAUTION, LƯU TRẠNG THÁI
// =====================================================

let lastPredict = null;           // { forId, ...quyết định }
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
      version: STATE_VERSION,
      brain: brain.toJSON(),
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
    // State của phiên bản cũ (có trọng số khởi tạo thiên vị) -> bỏ, học lại từ 0
    if (!st.brain || st.version !== STATE_VERSION) return false;

    brain.load(st.brain);
    lastId = st.lastId || 0;
    Object.assign(STATS, st.stats || {});
    CAUTION.data = Array.isArray(st.caution) ? st.caution : [];
    lastPredict = st.lastPredict || null;
    predLog = Array.isArray(st.predLog) ? st.predLog : [];

    console.log('[INIT] Brain state restored');
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
    const win = lastPredict.final === real;

    brain.learn(lastPredict, session.result);

    STATS.total++;
    if (win) STATS.win++;
    else STATS.lose++;

    predLog.push({
      phien: session.id,
      predict: lastPredict.final,
      real,
      win,
      key: lastPredict.key,
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

  const d = brain.decide(seq, session._id ?? session.id);
  lastPredict = { forId: session.id + 1, ...d };

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
      brain.observe(s.result);
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
  const d = brain.decide(seq, current._id ?? current.id);

  return {
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,
    dudoanphien: current.id + 1,
    dudoan: d.final === 'TÀI' ? 'tai' : 'xiu',
    dotincay: `${d.confidence}%`,
    dongthuan: `${d.k}/${d.N}`,
    uptime: formatTimeVN()
  };
}

// =====================================================
// 8. THỐNG KÊ & BACKTEST (kiểm chứng độ chính xác thật)
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
  const winLast50 = last50.filter(e => e.win).length;

  return {
    overall: { ...STATS, rate: rate(STATS.win, STATS.total) },
    last50: { total: last50.length, win: winLast50, rate: rate(winLast50, last50.length) },
    byConsensus: Object.entries(brain.calib)
      .map(([key, v]) => ({ key, n: v.n, win: v.win, rate: rate(v.win, v.n) }))
      .sort((a, b) => b.n - a.n),
    voters: brain.ema,
    byHour: byHour.map(h => ({ ...h, rate: rate(h.win, h.total) })),
    weights: brain.engine.w,
    note: 'byConsensus cho biết tỷ lệ đúng thật ở từng mức đồng thuận; dotincay được hiệu chỉnh theo số này.'
  };
}

// Walk-forward: mỗi phiên chỉ dùng dữ liệu TRƯỚC nó để dự đoán.
function backtest(minTrain = 40) {
  const hist = readHistory();
  const b = new Brain();
  const seq = [];
  let total = 0;
  let win = 0;
  let confSum = 0;

  for (let i = 0; i < hist.length; i++) {
    if (i >= 1) {
      const d = b.decide(seq, hist[i - 1]._id ?? hist[i - 1].id);
      const ok = (d.final === 'TÀI' ? 1 : 0) === hist[i].result;
      if (i >= minTrain) {
        total++;
        confSum += d.confidence;
        if (ok) win++;
      }
      b.learn(d, hist[i].result);
    }
    seq.push(hist[i].result);
    b.observe(hist[i].result);
  }

  const pr = total ? win / total : 0;
  const se = total ? Math.sqrt((pr * (1 - pr)) / total) : 0;

  return {
    sessions: hist.length,
    tested: total,
    win,
    rate: rate(win, total),
    avgConfidence: total ? Math.round(confSum / total) : null,
    ci95: total
      ? [Math.round((pr - 1.96 * se) * 1000) / 10, Math.round((pr + 1.96 * se) * 1000) / 10]
      : null,
    baseline: 50,
    note: 'Nếu avgConfidence ≈ rate thì độ tin cậy đang chuẩn. Nếu ci95 chứa 50 thì chưa tốt hơn đoán ngẫu nhiên.'
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
  h.forEach(s => brain.observe(s.result));
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

  console.log('[BOOT] Engine v3: 6-model consensus + calibrated confidence');
});
