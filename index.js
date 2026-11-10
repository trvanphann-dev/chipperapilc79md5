
'use strict';

const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '1mb' }));

// ================== CẤU HÌNH ==================

const PORT = process.env.PORT || 10000;
const POLL_MS = 15000;
const MAX_HISTORY = 1500;
const MAX_FORECASTS = 500;

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const LEARNING_FILE = path.join(__dirname, 'learning_state.json');

const SOURCE_API =
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

// ================== TRẠNG THÁI ==================

let history = [];
let forecasts = {};
let stats = {
  evaluated: 0,
  correct: 0,
  wrong: 0
};

let lastSync = null;
let lastError = null;
let isSyncing = false;
let startedAt = new Date().toISOString();

// ================== HÀM CƠ BẢN ==================

function safeNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeLabel(value) {
  if (value === null || value === undefined) return null;

  const s = String(value).trim().toLowerCase();

  if (
    s === 'tai' ||
    s === 'tài' ||
    s === 'big' ||
    s === 'over' ||
    s === '1'
  ) return 1;

  if (
    s === 'xiu' ||
    s === 'xỉu' ||
    s === 'small' ||
    s === 'under' ||
    s === '0'
  ) return 0;

  return null;
}

function getDice(obj) {
  const possible = [
    obj.dices,
    obj.dice,
    obj.xucxac,
    obj.xuc_xac,
    obj.rolls,
    obj.faces
  ];

  for (const item of possible) {
    if (Array.isArray(item)) {
      const dice = item.map(Number);

      if (
        dice.length === 3 &&
        dice.every(n => Number.isInteger(n) && n >= 1 && n <= 6)
      ) {
        return dice;
      }
    }

    if (typeof item === 'string') {
      const nums = item.match(/[1-6]/g);

      if (nums && nums.length === 3) {
        return nums.map(Number);
      }
    }
  }

  return null;
}

function normalizeSession(obj) {
  if (!obj || typeof obj !== 'object') return null;

  const rawId =
    obj.id ??
    obj.session ??
    obj.sessionId ??
    obj.issue ??
    obj.period ??
    obj.round ??
    obj.phien ??
    obj.phienhientai;

  if (rawId === undefined || rawId === null) return null;

  const id = String(rawId).trim();
  if (!id) return null;

  const dice = getDice(obj);

  let sum = safeNumber(
    obj.sum ?? obj.total ?? obj.tong ?? obj.totalScore
  );

  if (sum === null && dice) {
    sum = dice.reduce((a, b) => a + b, 0);
  }

  const rawResult =
    obj.result ??
    obj.ketqua ??
    obj.outcome ??
    obj.tx ??
    obj.taiXiu ??
    obj.bigSmall;

  let result = normalizeLabel(rawResult);

  // Nếu API không trả kết quả nhưng có tổng điểm,
  // dùng luật thông thường: 11-17 là Tài, 4-10 là Xỉu.
  if (result === null && sum !== null) {
    result = sum >= 11 ? 1 : 0;
  }

  if (result === null) return null;

  return {
    id,
    dice,
    sum,
    result,
    label: result === 1 ? 'tai' : 'xiu',
    timestamp: obj.timestamp || obj.time || new Date().toISOString()
  };
}

function collectObjects(input, output = [], depth = 0) {
  if (depth > 8 || input === null || input === undefined) {
    return output;
  }

  if (Array.isArray(input)) {
    for (const item of input) {
      collectObjects(item, output, depth + 1);
    }
    return output;
  }

  if (typeof input !== 'object') return output;

  output.push(input);

  for (const value of Object.values(input)) {
    if (value && typeof value === 'object') {
      collectObjects(value, output, depth + 1);
    }
  }

  return output;
}

function extractSessions(payload) {
  const objects = collectObjects(payload);
  const found = new Map();

  for (const obj of objects) {
    const session = normalizeSession(obj);

    if (session) {
      found.set(session.id, session);
    }
  }

  return [...found.values()].sort((a, b) => {
    const na = Number(a.id);
    const nb = Number(b.id);

    if (Number.isFinite(na) && Number.isFinite(nb)) {
      return na - nb;
    }

    return a.id.localeCompare(b.id);
  });
}

function nextSessionId(id) {
  const n = Number(id);

  if (Number.isFinite(n)) return String(n + 1);

  return null;
}

function sameId(a, b) {
  return String(a) === String(b);
}

// ================== LƯU DỮ LIỆU ==================

function writeJsonAtomic(file, data) {
  const temp = file + '.tmp';

  fs.writeFileSync(temp, JSON.stringify(data, null, 2));
  fs.renameSync(temp, file);
}

function saveHistory() {
  try {
    writeJsonAtomic(DATA_FILE, history);
  } catch (err) {
    lastError = 'Lỗi lưu lịch sử: ' + err.message;
  }
}

function saveLearning() {
  try {
    writeJsonAtomic(LEARNING_FILE, {
      version: 4,
      forecasts,
      stats,
      updatedAt: new Date().toISOString()
    });
  } catch (err) {
    lastError = 'Lỗi lưu learning: ' + err.message;
  }
}

function loadData() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const raw = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

      if (Array.isArray(raw)) {
        history = raw
          .map(normalizeSession)
          .filter(Boolean);
      }
    }
  } catch (err) {
    console.error('[LOAD] Không đọc được history:', err.message);
    history = [];
  }

  try {
    if (fs.existsSync(LEARNING_FILE)) {
      const raw = JSON.parse(fs.readFileSync(LEARNING_FILE, 'utf8'));

      if (raw && raw.version === 4) {
        forecasts = raw.forecasts || {};
        stats = raw.stats || stats;
      }
    }
  } catch (err) {
    console.error('[LOAD] Không đọc được learning:', err.message);
  }

  const unique = new Map();

  for (const item of history) {
    unique.set(String(item.id), item);
  }

  history = [...unique.values()].sort((a, b) => {
    const na = Number(a.id);
    const nb = Number(b.id);

    if (Number.isFinite(na) && Number.isFinite(nb)) {
      return na - nb;
    }

    return a.id.localeCompare(b.id);
  });

  history = history.slice(-MAX_HISTORY);

  console.log('[LOAD] Lịch sử:', history.length);
}

// ================== THUẬT TOÁN ==================

// Xác suất có làm mượt để tránh kết luận quá mạnh từ ít dữ liệu.
function smoothedProbability(values, alpha = 2) {
  let tai = 0;
  let xiu = 0;

  for (const value of values) {
    if (value === 1) tai++;
    else if (value === 0) xiu++;
  }

  return (tai + alpha) / (tai + xiu + alpha * 2);
}

// Trọng số gần đây giảm dần theo thời gian.
function recentProbability(data, limit = 100) {
  const rows = data.slice(-limit);

  if (!rows.length) return 0.5;

  let tai = 0;
  let xiu = 0;
  let weightSum = 0;

  rows.forEach((row, index) => {
    const weight = Math.pow(0.97, rows.length - 1 - index);

    if (row.result === 1) tai += weight;
    else xiu += weight;

    weightSum += weight;
  });

  if (!weightSum) return 0.5;

  return (tai + 1) / (tai + xiu + 2);
}

// Xác suất kết quả tiếp theo sau kết quả gần nhất.
function markovOneProbability(data) {
  if (data.length < 3) return 0.5;

  const last = data[data.length - 1].result;
  let tai = 0;
  let xiu = 0;

  for (let i = 0; i < data.length - 1; i++) {
    if (data[i].result !== last) continue;

    const next = data[i + 1].result;

    if (next === 1) tai++;
    else xiu++;
  }

  // Co về xác suất nền nếu số lần xuất hiện quá ít.
  const base = recentProbability(data, 150);
  const observed = (tai + 2 * base) / (tai + xiu + 2);

  const reliability = Math.min(1, (tai + xiu) / 12);

  return base * (1 - reliability) + observed * reliability;
}

// Xác suất sau cặp hai kết quả gần nhất.
function markovTwoProbability(data) {
  if (data.length < 4) return 0.5;

  const a = data[data.length - 2].result;
  const b = data[data.length - 1].result;

  let tai = 0;
  let xiu = 0;

  for (let i = 0; i < data.length - 2; i++) {
    if (
      data[i].result === a &&
      data[i + 1].result === b
    ) {
      if (data[i + 2].result === 1) tai++;
      else xiu++;
    }
  }

  const base = recentProbability(data, 150);
  const observed = (tai + 2 * base) / (tai + xiu + 2);

  const reliability = Math.min(1, (tai + xiu) / 10);

  return base * (1 - reliability) + observed * reliability;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function makePrediction(data) {
  if (!data.length) {
    return {
      prediction: null,
      probabilityTai: 0.5,
      confidence: 50,
      shouldBet: false,
      reason: 'Chưa có dữ liệu'
    };
  }

  const globalP = smoothedProbability(
    data.map(x => x.result),
    3
  );

  const recentP = recentProbability(data, 100);
  const markov1 = markovOneProbability(data);
  const markov2 = markovTwoProbability(data);

  // Không để Markov lấn át xác suất nền khi mẫu nhỏ.
  let probabilityTai =
    globalP * 0.20 +
    recentP * 0.35 +
    markov1 * 0.25 +
    markov2 * 0.20;

  probabilityTai = clamp(probabilityTai, 0.05, 0.95);

  const prediction = probabilityTai >= 0.5 ? 1 : 0;
  const confidence = Math.round(
    Math.max(probabilityTai, 1 - probabilityTai) * 100
  );

  const enoughData = data.length >= 50;
  const shouldBet = enoughData && confidence >= 56;

  let reason = 'Xác suất chưa đủ chênh lệch';

  if (!enoughData) {
    reason = 'Cần ít nhất 50 phiên để đánh giá dữ liệu';
  } else if (shouldBet) {
    reason = 'Có chênh lệch xác suất trong dữ liệu lịch sử';
  }

  return {
    prediction,
    probabilityTai: Number(probabilityTai.toFixed(4)),
    confidence,
    shouldBet,
    reason,
    models: {
      global: Number(globalP.toFixed(4)),
      recent: Number(recentP.toFixed(4)),
      markov1: Number(markov1.toFixed(4)),
      markov2: Number(markov2.toFixed(4))
    }
  };
}

// ================== ĐÁNH GIÁ DỰ ĐOÁN ==================

function evaluateForecast(session) {
  const forecast = forecasts[String(session.id)];

  if (!forecast || forecast.evaluated) return;

  const correct = forecast.prediction === session.result;

  forecast.evaluated = true;
  forecast.actual = session.result;
  forecast.correct = correct;
  forecast.evaluatedAt = new Date().toISOString();

  stats.evaluated++;

  if (correct) stats.correct++;
  else stats.wrong++;

  console.log(
    `[EVALUATE] Phiên ${session.id}: dự đoán ${
      forecast.prediction === 1 ? 'TÀI' : 'XỈU'
    }, thực tế ${session.label}, ${correct ? 'ĐÚNG' : 'SAI'}`
  );
}

function createForecast(sessionId) {
  if (!sessionId || forecasts[String(sessionId)]) return;

  const result = makePrediction(history);

  if (result.prediction === null) return;

  forecasts[String(sessionId)] = {
    sessionId: String(sessionId),
    prediction: result.prediction,
    label: result.prediction === 1 ? 'tai' : 'xiu',
    confidence: result.confidence,
    shouldBet: result.shouldBet,
    probabilityTai: result.probabilityTai,
    models: result.models,
    evaluated: false,
    createdAt: new Date().toISOString()
  };

  const keys = Object.keys(forecasts);

  if (keys.length > MAX_FORECASTS) {
    keys
      .sort((a, b) => {
        const na = Number(a);
        const nb = Number(b);

        if (Number.isFinite(na) && Number.isFinite(nb)) {
          return na - nb;
        }

        return a.localeCompare(b);
      })
      .slice(0, keys.length - MAX_FORECASTS)
      .forEach(key => delete forecasts[key]);
  }
}

// ================== XỬ LÝ PHIÊN MỚI ==================

function processSession(session) {
  const exists = history.some(
    item => sameId(item.id, session.id)
  );

  if (exists) return false;

  // Chấm dự đoán đã tạo cho chính phiên này trước khi học kết quả.
  evaluateForecast(session);

  history.push(session);
  history = history.slice(-MAX_HISTORY);

  // Sau khi học kết quả, tạo dự đoán cho phiên kế tiếp.
  const nextId = nextSessionId(session.id);

  if (nextId) {
    createForecast(nextId);
  }

  return true;
}

// ================== GỌI API NGUỒN ==================

async function fetchSource() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch(SOURCE_API, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        'user-agent': 'Mozilla/5.0'
      },
      signal: controller.signal
    });

    if (!response.ok) {
      throw new Error(`Source API HTTP ${response.status}`);
    }

    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

async function syncData() {
  if (isSyncing) return;

  isSyncing = true;

  try {
    const payload = await fetchSource();
    const sessions = extractSessions(payload);

    if (!sessions.length) {
      throw new Error(
        'API nguồn không trả về phiên hợp lệ. Cần kiểm tra cấu trúc JSON nguồn.'
      );
    }

    let added = 0;

    for (const session of sessions) {
      if (processSession(session)) added++;
    }

    // Luôn đảm bảo có dự đoán cho phiên kế tiếp mới nhất.
    const latest = history[history.length - 1];

    if (latest) {
      const nextId = nextSessionId(latest.id);

      if (nextId) createForecast(nextId);
    }

    saveHistory();
    saveLearning();

    lastSync = new Date().toISOString();
    lastError = null;

    console.log(
      `[SYNC] API có ${sessions.length} phiên, thêm mới ${added}, tổng lịch sử ${history.length}`
    );
  } catch (err) {
    lastError = err.name === 'AbortError'
      ? 'API nguồn phản hồi quá chậm hoặc timeout'
      : err.message;

    console.error('[SYNC ERROR]', lastError);
  } finally {
    isSyncing = false;
  }
}

// ================== API DỰ ĐOÁN ==================

function getCurrentPrediction() {
  const latest = history[history.length - 1];

  if (!latest) {
    return {
      status: 'waiting',
      message: 'Đang chờ dữ liệu từ API nguồn',
      count: 0
    };
  }

  const nextId = nextSessionId(latest.id);
  const forecast = nextId ? forecasts[String(nextId)] : null;

  const prediction = forecast || {
    ...makePrediction(history),
    sessionId: nextId,
    label: makePrediction(history).prediction === 1 ? 'tai' : 'xiu'
  };

  const p = prediction.prediction;

  return {
    status: 'ok',
    id: latest.id,
    phienhientai: latest.id,
    ketqua: latest.label,
    xucxac: latest.dice,
    tong: latest.sum,

    dudoanphien: nextId,
    dudoan: p === 1 ? 'tai' : 'xiu',
    dotincay: `${prediction.confidence}%`,
    xacsuattai: prediction.probabilityTai,

    nenchoi: Boolean(prediction.shouldBet),
    khuyennghi: prediction.shouldBet
      ? 'Có chênh lệch xác suất, nhưng không đảm bảo thắng'
      : 'Nên bỏ qua vì dữ liệu chưa đủ lợi thế',

    models: prediction.models || null,
    lichsu: history.length,
    lastSync,
    uptime: new Date().toISOString()
  };
}

// ================== ROUTES ==================

// Trang chủ
app.get('/', (req, res) => {
  res.json({
    name: 'LC79 MD5 Prediction API',
    status: 'running',
    endpoints: [
      '/status',
      '/api/status',
      '/ping',
      '/predict',
      '/history',
      '/forecast-history'
    ]
  });
});

// Hai route trạng thái được khai báo trực tiếp.
app.get('/status', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'LC79 MD5',
    uptime: process.uptime(),
    startedAt,
    lastSync,
    lastError,
    syncing: isSyncing,
    historyCount: history.length,
    forecastCount: Object.keys(forecasts).length,
    stats,
    accuracy: stats.evaluated
      ? Number((stats.correct / stats.evaluated * 100).toFixed(2))
      : null,
    time: new Date().toISOString()
  });
});

app.get('/api/status', (req, res) => {
  res.status(200).json({
    status: 'ok',
    uptime: process.uptime(),
    lastSync,
    lastError,
    syncing: isSyncing,
    historyCount: history.length,
    stats,
    time: new Date().toISOString()
  });
});

app.get('/ping', (req, res) => {
  res.status(200).send('pong');
});

// Dự đoán phiên kế tiếp
app.get('/predict', (req, res) => {
  res.status(200).json(getCurrentPrediction());
});

// Lịch sử kết quả
app.get('/history', (req, res) => {
  const limit = Math.min(
    Math.max(parseInt(req.query.limit, 10) || 20, 1),
    500
  );

  res.json({
    count: history.length,
    last: history.slice(-limit).reverse()
  });
});

// Lịch sử dự đoán và kết quả kiểm chứng
app.get('/forecast-history', (req, res) => {
  const limit = Math.min(
    Math.max(parseInt(req.query.limit, 10) || 50, 1),
    300
  );

  const rows = Object.values(forecasts)
    .sort((a, b) => {
      const na = Number(a.sessionId);
      const nb = Number(b.sessionId);

      if (Number.isFinite(na) && Number.isFinite(nb)) {
        return na - nb;
      }

      return a.sessionId.localeCompare(b.sessionId);
    })
    .slice(-limit)
    .reverse();

  res.json({
    count: rows.length,
    stats,
    accuracy: stats.evaluated
      ? Number((stats.correct / stats.evaluated * 100).toFixed(2))
      : null,
    data: rows
  });
});

// Đồng bộ thủ công
app.get('/sync', async (req, res) => {
  await syncData();

  res.json({
    status: lastError ? 'error' : 'ok',
    lastSync,
    lastError,
    historyCount: history.length
  });
});

// Reset có bảo vệ bằng token môi trường.
app.post('/admin/reset', (req, res) => {
  const token = process.env.RESET_TOKEN;

  if (!token || req.headers['x-reset-token'] !== token) {
    return res.status(403).json({
      status: 'error',
      message: 'Forbidden'
    });
  }

  history = [];
  forecasts = {};
  stats = {
    evaluated: 0,
    correct: 0,
    wrong: 0
  };

  saveHistory();
  saveLearning();

  res.json({
    status: 'ok',
    message: 'Đã reset dữ liệu học'
  });
});

// Route không tồn tại
app.use((req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Route not found',
    path: req.originalUrl,
    available: [
      '/',
      '/status',
      '/api/status',
      '/ping',
      '/predict',
      '/history',
      '/forecast-history',
      '/sync'
    ]
  });
});

// ================== KHỞI ĐỘNG ==================

loadData();

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log('====================================');
  console.log('[BOOT] LC79 MD5 API started');
  console.log('[BOOT] Port:', PORT);
  console.log('[BOOT] /status enabled');
  console.log('[BOOT] /api/status enabled');
  console.log('[BOOT] /ping enabled');
  console.log('[BOOT] History:', history.length);
  console.log('====================================');

  // Đồng bộ ngay khi khởi động.
  syncData();

  // Sau đó đồng bộ định kỳ.
  setInterval(syncData, POLL_MS);
});

server.on('error', err => {
  console.error('[SERVER ERROR]', err);
});

process.on('uncaughtException', err => {
  console.error('[UNCAUGHT]', err);
});

process.on('unhandledRejection', err => {
  console.error('[UNHANDLED REJECTION]', err);
});
