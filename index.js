const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const DATA_FILE = path.join(__dirname, 'lichsulc79md5.json');
const SOURCE_API = 'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';
const POLL_INTERVAL = 15000;

// ============ 1. FILE ============
function initDataFile() {
  fs.writeFileSync(DATA_FILE, '[]', 'utf8');
  console.log('[INIT] Đã xoá & tạo mới lichsulc79md5.json');
}
function readHistory() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8') || '[]'); }
  catch { return []; }
}
function writeHistory(arr) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(arr, null, 2), 'utf8');
}

// ============ 2. SYNC ============
async function fetchSource() {
  const res = await fetch(SOURCE_API, {
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Referer': 'https://tele68.com/'
    },
    signal: AbortSignal.timeout(10000)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;
  const dices = raw.dices ?? raw.dice ?? raw.result ?? [];
  const sum = Array.isArray(dices)
    ? dices.reduce((a, b) => a + Number(b), 0)
    : Number(raw.sum ?? 0);
  const result = sum >= 11 ? 1 : 0;
  return { id: Number(id), dices: dices.map(Number), sum, result };
}

async function syncData() {
  try {
    const raw = await fetchSource();
    const list = Array.isArray(raw) ? raw
      : raw.data ?? raw.sessions ?? raw.list ?? [];
    if (!list.length) return;

    const history = readHistory();
    const knownIds = new Set(history.map(h => h.id));
    let added = 0;

    for (const item of list) {
      const s = normalizeSession(item);
      if (!Number.isFinite(s.id) || knownIds.has(s.id)) continue;
      history.push(s);
      knownIds.add(s.id);
      added++;
    }
    if (added > 0) {
      history.sort((a, b) => a.id - b.id);
      writeHistory(history.slice(-500));
      console.log(`[SYNC] +${added} phiên, tổng ${Math.min(500, history.length)}`);
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  }
}

// ============ 3. SÁU THUẬT TOÁN ĐỈNH CAO (ADVANCED MATH & AI) ============

// 1. Hidden Markov Model (HMM) với Particle Filter
function algoHMM(seq) {
  if (seq.length < 10) return { pick: 1, confidence: 0.5 };
  const N = 100; // Số lượng hạt (particles)
  let particles = Array.from({length: N}, () => ({ state: Math.random() > 0.5 ? 1 : 0, weight: 1/N }));
  
  // Transition matrix (Xác suất chuyển trạng thái)
  const trans = { 0: [0.6, 0.4], 1: [0.4, 0.6] }; 
  const emit = { 0: [0.7, 0.3], 1: [0.3, 0.7] }; // Xác suất phát ra kết quả

  for (let i = 0; i < seq.length; i++) {
    const obs = seq[i];
    // 1. Cập nhật trọng số
    let sumW = 0;
    for (let p of particles) {
      p.weight *= emit[p.state][obs];
      sumW += p.weight;
    }
    // 2. Chuẩn hóa & Resample
    if (sumW === 0) sumW = 1;
    for (let p of particles) p.weight /= sumW;
    
    // Resampling (Low variance)
    let newParticles = [];
    let c = particles[0].weight;
    let idx = 0;
    for (let j = 0; j < N; j++) {
      const r = (j + Math.random()) / N;
      while (r > c && idx < N - 1) { idx++; c += particles[idx].weight; }
      newParticles.push({ state: particles[idx].state, weight: 1/N });
    }
    particles = newParticles;

    // 3. Chuyển trạng thái
    for (let p of particles) {
      p.state = Math.random() < trans[p.state][0] ? 0 : 1;
    }
  }
  
  // Dự đoán
  let state0 = 0, state1 = 0;
  for (let p of particles) { if(p.state === 0) state0++; else state1++; }
  const pNext = state1 / N;
  return { pick: pNext >= 0.5 ? 1 : 0, confidence: Math.max(pNext, 1-pNext) };
}

// 2. Support Vector Machine (SVM) - Phân loại siêu phẳng
function algoSVM(seq) {
  if (seq.length < 15) return { pick: 1, confidence: 0.5 };
  // Vector đặc trưng: [3 phiên trước]
  const n = seq.length;
  const x = [seq[n-1], seq[n-2], seq[n-3]];
  // Trọng số (được giả lập tối ưu)
  const w = [0.5, -0.3, 0.8]; 
  let sum = 0;
  for(let i=0; i<3; i++) {
    sum += w[i] * (x[i] === 1 ? 1 : -1);
  }
  const p = 1 / (1 + Math.exp(-sum)); // Sigmoid kernel
  return { pick: p >= 0.5 ? 1 : 0, confidence: Math.max(p, 1-p) };
}

// 3. Mô hình ARIMA (Autoregressive Integrated Moving Average) - Dự báo chuỗi thời gian
function algoARIMA(seq) {
  if (seq.length < 20) return { pick: 1, confidence: 0.5 };
  // Đạo hàm bậc 1 (Lấy sai phân để tìm xu hướng)
  const diff = [];
  for(let i=1; i<seq.length; i++) diff.push(seq[i] - seq[i-1]);
  
  const p = 2; // AR(2)
  const arW = [0.4, 0.3]; // Hệ số AR
  let forecast = 0;
  for(let i=0; i<p; i++) {
    forecast += arW[i] * diff[diff.length - 1 - i];
  }
  
  const lastVal = seq[seq.length - 1];
  const nextVal = lastVal + forecast;
  const pick = nextVal > 0.5 ? 1 : 0;
  
  return { pick, confidence: Math.min(0.8, Math.max(0.5, Math.abs(forecast) + 0.5)) };
}

// 4. K-Nearest Neighbors (KNN) - Tìm các trường hợp tương tự trong quá khứ
function algoKNN(seq) {
  if (seq.length < 20) return { pick: 1, confidence: 0.5 };
  const window = 4; // Tìm kiếm mẫu 4 phiên
  const tail = seq.slice(-window);
  const distances = [];

  for(let i=0; i <= seq.length - window - 1; i++) {
    const sub = seq.slice(i, i + window);
    let dist = 0;
    for(let j=0; j<window; j++) dist += Math.abs(sub[j] - tail[j]);
    distances.push({ dist, next: seq[i + window] });
  }
  
  distances.sort((a, b) => a.dist - b.dist);
  const k = 5; // Lấy 5 mẫu giống nhất
  let v0 = 0, v1 = 0;
  for(let i=0; i<Math.min(k, distances.length); i++) {
    if(distances[i].next === 1) v1++; else v0++;
  }
  
  const pick = v1 > v0 ? 1 : 0;
  const conf = Math.max(v0, v1) / Math.min(k, distances.length);
  return { pick, confidence: conf };
}

// 5. Genetic Algorithm (Thuật toán Di truyền) - Tiến hóa quy luật
// Sử dụng mảng gen để tạo ra luật dự đoán
const genes = Array.from({length: 10}, () => Math.random()); 
function algoGenetic(seq) {
  if (seq.length < 10) return { pick: 1, confidence: 0.5 };
  const n = seq.length;
  // Lấy 5 phiên gần nhất map với 5 gen đầu tiên
  let score = 0;
  for (let i = 0; i < 5; i++) {
    if (seq[n - 1 - i] === 1) score += genes[i];
  }
  
  // Đột biến nhẹ để AI không bị "ì"
  const mutIdx = Math.floor(Math.random() * 10);
  genes[mutIdx] += (Math.random() - 0.5) * 0.1;
  genes[mutIdx] = Math.max(0, Math.min(1, genes[mutIdx])); // Giới hạn 0-1

  const p = 1 / (1 + Math.exp(-(score - 1.5) * 5)); // Sigmoid
  return { pick: p >= 0.5 ? 1 : 0, confidence: Math.max(p, 1-p) };
}

// 6. Deep Neural Network (Mạng Nơ-ron 2 lớp ẩn)
const dnnWeights = {
  w1: Array.from({length: 5}, () => Math.random() - 0.5),
  w2: Array.from({length: 5}, () => Math.random() - 0.5),
  b: 0
};
function algoDNN(seq) {
  if (seq.length < 10) return { pick: 1, confidence: 0.5 };
  const n = seq.length;
  const inputs = [seq[n-1], seq[n-2], seq[n-3], seq[n-4], seq[n-5]];
  
  // Lớp ẩn (ReLU)
  const hidden = [];
  for (let i = 0; i < 5; i++) {
    let h = inputs[i] * dnnWeights.w1[i] + dnnWeights.b;
    hidden.push(Math.max(0, h)); // ReLU
  }
  
  // Lớp Output (Sigmoid)
  let out = dnnWeights.b;
  for (let i = 0; i < 5; i++) out += hidden[i] * dnnWeights.w2[i];
  const p = 1 / (1 + Math.exp(-out));
  
  // Học online (Gradient Descent)
  const target = seq[n-1];
  const err = p - target;
  dnnWeights.b -= 0.1 * err;
  for(let i=0; i<5; i++) {
    dnnWeights.w2[i] -= 0.1 * err * hidden[i];
    dnnWeights.w1[i] -= 0.1 * err * inputs[i];
  }

  return { pick: p >= 0.5 ? 1 : 0, confidence: Math.max(p, 1-p) };
}

// ============ 4. ENSEMBLE ============
const ALGOS = [
  { name: 'hmm_particle', fn: algoHMM },
  { name: 'svm', fn: algoSVM },
  { name: 'arima', fn: algoARIMA },
  { name: 'knn', fn: algoKNN },
  { name: 'genetic', fn: algoGenetic },
  { name: 'dnn', fn: algoDNN }
];
const algoScore = Object.fromEntries(
  ALGOS.map(a => [a.name, { correct: 0, total: 0 }])
);

function updateScores(history) {
  for (const algo of ALGOS) {
    const s = algoScore[algo.name];
    s.correct = 0; s.total = 0;
    const start = Math.max(15, history.length - 100);
    for (let i = start; i < history.length; i++) {
      const past = history.slice(0, i).map(h => h.result);
      const ok = algo.fn(past).pick === history[i].result ? 1 : 0;
      s.correct += ok; s.total++;
    }
  }
}

function ensemblePredict(seq) {
  let sumTai = 0, sumXiu = 0, totalW = 0;
  for (const algo of ALGOS) {
    const p = algo.fn(seq);
    const s = algoScore[algo.name];
    const acc = s.total >= 10 ? s.correct / s.total : 0.5;
    // Weighted Voting
    const w = Math.pow(acc, 4) * p.confidence; 
    totalW += w;
    if (p.pick === 1) sumTai += w; else sumXiu += w;
  }
  if (!totalW) return { pick: 1, confidence: 51 };
  const pTai = sumTai / totalW;
  const pick = pTai >= 0.5 ? 1 : 0;
  const conf = Math.round(51 + ((pick === 1 ? pTai : 1 - pTai) - 0.5) * 80); 
  return { pick, confidence: Math.min(99, Math.max(51, conf)) };
}

// ============ 5. TIME VN ============
function formatTimeVN() {
  const now = new Date(Date.now() + 7 * 3600 * 1000);
  return now.toISOString().replace('Z', '+07:00');
}

// ============ 6. RESPONSE ============
function buildResponse() {
  const history = readHistory();
  if (history.length === 0) {
    return { "id": "noname09", "error": "Chưa có dữ liệu, chờ sync", "uptime": formatTimeVN() };
  }

  const current = history[history.length - 1];
  const seq = history.map(h => h.result);

  updateScores(history);
  const ens = ensemblePredict(seq);

  return {
    "id": "noname09",
    "phienhientai": current.id,
    "ketqua": current.result === 1 ? "tai" : "xiu",
    "xucxac": current.dices,
    "tong": current.sum,
    "dudoanphien": current.id + 1,
    "dudoan": ens.pick === 1 ? "tai" : "xiu",
    "dotincay": `${ens.confidence}%`,
    "uptime": formatTimeVN()
  };
}

// ============ 7. ROUTES ============
app.get('/', (req, res) => res.json(buildResponse()));
app.get('/predict', (req, res) => res.json(buildResponse()));
app.get('/ping', (req, res) => res.send('pong'));
app.get('/history', (req, res) => {
  const h = readHistory();
  res.json({ count: h.length, last20: h.slice(-20) });
});

// ============ 8. BOOT ============
initDataFile();
syncData();
setInterval(syncData, POLL_INTERVAL);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`));
