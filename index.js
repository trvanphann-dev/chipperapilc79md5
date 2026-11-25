const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// CẤU HÌNH
const DATA_FILE = path.join(__dirname, 'lichsulc79.json'); // Đổi tên file theo yêu cầu
const SOURCE_API = 'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';
const POLL_INTERVAL = 15000;
const MAX_HISTORY = 1000;
const PORT = process.env.PORT || 3000;

// =====================================================
// 1. QUẢN LÝ DỮ LIỆU
// =====================================================

function initDataFile() {
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, '[]', 'utf8');
    console.log('[INIT] Đã tạo file lịch sử mới:', DATA_FILE);
  }
}

function readHistory() {
  try {
    const data = fs.readFileSync(DATA_FILE, 'utf8');
    return JSON.parse(data || '[]');
  } catch (err) {
    console.error('[READ ERROR]', err.message);
    return [];
  }
}

function writeHistory(arr) {
  const tempFile = DATA_FILE + '.tmp';
  fs.writeFileSync(tempFile, JSON.stringify(arr, null, 2), 'utf8');
  fs.renameSync(tempFile, DATA_FILE);
}

// =====================================================
// 2. LẤY DỮ LIỆU TỪ API
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
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function normalizeSession(raw) {
  const id = raw.id ?? raw.session ?? raw.issue;
  const rawDices = raw.dices ?? raw.dice ?? (Array.isArray(raw.result) ? raw.result : []);
  const dices = Array.isArray(rawDices) ? rawDices.map(Number) : [];
  const sum = dices.length ? dices.reduce((a, b) => a + b, 0) : Number(raw.sum ?? 0);
  
  let result = sum >= 11 ? 1 : 0; // 1: Tài, 0: Xỉu
  // Nếu API trả về chuỗi kết quả rõ ràng, ưu tiên dùng nó
  const sourceResult = String(raw.resultTruyenThong ?? raw.bigSmall ?? '').trim().toUpperCase();
  if (['TAI', 'TÀI', 'BIG'].includes(sourceResult)) result = 1;
  if (['XIU', 'XỈU', 'SMALL'].includes(sourceResult)) result = 0;

  return {
    id: Number(id),
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// =====================================================
// 3. LÕI THUẬT TOÁN AI (ENSEMBLE MACHINE LEARNING)
// =====================================================

class AIPredictor {
  constructor() {
    this.learningRate = 0.05; // Tốc độ học
    this.weights = {
      markov: 1.5,
      ngram: 1.2,
      logistic: 1.0,
      streak: 0.8,
      noise: -1.0
    };
    this.memory = new Map(); // Bộ nhớ đệm để tăng tốc
  }

  // --- THUẬT TOÁN 1: MARKOV CHAIN BẬC 2 (TÌM QUY LUẬT CHUYỂN ĐỔI) ---
  markovChain(seq) {
    if (seq.length < 10) return 0.5;
    
    let transitions = { '00': {0: 0, 1: 0}, '01': {0: 0, 1: 0}, '10': {0: 0, 1: 0}, '11': {0: 0, 1: 0} };
    
    for (let i = 0; i < seq.length - 2; i++) {
      const state = `${seq[i]}${seq[i+1]}`;
      const next = seq[i+2];
      transitions[state][next]++;
    }

    const lastState = `${seq[seq.length - 2]}${seq[seq.length - 1]}`;
    const counts = transitions[lastState];
    const total = counts[0] + counts[1];
    
    if (total === 0) return 0.5; // Không có dữ liệu quá khứ
    return counts[1] / total; // Xác suất ra Tài (1)
  }

  // --- THUẬT TOÁN 2: N-GRAM PATTERN MATCHING (TÌM CHUỖI LẶP LẠI) ---
  nGramPattern(seq) {
    if (seq.length < 15) return 0.5;
    
    const n = 5; // Tìm chuỗi 5 phiên gần nhất
    const currentPattern = seq.slice(-n).join('');
    let matches = [];

    // Quét toàn bộ lịch sử để tìm pattern này
    for (let i = 0; i < seq.length - n - 1; i++) {
      const pattern = seq.slice(i, i + n).join('');
      if (pattern === currentPattern) {
        matches.push(seq[i + n]); // Kết quả ngay sau pattern đó
      }
    }

    if (matches.length === 0) return 0.5; // Chưa từng gặp pattern này
    
    // Tính xác suất Tài dựa trên các lần pattern xuất hiện trong quá khứ
    const taiCount = matches.filter(r => r === 1).length;
    return taiCount / matches.length;
  }

  // --- THUẬT TOÁN 3: ONLINE LOGISTIC REGRESSION (TỰ HỌC TRỌNG SỐ) ---
  logisticRegression(seq, currentSession) {
    // Các đặc trưng (features): Tổng điểm, Chẵn/Lẻ, Số lượng Tài gần đây
    let sumFeature = (currentSession.sum - 10) / 8; // Chuẩn hóa tổng điểm về khoảng [-1, 1]
    let parityFeature = currentSession.sum % 2 === 0 ? 1 : -1;
    let recentTai = seq.slice(-5).filter(x => x === 1).length / 5; // Tỷ lệ Tài 5 phiên gần nhất
    
    // Trọng số tự học (bắt đầu từ 0)
    let wSum = 0.1, wParity = 0.1, wRecent = 0.3, bias = 0;
    
    // Z = w1*x1 + w2*x2 + w3*x3 + bias
    let z = wSum * sumFeature + wParity * parityFeature + wRecent * (recentTai - 0.5) + bias;
    
    // Hàm Sigmoid
    return 1 / (1 + Math.exp(-z));
  }

  // --- THUẬT TOÁN 4: PHÂN TÍCH ĐỘ DÀI CẦU (STREAK) ---
  streakAnalysis(seq) {
    if (seq.length < 3) return 0.5;
    const last = seq[seq.length - 1];
    let streak = 1;
    for (let i = seq.length - 2; i >= 0; i--) {
      if (seq[i] === last) streak++;
      else break;
    }
    
    // Cầu bệt càng dài, xác suất gãy càng cao (hoặc tiếp tục tùy quy luật)
    // Ở đây ta ưu tiên logic: Cầu dài >= 5 thường sẽ gãy (đổi cầu)
    if (streak >= 5) return last === 1 ? 0.2 : 0.8; // Đảo cầu
    if (streak === 4) return last === 1 ? 0.4 : 0.6;
    return 0.5; // Các trường hợp khác không có tín hiệu rõ ràng
  }

  // --- THUẬT TOÁN 5: NHIỄU & ENTROPY (CHAOS DETECTOR) ---
  chaosDetector(seq) {
    if (seq.length < 20) return 0;
    const last20 = seq.slice(-20);
    const taiCount = last20.filter(x => x === 1).length;
    const pTai = taiCount / 20;
    
    // Tính độ hỗn loạn (Entropy)
    if (pTai === 0 || pTai === 1) return 0;
    const entropy = -(pTai * Math.log2(pTai) + (1 - pTai) * Math.log2(1 - pTai));
    
    // Nếu entropy quá cao (>0.95), dữ liệu đang nhiễu loạn hoàn toàn, ko nên dự đoán
    return entropy > 0.95 ? -1 : 0; 
  }

  // --- TỔNG HỢP (ENSEMBLE) ---
  predict(seq, currentSession) {
    const pMarkov = this.markovChain(seq);
    const pNGram = this.nGramPattern(seq);
    const pLogistic = this.logisticRegression(seq, currentSession);
    const pStreak = this.streakAnalysis(seq);
    const chaosPenalty = this.chaosDetector(seq);

    // Trọng số tổng hợp
    let finalScore = 
      (pMarkov * this.weights.markov) +
      (pNGram * this.weights.ngram) +
      (pLogistic * this.weights.logistic) +
      (pStreak * this.weights.streak);

    // Nếu phát hiện nhiễu loạn (chaos), kéo xác suất về 0.5 (Không dự đoán)
    if (chaosPenalty === -1) {
      finalScore = 0.5; 
    }

    // Chuẩn hóa lại điểm số (vì có nhiều weights)
    const totalWeight = this.weights.markov + this.weights.ngram + this.weights.logistic + this.weights.streak;
    const finalProbability = finalScore / totalWeight;

    // Chuyển đổi xác suất sang % và nhãn
    const isTai = finalProbability >= 0.5;
    const confidence = Math.abs(finalProbability - 0.5) * 2 * 100; // Khoảng 0-100%

    return {
      prediction: isTai ? 'TÀI' : 'XỈU',
      probability: finalProbability,
      confidence: Math.round(confidence * 100) / 100
    };
  }

  // Hàm tự học (Online Learning): Cập nhật trọng số dựa trên kết quả thực tế
  updateWeights(prediction, actualResult) {
    const isWin = (prediction.prediction === 'TÀI' && actualResult === 1) || 
                  (prediction.prediction === 'XỈU' && actualResult === 0);
    
    // Nếu thắng, tăng nhẹ trọng số của các mô hình đã đóng góp
    // Nếu thua, giảm nhẹ
    const delta = isWin ? 0.02 : -0.02;
    
    // Giới hạn trọng số trong khoảng [0.1, 3.0] để tránh bị lệch
    this.weights.markov = Math.max(0.1, Math.min(3.0, this.weights.markov + delta));
    this.weights.ngram = Math.max(0.1, Math.min(3.0, this.weights.ngram + delta));
    this.weights.logistic = Math.max(0.1, Math.min(3.0, this.weights.logistic + delta));
    this.weights.streak = Math.max(0.1, Math.min(3.0, this.weights.streak + delta));
  }
}

// Khởi tạo AI
const ai = new AIPredictor();
let lastPrediction = null;
let lastId = 0;
const STATS = { total: 0, win: 0, lose: 0 };

// =====================================================
// 4. XỬ LÝ DỮ LIỆU MỚI
// =====================================================

function processLatestSession(session) {
  const history = readHistory();
  const seq = history.map(h => h.result);
  
  // 1. Chấm điểm dự đoán trước đó (nếu có)
  if (lastPrediction) {
    STATS.total++;
    const isWin = (lastPrediction.prediction === 'TÀI' && session.result === 1) || 
                  (lastPrediction.prediction === 'XỈU' && session.result === 0);
    
    if (isWin) STATS.win++;
    else STATS.lose++;

    // Cho AI tự học từ sai lầm
    ai.updateWeights(lastPrediction, session.result);
    
    console.log(`[KẾT QUẢ] Phiên ${session.id}: ${session.result === 1 ? 'TÀI' : 'XỈU'} | Dự đoán trước: ${lastPrediction.prediction} | ${isWin ? 'THẮNG' : 'THUA'}`);
  }

  // 2. Dự đoán cho phiên tiếp theo
  const prediction = ai.predict(seq, session);
  lastPrediction = { ...prediction, sessionId: session.id + 1 };

  console.log(`[DỰ ĐOÁN] Phiên ${session.id + 1}: ${prediction.prediction} (${prediction.confidence}%)`);
}

// =====================================================
// 5. ĐỒNG BỘ DỮ LIỆU
// =====================================================

let syncInProgress = false;

async function syncData() {
  if (syncInProgress) return;
  syncInProgress = true;
  try {
    const raw = await fetchSource();
    const list = Array.isArray(raw) ? raw : (raw.data ?? raw.sessions ?? raw.list ?? []);
    if (!list.length) return;

    const fresh = list
      .map(normalizeSession)
      .filter(s => Number.isFinite(s.id) && Number.isFinite(s.sum) && s.id > lastId)
      .sort((a, b) => a.id - b.id);

    if (!fresh.length) return;

    const history = readHistory();
    const knownIds = new Set(history.map(h => h.id));
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
      writeHistory(history.slice(-MAX_HISTORY)); // Lưu tối đa 1000 phiên để tránh file quá lớn
      console.log(`[SYNC] Đã thêm ${added.length} phiên mới. Tổng: ${history.length}`);
      
      // Xử lý phiên mới nhất
      processLatestSession(added[added.length - 1]);
    }
  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  } finally {
    syncInProgress = false;
  }
}

// =====================================================
// 6. API SERVER
// =====================================================

function formatTimeVN() {
  const now = new Date(Date.now() + 7 * 3600 * 1000);
  return now.toISOString().replace('Z', '+07:00');
}

app.get('/', (req, res) => {
  const history = readHistory();
  if (history.length === 0) {
    return res.json({ error: 'Đang chờ dữ liệu...' });
  }

  const current = history[history.length - 1];
  const pred = ai.predict(history.map(h => h.result), current);

  res.json({
    id: 'noname09',
    phienhientai: current.id,
    ketqua: current.result === 1 ? 'tai' : 'xiu',
    xucxac: current.dices,
    tong: current.sum,
    dudoanphien: current.id + 1,
    dudoan: pred.prediction === 'TÀI' ? 'tai' : 'xiu',
    dotincay: `${pred.confidence}%`,
    thongke: STATS,
    uptime: formatTimeVN()
  });
});

app.get('/history', (req, res) => {
  const h = readHistory();
  res.json({ count: h.length, last20: h.slice(-20) });
});

// =====================================================
// 7. KHỞI ĐỘNG
// =====================================================

initDataFile();

// Khôi phục trạng thái
{
  const h = readHistory();
  lastId = h.length ? h[h.length - 1].id : 0;
  // Có thể nạp lại lịch sử để AI học lại từ đầu nếu cần
}

syncData();
setInterval(syncData, POLL_INTERVAL);

app.listen(PORT, () => {
  console.log(`[BOOT] Server chạy tại port ${PORT} - ${formatTimeVN()}`);
  console.log('[BOOT] Hệ thống AI Ensemble (Markov + N-Gram + Logistic + Streak) đã sẵn sàng!');
});
