
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
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8') || '[]');
  } catch {
    return [];
  }
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

  return {
    id: Number(id),
    dices: dices.map(Number),
    sum,
    result
  };
}

async function syncData() {
  try {
    const raw = await fetchSource();

    const list = Array.isArray(raw)
      ? raw
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

      console.log(
        `[SYNC] +${added} phiên, tổng ${Math.min(500, history.length)}`
      );
    }

  } catch (err) {
    console.error('[SYNC ERROR]', err.message);
  }
}

// ============ 3. STATISTICAL ENGINE ============

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

const sigmoid = x => {
  x = clamp(x, -20, 20);
  return 1 / (1 + Math.exp(-x));
};

function modelResult(pTai, support = 0) {
  return {
    pTai: clamp(pTai, 0.01, 0.99),
    support: Math.max(0, support)
  };
}

// Xác suất nền với Laplace Smoothing
function getBaseProbability(seq, window = 200) {
  const data = seq.slice(-window);

  if (!data.length) return 0.5;

  let tai = 1;
  let xiu = 1;

  for (const value of data) {
    if (value === 1) tai++;
    else xiu++;
  }

  return tai / (tai + xiu);
}

// ============ 3.1 MARKOV TRANSITION ============

function algoHMM(seq) {
  if (seq.length < 5) {
    return modelResult(0.5, 0);
  }

  const last = seq[seq.length - 1];
  const start = Math.max(0, seq.length - 201);

  let tai = 1;
  let xiu = 1;
  let support = 0;

  for (let i = start; i < seq.length - 1; i++) {
    if (seq[i] !== last) continue;

    if (seq[i + 1] === 1) tai++;
    else xiu++;

    support++;
  }

  return modelResult(
    tai / (tai + xiu),
    support
  );
}

// ============ 3.2 LOGISTIC REGRESSION ============

function algoSVM(seq) {
  const window = 5;

  if (seq.length < 25) {
    return modelResult(0.5, 0);
  }

  const samples = [];

  for (let i = window; i < seq.length; i++) {
    const features = [];

    for (let j = 1; j <= window; j++) {
      features.push(seq[i - j] === 1 ? 1 : -1);
    }

    samples.push({
      x: features,
      y: seq[i]
    });
  }

  let weights = Array(window).fill(0);
  let bias = 0;

  const lr = 0.025;
  const lambda = 0.01;

  // Huấn luyện Logistic Regression
  for (let epoch = 0; epoch < 100; epoch++) {
    for (const sample of samples) {
      let z = bias;

      for (let j = 0; j < window; j++) {
        z += weights[j] * sample.x[j];
      }

      const error = sigmoid(z) - sample.y;

      for (let j = 0; j < window; j++) {
        weights[j] -= lr * (
          error * sample.x[j] +
          lambda * weights[j]
        );
      }

      bias -= lr * error;
    }
  }

  // Dự đoán phiên tiếp theo
  const input = [];

  for (let j = 1; j <= window; j++) {
    input.push(
      seq[seq.length - j] === 1 ? 1 : -1
    );
  }

  let z = bias;

  for (let j = 0; j < window; j++) {
    z += weights[j] * input[j];
  }

  return modelResult(
    sigmoid(z),
    samples.length
  );
}

// ============ 3.3 AUTOREGRESSIVE MODEL ============

function algoARIMA(seq) {
  const lag = 3;

  if (seq.length < 30) {
    return modelResult(0.5, 0);
  }

  const samples = [];

  for (let i = lag; i < seq.length; i++) {
    samples.push({
      x: [
        1,
        seq[i - 1] - 0.5,
        seq[i - 2] - 0.5,
        seq[i - 3] - 0.5
      ],
      y: seq[i] - 0.5
    });
  }

  let weights = [0, 0, 0, 0];

  const lr = 0.015;

  // Huấn luyện hồi quy tuyến tính
  for (let epoch = 0; epoch < 120; epoch++) {
    for (const sample of samples) {
      let predicted = 0;

      for (let j = 0; j < 4; j++) {
        predicted += weights[j] * sample.x[j];
      }

      const error = predicted - sample.y;

      for (let j = 0; j < 4; j++) {
        weights[j] -= lr * error * sample.x[j];
      }
    }
  }

  const input = [
    1,
    seq[seq.length - 1] - 0.5,
    seq[seq.length - 2] - 0.5,
    seq[seq.length - 3] - 0.5
  ];

  let forecast = 0;

  for (let j = 0; j < 4; j++) {
    forecast += weights[j] * input[j];
  }

  return modelResult(
    clamp(0.5 + forecast, 0.05, 0.95),
    samples.length
  );
}

// ============ 3.4 K-NEAREST NEIGHBORS ============

function algoKNN(seq) {
  const window = 5;

  if (seq.length < 25) {
    return modelResult(0.5, 0);
  }

  const tail = seq.slice(-window);
  const candidates = [];

  const start = Math.max(0, seq.length - 250);

  // Chỉ lấy mẫu quá khứ, không để mẫu chồng lên chuỗi hiện tại
  for (
    let i = start;
    i <= seq.length - 2 * window - 1;
    i++
  ) {
    let distance = 0;

    for (let j = 0; j < window; j++) {
      if (seq[i + j] !== tail[j]) {
        distance++;
      }
    }

    candidates.push({
      distance,
      next: seq[i + window],
      index: i
    });
  }

  candidates.sort((a, b) => {
    return a.distance - b.distance || b.index - a.index;
  });

  const neighbors = candidates.slice(0, 10);

  if (!neighbors.length) {
    return modelResult(0.5, 0);
  }

  let tai = 1;
  let xiu = 1;

  for (const item of neighbors) {
    const weight = 1 / (1 + item.distance);

    if (item.next === 1) tai += weight;
    else xiu += weight;
  }

  return modelResult(
    tai / (tai + xiu),
    neighbors.length
  );
}

// ============ 3.5 STREAK BAYESIAN MODEL ============

function algoGenetic(seq) {
  if (seq.length < 15) {
    return modelResult(0.5, 0);
  }

  let run = 1;

  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === seq[i + 1]) {
      run++;
    } else {
      break;
    }
  }

  run = Math.min(run, 6);

  const start = Math.max(0, seq.length - 250);

  let tai = 1;
  let xiu = 1;
  let support = 0;

  for (let i = start + 1; i < seq.length - 1; i++) {
    let previousRun = 1;

    for (
      let j = i - 1;
      j >= Math.max(start, i - 6);
      j--
    ) {
      if (seq[j] === seq[j + 1]) {
        previousRun++;
      } else {
        break;
      }
    }

    if (
      seq[i] === seq[seq.length - 1] &&
      Math.min(previousRun, 6) === run
    ) {
      if (seq[i + 1] === 1) tai++;
      else xiu++;

      support++;
    }
  }

  return modelResult(
    tai / (tai + xiu),
    support
  );
}

// ============ 3.6 NEURAL NETWORK ============

function algoDNN(seq) {
  const inputSize = 5;
  const hiddenSize = 6;

  if (seq.length < 35) {
    return modelResult(0.5, 0);
  }

  const samples = [];

  for (let i = inputSize; i < seq.length; i++) {
    samples.push({
      x: Array.from(
        { length: inputSize },
        (_, j) => seq[i - 1 - j] === 1 ? 1 : -1
      ),
      y: seq[i]
    });
  }

  // Khởi tạo trọng số xác định
  let w1 = Array.from(
    { length: hiddenSize },
    (_, i) => Array.from(
      { length: inputSize },
      (_, j) => Math.sin((i + 1) * (j + 2)) * 0.1
    )
  );

  let b1 = Array(hiddenSize).fill(0);

  let w2 = Array.from(
    { length: hiddenSize },
    (_, i) => Math.cos(i + 1) * 0.1
  );

  let b2 = 0;

  const lr = 0.015;

  // Huấn luyện mạng Neural Network
  for (let epoch = 0; epoch < 100; epoch++) {
    for (const sample of samples) {
      const hidden = [];

      for (let i = 0; i < hiddenSize; i++) {
        let z = b1[i];

        for (let j = 0; j < inputSize; j++) {
          z += w1[i][j] * sample.x[j];
        }

        hidden.push(Math.tanh(z));
      }

      let output = b2;

      for (let i = 0; i < hiddenSize; i++) {
        output += hidden[i] * w2[i];
      }

      const p = sigmoid(output);
      const error = p - sample.y;

      const oldW2 = [...w2];

      for (let i = 0; i < hiddenSize; i++) {
        w2[i] -= lr * error * hidden[i];

        const delta =
          error *
          oldW2[i] *
          (1 - hidden[i] * hidden[i]);

        for (let j = 0; j < inputSize; j++) {
          w1[i][j] -= lr * delta * sample.x[j];
        }

        b1[i] -= lr * delta;
      }

      b2 -= lr * error;
    }
  }

  // Dự đoán phiên tiếp theo
  const input = Array.from(
    { length: inputSize },
    (_, j) => seq[seq.length - 1 - j] === 1 ? 1 : -1
  );

  const hidden = [];

  for (let i = 0; i < hiddenSize; i++) {
    let z = b1[i];

    for (let j = 0; j < inputSize; j++) {
      z += w1[i][j] * input[j];
    }

    hidden.push(Math.tanh(z));
  }

  let output = b2;

  for (let i = 0; i < hiddenSize; i++) {
    output += hidden[i] * w2[i];
  }

  return modelResult(
    sigmoid(output),
    samples.length
  );
}

// ============ 4. PROBABILISTIC ENSEMBLE ============

const ALGOS = [
  { name: 'hmm_particle', fn: algoHMM, weight: 1.2 },
  { name: 'svm', fn: algoSVM, weight: 1.0 },
  { name: 'arima', fn: algoARIMA, weight: 0.8 },
  { name: 'knn', fn: algoKNN, weight: 1.1 },
  { name: 'genetic', fn: algoGenetic, weight: 0.8 },
  { name: 'dnn', fn: algoDNN, weight: 0.9 }
];

function ensemblePredict(seq) {
  if (!seq || seq.length < 10) {
    const base = getBaseProbability(seq || []);

    return {
      pick: base >= 0.5 ? 1 : 0,
      confidence: 50
    };
  }

  const base = getBaseProbability(seq, 200);

  // Xác suất nền có trọng số riêng
  let weightedProbability = base * 2;
  let totalWeight = 2;

  for (const algo of ALGOS) {
    try {
      const result = algo.fn(seq);

      if (
        !result ||
        !Number.isFinite(result.pTai) ||
        !Number.isFinite(result.support)
      ) {
        continue;
      }

      // Giảm ảnh hưởng mô hình thiếu dữ liệu
      const reliability =
        result.support / (result.support + 15);

      const effectiveWeight =
        algo.weight * reliability;

      if (effectiveWeight <= 0) continue;

      weightedProbability +=
        result.pTai * effectiveWeight;

      totalWeight += effectiveWeight;

    } catch (err) {
      console.error(
        `[MODEL ERROR] ${algo.name}:`,
        err.message
      );
    }
  }

  let pTai = weightedProbability / totalWeight;

  // Hạn chế xác suất quá cực đoan
  pTai = 0.5 + (pTai - 0.5) * 0.85;

  pTai = clamp(pTai, 0.05, 0.95);

  const pick = pTai >= 0.5 ? 1 : 0;

  // Xác suất ước lượng, chưa phải tỷ lệ thắng đã kiểm chứng
  const confidence = Math.round(
    Math.max(pTai, 1 - pTai) * 100
  );

  return {
    pick,
    confidence,
    probability: {
      tai: Number((pTai * 100).toFixed(2)),
      xiu: Number(((1 - pTai) * 100).toFixed(2))
    }
  };
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
    return {
      "id": "noname09",
      "error": "Chưa có dữ liệu, chờ sync",
      "uptime": formatTimeVN()
    };
  }

  const current = history[history.length - 1];

  const seq = history.map(h => h.result);

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

app.get('/', (req, res) => {
  res.json(buildResponse());
});

app.get('/predict', (req, res) => {
  res.json(buildResponse());
});

app.get('/ping', (req, res) => {
  res.send('pong');
});

app.get('/history', (req, res) => {
  const h = readHistory();

  res.json({
    count: h.length,
    last20: h.slice(-20)
  });
});

// ============ 8. BOOT ============

initDataFile();

syncData();

setInterval(syncData, POLL_INTERVAL);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`[BOOT] Port ${PORT} — ${formatTimeVN()}`);
});
