
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

// ============ 3. FIVE DETERMINISTIC ALGORITHMS ============

// Chuẩn hóa xác suất có làm trơn.
function probability(tai, total, prior = 1) {
  return (tai + prior) / (total + prior * 2);
}

// Chuẩn hóa kết quả dự đoán.
function makePrediction(pTai, support = 0) {
  const p = Math.max(0, Math.min(1, pTai));

  return {
    pick: p >= 0.5 ? 1 : 0,
    pTai: p,
    confidence: Math.max(p, 1 - p),
    support
  };
}

// --------------------------------------------------
// 1. N-GRAM MARKOV
// Phân tích chuỗi T/X xuất hiện trước đó.
// --------------------------------------------------
function algoNGram(seq) {
  const n = seq.length;
  const maxOrder = Math.min(7, n - 1);

  if (maxOrder < 1) return makePrediction(0.5);

  for (let order = maxOrder; order >= 1; order--) {
    const pattern = seq.slice(n - order).join('');

    let tai = 0;
    let total = 0;

    for (let i = 0; i + order < n; i++) {
      const candidate = seq.slice(i, i + order).join('');

      if (candidate !== pattern) continue;

      const next = seq[i + order];

      if (next !== 0 && next !== 1) continue;

      total++;

      if (next === 1) tai++;
    }

    if (total >= 2) {
      return makePrediction(
        probability(tai, total),
        total
      );
    }
  }

  return makePrediction(0.5);
}

// --------------------------------------------------
// 2. HIGHER-ORDER MARKOV
// Kết hợp chuyển trạng thái cấp 1, 2 và 3.
// --------------------------------------------------
function algoHigherMarkov(seq) {
  const n = seq.length;
  const maxOrder = Math.min(3, n - 1);

  let weightedTai = 0;
  let totalWeight = 0;
  let totalSupport = 0;

  for (let order = 1; order <= maxOrder; order++) {
    const pattern = seq.slice(n - order).join('');

    let tai = 0;
    let total = 0;

    for (let i = 0; i + order < n; i++) {
      const candidate = seq.slice(i, i + order).join('');

      if (candidate !== pattern) continue;

      const next = seq[i + order];

      if (next !== 0 && next !== 1) continue;

      total++;

      if (next === 1) tai++;
    }

    if (total === 0) continue;

    const p = probability(tai, total);
    const weight = total / (total + 4);

    weightedTai += p * weight;
    totalWeight += weight;
    totalSupport += total;
  }

  if (totalWeight === 0) {
    return makePrediction(0.5);
  }

  return makePrediction(
    weightedTai / totalWeight,
    totalSupport
  );
}

// --------------------------------------------------
// 3. RUN-LENGTH ANALYSIS
// Phân tích cầu bệt và độ dài chuỗi liên tiếp.
// --------------------------------------------------
function algoRunLength(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const current = seq[n - 1];

  let runLength = 1;

  for (let i = n - 2; i >= 0; i--) {
    if (seq[i] !== current) break;
    runLength++;
  }

  let tai = 0;
  let total = 0;
  let bestDistance = Infinity;

  // Duyệt những cầu đã hoàn thành trong lịch sử.
  // Không sử dụng kết quả hiện tại làm kết quả tương lai.
  let start = 0;

  while (start < n - 1) {
    let end = start;

    while (
      end + 1 < n &&
      seq[end + 1] === seq[start]
    ) {
      end++;
    }

    // Chỉ sử dụng cầu đã có kết quả tiếp theo.
    if (end < n - 1) {
      const historicalRunLength = end - start + 1;
      const distance = Math.abs(
        historicalRunLength - runLength
      );

      if (distance <= 2) {
        const next = seq[end + 1];

        if (distance < bestDistance) {
          bestDistance = distance;
          tai = 0;
          total = 0;
        }

        if (distance === bestDistance) {
          total++;

          if (next === 1) tai++;
        }
      }
    }

    start = end + 1;
  }

  if (total === 0) {
    return makePrediction(0.5);
  }

  return makePrediction(
    probability(tai, total),
    total
  );
}

// --------------------------------------------------
// 4. PATTERN MATCHING
// Phân tích cầu xen kẽ, đảo cầu và mẫu lặp.
// --------------------------------------------------
function algoPattern(seq) {
  const n = seq.length;
  const maxOrder = Math.min(8, n - 1);

  let taiScore = 0;
  let totalScore = 0;
  let support = 0;

  for (let order = 2; order <= maxOrder; order++) {
    const pattern = seq.slice(n - order).join('');

    for (let i = 0; i + order < n; i++) {
      const candidate = seq.slice(i, i + order).join('');

      if (candidate !== pattern) continue;

      const next = seq[i + order];

      if (next !== 0 && next !== 1) continue;

      // Chuỗi dài hơn có trọng số cao hơn.
      const weight = order / (order + 2);

      taiScore += next * weight;
      totalScore += weight;

      support++;
    }
  }

  if (totalScore === 0) {
    return makePrediction(0.5);
  }

  return makePrediction(
    probability(taiScore, totalScore, 0.5),
    support
  );
}

// --------------------------------------------------
// 5. EWMA TRANSITION ANALYSIS
// Phân tích xu hướng gần đây bằng trọng số thời gian.
// --------------------------------------------------
function algoEWMA(seq) {
  const n = seq.length;

  if (n < 2) return makePrediction(0.5);

  const current = seq[n - 1];
  const alpha = 0.25;

  let taiWeight = 0;
  let totalWeight = 0;

  let weight = 1;

  for (let i = n - 1; i >= 1; i--) {
    if (seq[i - 1] === current) {
      taiWeight += seq[i] * weight;
      totalWeight += weight;
    }

    weight *= (1 - alpha);
  }

  if (totalWeight === 0) {
    return makePrediction(0.5);
  }

  return makePrediction(
    probability(taiWeight, totalWeight, 0.5),
    totalWeight
  );
}

// ============ 4. ENSEMBLE + WALK-FORWARD BACKTEST ============

const ALGOS = [
  { name: 'ngram_markov', fn: algoNGram },
  { name: 'higher_markov', fn: algoHigherMarkov },
  { name: 'run_length', fn: algoRunLength },
  { name: 'pattern_matching', fn: algoPattern },
  { name: 'ewma_transition', fn: algoEWMA }
];

// Trọng số thuật toán dựa trên kết quả kiểm nghiệm.
function calculateAlgoWeight(correct, total) {
  const accuracy = (correct + 2) / (total + 4);

  return 0.25 + Math.max(0, accuracy - 0.5) * 2;
}

// Tổng hợp xác suất của các thuật toán.
function combinePredictions(predictions, weights) {
  let weightedTai = 0;
  let totalWeight = 0;

  for (let i = 0; i < predictions.length; i++) {
    weightedTai += predictions[i].pTai * weights[i];
    totalWeight += weights[i];
  }

  if (totalWeight === 0) return 0.5;

  return weightedTai / totalWeight;
}

// Thuật toán Ensemble chính.
function ensemblePredict(seq) {
  const n = seq.length;

  if (n === 0) {
    return {
      pick: 0,
      confidence: 50
    };
  }

  // Lưu các dự đoán quá khứ theo từng thời điểm.
  const records = [];

  // Walk-forward backtesting.
  // Mỗi lần dự đoán chỉ được sử dụng dữ liệu quá khứ.
  for (let t = 20; t < n; t++) {
    const training = seq.slice(0, t);
    const actual = seq[t];

    const predictions = ALGOS.map(algo => {
      return algo.fn(training);
    });

    // Tính trọng số từ những lần dự đoán trước đó.
    const weights = ALGOS.map((algo, index) => {
      let correct = 0;
      let total = 0;

      for (const record of records) {
        if (record.predictions[index].pick === record.actual) {
          correct++;
        }

        total++;
      }

      return calculateAlgoWeight(correct, total);
    });

    const pTai = combinePredictions(
      predictions,
      weights
    );

    const pick = pTai >= 0.5 ? 1 : 0;

    records.push({
      actual,
      pick,
      predictions
    });
  }

  // Thống kê hiệu suất từng thuật toán.
  const algoStats = ALGOS.map((algo, index) => {
    let correct = 0;
    let total = 0;

    for (const record of records) {
      if (record.predictions[index].pick === record.actual) {
        correct++;
      }

      total++;
    }

    return {
      correct,
      total
    };
  });

  // Trọng số Ensemble hiện tại.
  const weights = algoStats.map(stat => {
    return calculateAlgoWeight(
      stat.correct,
      stat.total
    );
  });

  // Phân tích chuỗi hiện tại.
  const currentPredictions = ALGOS.map(algo => {
    return algo.fn(seq);
  });

  const pTai = combinePredictions(
    currentPredictions,
    weights
  );

  const pick = pTai >= 0.5 ? 1 : 0;

  // Backtesting riêng cho cửa đang được dự đoán.
  let correct = 0;
  let total = 0;

  for (const record of records) {
    if (record.pick !== pick) continue;

    total++;

    if (record.actual === record.pick) {
      correct++;
    }
  }

  // Độ tin cậy có làm trơn Bayesian.
  // Tránh trường hợp ít mẫu nhưng hiển thị 100%.
  const confidence = total > 0
    ? ((correct + 1) / (total + 2)) * 100
    : 50;

  return {
    pick,

    confidence: Math.round(
      confidence * 100
    ) / 100,

    pTai: Math.round(
      pTai * 10000
    ) / 10000,

    totalBacktest: records.length,

    correctBacktest: records.filter(
      r => r.pick === r.actual
    ).length,

    algorithms: ALGOS.map((algo, index) => ({
      name: algo.name,

      pick: currentPredictions[index].pick,

      probabilityTai: Math.round(
        currentPredictions[index].pTai * 10000
      ) / 10000,

      accuracy: algoStats[index].total
        ? Math.round(
            algoStats[index].correct /
            algoStats[index].total * 10000
          ) / 100
        : 0,

      samples: algoStats[index].total
    }))
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

app.get('/', (req, res) => res.json(buildResponse()));

app.get('/predict', (req, res) => res.json(buildResponse()));

app.get('/ping', (req, res) => res.send('pong'));

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
