const express = require('express');
const fs = require('node:fs');
const path = require('node:path');

const app = express();

const DATA_FILE = path.join(__dirname, 'lichsulc79.json');
const LEGACY_FILE = path.join(__dirname, 'lichsulc79md5.json');

const SOURCE_API = process.env.SOURCE_API ||
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const PORT = process.env.PORT || 3000;
const POLL_INTERVAL = 15000;

app.use(express.json());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// Chuẩn hóa dữ liệu phiên

function integer(value) {
  const isNumericString =
    typeof value === 'string' && /^\d+$/.test(value.trim());

  if (typeof value !== 'number' && !isNumericString) return null;

  const n = Number(value);
  return Number.isSafeInteger(n) ? n : null;
}

function label(value) {
  const s = String(value ?? '')
    .trim()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toUpperCase();

  if (s === 'TAI' || s === 'BIG') return 1;
  if (s === 'XIU' || s === 'SMALL') return 0;
  return null;
}

function normalize(raw, fromDisk = false) {
  if (!raw || typeof raw !== 'object') return null;

  const id = integer(raw.id ?? raw.session ?? raw.issue);
  if (id === null || id < 1 || id >= Number.MAX_SAFE_INTEGER) {
    return null;
  }

  const faces = raw.dices ?? raw.dice ?? raw.result;
  let dices = [];

  if (Array.isArray(faces) && faces.length === 3) {
    const values = faces.map(integer);
    if (values.every(v => v >= 1 && v <= 6)) {
      dices = values;
    }
  }

  const total = integer(raw.sum);
  let sum = null;

  if (dices.length === 3) {
    sum = dices.reduce((a, b) => a + b, 0);
  } else if (total >= 3 && total <= 18) {
    sum = total;
  }

  let result =
    label(raw.resultTruyenThong) ??
    label(raw.bigSmall) ??
    label(raw.result);

  // Trong file lịch sử, 1 là Tài và 0 là Xỉu.
  if (fromDisk && (raw.result === 0 || raw.result === 1)) {
    result = raw.result;
  }

  // Giữ cách phân loại theo tổng của API cũ khi thiếu nhãn.
  if (result === null && sum !== null) {
    result = Number(sum >= 11);
  }

  // Phiên chưa có kết quả sẽ được xét lại trong lần đồng bộ sau.
  if (result === null) return null;

  return {
    id,
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// Mô hình dự đoán xác định, không dùng random hoặc MD5

class Predictor {
  constructor() {
    // Bậc -1 là mốc 50/50. Bậc 0 học tần suất chung.
    // Bậc 1–3 học tần suất sau 1–3 kết quả liên tiếp.
    // Ban đầu, mốc 50/50 nhận một nửa tổng trọng số.
    this.models = [-1, 0, 1, 2, 3].map(order => ({
      order,
      counts: new Map(),
      logWeight: order < 0 ? Math.log(4) : 0
    }));

    this.tail = '';
    this.lastId = null;
    this.scored = 0;
    this.hits = 0;
    this.brier = 0;
  }

  context(order) {
    if (order < 0 || this.tail.length < order) return null;
    return order === 0 ? '' : this.tail.slice(-order);
  }

  estimate() {
    let weighted = 0;
    let totalWeight = 0;

    const parts = this.models.map(model => {
      const key = this.context(model.order);
      const counts = model.counts.get(key) ?? [0, 0];

      // Làm trơn để một mẫu ngắn chưa thể cho xác suất 0% hoặc 100%.
      const p = (counts[1] + 0.5) / (counts[0] + counts[1] + 1);
      const weight = Math.exp(model.logWeight);

      weighted += weight * p;
      totalWeight += weight;

      return { model, key, p };
    });

    return { p: weighted / totalWeight, parts };
  }

  observe(row) {
    const consecutive =
      this.lastId !== null && row.id === this.lastId + 1;

    // Khi thiếu phiên, bắt đầu lại chuỗi ngữ cảnh tại vị trí mới.
    if (!consecutive) this.tail = '';

    // Lấy dự đoán trước khi học kết quả thật của phiên này.
    const { p, parts } = this.estimate();
    const y = row.result;

    if (consecutive) {
      this.scored++;
      if ((p >= 0.5) === (y === 1)) this.hits++;
      this.brier += (p - y) ** 2;
    }

    for (const { model, key, p: q } of parts) {
      // Trọng số thay đổi theo xác suất từng mô hình đã gán
      // cho kết quả thực tế trước khi nhìn thấy kết quả đó.
      model.logWeight += y ? Math.log(q) : Math.log1p(-q);

      if (key === null) continue;

      const counts = model.counts.get(key) ?? [0, 0];
      counts[y]++;
      model.counts.set(key, counts);
    }

    // Giữ phép tính trọng số ổn định khi lịch sử dài.
    const peak = Math.max(...this.models.map(m => m.logWeight));
    for (const model of this.models) {
      model.logWeight -= peak;
    }

    this.tail = (this.tail + y).slice(-3);
    this.lastId = row.id;
  }
}

// Lưu lịch sử và khôi phục khi khởi động

function save(rows) {
  const temp = DATA_FILE + '.' + process.pid + '.tmp';
  fs.writeFileSync(temp, JSON.stringify(rows), 'utf8');
  fs.renameSync(temp, DATA_FILE);
}

function load() {
  const migrate = !fs.existsSync(DATA_FILE);

  if (migrate && !fs.existsSync(LEGACY_FILE)) {
    save([]);
    return [];
  }

  const file = migrate ? LEGACY_FILE : DATA_FILE;
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));

  if (!Array.isArray(raw)) {
    throw new Error('Lịch sử phải là mảng JSON: ' + file);
  }

  const byId = new Map();
  let skipped = 0;

  for (const item of raw) {
    const row = normalize(item, true);

    // Mã cũ có thể ghi nhầm phiên thiếu dữ liệu thành Xỉu, tổng 0.
    if (!row || (migrate && row.sum === null)) {
      if (!migrate) {
        throw new Error('Bản ghi không hợp lệ trong ' + file);
      }
      skipped++;
      continue;
    }

    byId.set(row.id, row);
  }

  const rows = [...byId.values()].sort((a, b) => a.id - b.id);

  if (migrate) {
    save(rows);
    console.log(
      '[MIGRATE]', rows.length, 'phiên hợp lệ, bỏ qua',
      skipped, 'bản ghi thiếu dữ liệu'
    );
  }

  return rows;
}

let history = load();
const known = new Set(history.map(row => row.id));

let model = new Predictor();
for (const row of history) {
  model.observe(row);
}

// Đồng bộ nguồn API

async function sync() {
  try {
    const res = await fetch(SOURCE_API, {
      headers: {
        Accept: 'application/json',
        'User-Agent': 'Mozilla/5.0',
        Referer: 'https://tele68.com/'
      },
      signal: AbortSignal.timeout(10000)
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);

    const body = await res.json();
    const list = [
      body,
      body?.data,
      body?.sessions,
      body?.list,
      body?.data?.sessions,
      body?.data?.list
    ].find(Array.isArray);

    if (!list) throw new Error('API chưa trả về danh sách phiên');

    const batch = new Map();

    for (const raw of list) {
      const row = normalize(raw);

      if (row && !known.has(row.id) && !batch.has(row.id)) {
        batch.set(row.id, row);
      }
    }

    if (batch.size === 0) return;

    const added = [...batch.values()].sort((a, b) => a.id - b.id);
    const backfill =
      history.length > 0 && added[0].id < history.at(-1).id;

    const next = history.concat(added);
    if (backfill) next.sort((a, b) => a.id - b.id);

    // Nếu ghi lỗi, giữ trạng thái cũ để lần đồng bộ sau thử lại.
    save(next);

    // Khi có phiên đến muộn, phát lại lịch sử theo đúng thứ tự.
    if (backfill) model = new Predictor();

    for (const row of backfill ? next : added) {
      model.observe(row);
    }

    for (const row of added) known.add(row.id);
    history = next;
  } catch (err) {
    console.error('[SYNC]', err.message);
  }
}

// API trả kết quả

function respond(req, res) {
  const uptime = new Date(Date.now() + 7 * 3600000)
    .toISOString()
    .replace('Z', '+07:00');

  const row = history.at(-1);

  if (!row) {
    return res.json({
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime
    });
  }

  const { p } = model.estimate();
  const n = model.scored;

  res.json({
    id: 'noname09',
    phienhientai: row.id,
    ketqua: row.result ? 'tai' : 'xiu',
    xucxac: row.dices,
    tong: row.sum,
    dudoanphien: row.id + 1,
    dudoan: p >= 0.5 ? 'tai' : 'xiu',
    dotincay: (100 * Math.max(p, 1 - p)).toFixed(2) + '%',
    kiemtra_lichsu: {
      so_phien: n,
      so_dung: model.hits,
      ty_le_dung: n ? (100 * model.hits / n).toFixed(2) + '%' : null,
      brier: n ? Number((model.brier / n).toFixed(5)) : null
    },
    uptime
  });
}

app.get(['/', '/predict'], respond);

app.get('/ping', (req, res) => {
  res.send('pong');
});

app.get('/history', (req, res) => {
  res.json({
    count: history.length,
    last20: history.slice(-20)
  });
});

// Chờ lần đồng bộ hiện tại kết thúc rồi mới hẹn lần tiếp theo.
async function poll() {
  await sync();
  setTimeout(poll, POLL_INTERVAL).unref();
}

app.listen(PORT, () => {
  console.log('[BOOT] Port', PORT, '| Đã nạp', history.length, 'phiên');
  void poll();
});
