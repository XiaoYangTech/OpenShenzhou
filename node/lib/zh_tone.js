'use strict';
// zh_tone.js —— 中文上下文声调/读音修正（与 Python openshenzhou/zh_tone.py 一致）
//
// 数据：data/zh_tone_ctx_model.json.gz（8,756 句 UI 语料引擎 ids 训练的字级上下文修正表，
//       5 级 backoff；只保留 count>=5、ratio>=0.90 且与基线不同的覆盖项）。
// 验证：留出 1500 句 1419->1448；中文 3000 句 2915->2923；599 句不变。
// 只对简体路径生效（繁体由 zh_trad 处理）。

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const DATA = path.join(__dirname, '..', '..', 'data');
const BOUND = new Set(['-bos-', 'br0', 'br1', 'br2', 'br3', 'br4', '~', '-']);

let MODEL = null;
let TABLES = null;
let ENABLED = true;

function setEnabled(flag) { ENABLED = !!flag; }
function isEnabled() { return ENABLED; }

function load() {
  if (MODEL !== null) return;
  try {
    const buf = fs.readFileSync(path.join(DATA, 'zh_tone_ctx_model.json.gz'));
    MODEL = JSON.parse(zlib.gunzipSync(buf).toString('utf8'));
  } catch (e) {
    MODEL = { tables: [] };
  }
  TABLES = MODEL.tables || [];
}

function revOf(sym) {
  const rev = {};
  for (const [k, v] of Object.entries(sym || {})) rev[v] = k;
  return rev;
}

function charsOf(sent) {
  const out = [];
  for (const ch of sent) {
    const code = ch.codePointAt(0);
    if ((code >= 0x4e00 && code <= 0x9fff) || ch === '〇') out.push(ch);
  }
  return out;
}

function lookup(chars, i, cur) {
  const n = chars.length;
  const ch = chars[i];
  const p1 = i > 0 ? chars[i - 1] : '^';
  const p2 = i > 1 ? chars[i - 2] : '^';
  const n1 = i + 1 < n ? chars[i + 1] : '$';
  const curS = cur.join(' ');
  const keys = [curS + '\x1f' + p2 + p1 + ch + n1,
    curS + '\x1f' + p1 + ch + n1,
    curS + '\x1f' + ch + n1,
    curS + '\x1f' + ch,
    curS];
  for (let d = 0; d < keys.length; d++) {
    const table = TABLES[d];
    if (!table) continue;
    const hit = table[keys[d]];
    if (hit) return hit.slice();
  }
  return null;
}

function apply(text, ids, sym) {
  load();
  if (!ENABLED || !TABLES.length) return ids;
  try {
    const trad = require('./zh_trad');
    if (trad.detect(text)) return ids;
  } catch (e) { /* 保持可用 */ }
  const rev = revOf(sym);
  const names = [];
  for (const x of ids) {
    names.push(rev[x] !== undefined ? rev[x] : ('?' + x));
    if (x === 1) break;
  }
  const chars = charsOf(text);
  const pieces = [];
  let cur = [];
  for (const t of names) {
    if (BOUND.has(t) || t.startsWith('punc')) {
      if (cur.length) { pieces.push(['word', cur]); cur = []; }
      pieces.push(['sep', t]);
    } else {
      cur.push(t);
    }
  }
  if (cur.length) pieces.push(['word', cur]);
  const words = pieces.filter((p) => p[0] === 'word').map((p) => p[1]);
  if (words.length !== chars.length) return ids;
  const out = [];
  let changed = false;
  let wi = 0;
  for (const piece of pieces) {
    if (piece[0] === 'sep') {
      out.push(sym[piece[1]] !== undefined ? sym[piece[1]] : (sym.padding || 0));
      continue;
    }
    const i = wi;
    wi += 1;
    let w = piece[1];
    const pred = lookup(chars, i, w);
    if (pred && !(pred.length === w.length && pred.every((x, k) => x === w[k]))) {
      let ok = true;
      for (const t of pred) if (sym[t] === undefined) { ok = false; break; }
      if (ok) { w = pred; changed = true; }
    }
    for (const t of w) out.push(sym[t]);
  }
  return changed ? out : ids;
}

module.exports = { apply, setEnabled, isEnabled, load, detect: () => false };
