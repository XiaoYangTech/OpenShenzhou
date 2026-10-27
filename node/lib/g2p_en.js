'use strict';
// g2p_en.js —— 英文文本 → 音素 ID（引擎标定词表 + CMUdict）
const fs = require('fs');
const { resolveSeq } = require('./symbols');
const path = require('path');

const DATA = path.join(__dirname, '..', '..', 'data');
const CMU = JSON.parse(fs.readFileSync(path.join(DATA, 'cmudict_en.json'), 'utf8'));
const LEXICON = (() => {
  try {
    const p = path.join(DATA, 'lexicon_en.names.json');
    const f = fs.existsSync(p) ? p : path.join(DATA, 'lexicon_en.json');
    return JSON.parse(fs.readFileSync(f, 'utf8'));
  } catch (e) { return {}; }
})();

const PUNCT = { ',': 'punc,', '.': 'punc.', '?': 'punc?', '!': 'punc!', ';': 'punc;', '…': 'punc…' };
const LETTER_RULES = {
  a: ['AE1'], b: ['B'], c: ['K'], d: ['D'], e: ['EH1'], f: ['F'], g: ['G'], h: ['HH'],
  i: ['IH1'], j: ['JH'], k: ['K'], l: ['L'], m: ['M'], n: ['N'], o: ['OW1'], p: ['P'],
  q: ['K'], r: ['R'], s: ['S'], t: ['T'], u: ['AH1'], v: ['V'], w: ['W'], x: ['K', 'S'],
  y: ['Y'], z: ['Z'],
};

function cmuToEngine(ph) {
  const m = ph.match(/^([A-Z]+)([012])?$/);
  if (!m) return ph.toLowerCase();
  const base = m[1], stress = m[2];
  if (base === 'HH') return 'h';
  if (base === 'AH' && (!stress || stress === '0')) return 'ax';
  if (stress === '1' || stress === '2') return base.toLowerCase() + stress;
  return base.toLowerCase();
}

const EN_WEAK = { a: ['ax'], an: ['ax', 'n'], the: ['dh', 'ax1'],
  for: ['f', 'ax1', 'r'], of: ['ax', 'v'], to: ['t', 'ax'],
  and: ['ax', 'n', 'd'], are: ['aa1', 'r'] };
const VOWEL_LETTERS = new Set(['a', 'e', 'i', 'o', 'u']);

function wordPhones(word) {
  const w = word.toLowerCase();
  const cmu = CMU[w];
  if (cmu) return cmu.split(' ').map(cmuToEngine);
  const out = [];
  for (const ch of w) for (const p of (LETTER_RULES[ch] || [])) out.push(cmuToEngine(p));
  return out;
}

const VOWELS = new Set('aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw'.split(' '));
const ONSETS = new Set(`
p b t d k g f v th dh s z sh zh ch jh m n ng l r h w y
pl pr bl br tr dr kl kr gl gr fl fr th r sl sw sp st sk sm sn
spl spr str skr skw shr thr
`.split(/\s+/).filter(Boolean));

function splitSyllables(phones) {
  const n = phones.length;
  const isVowel = (p) => VOWELS.has(p.replace(/[012]$/, ''));
  const nuclei = [];
  for (let i = 0; i < n; i++) if (isVowel(phones[i])) nuclei.push(i);
  if (!nuclei.length) return [phones.slice()];
  const cuts = [];
  for (let k = 0; k + 1 < nuclei.length; k++) {
    const a = nuclei[k], b = nuclei[k + 1];
    const cluster = phones.slice(a + 1, b);
    if (!cluster.length) { cuts.push(b); continue; }
    let cut = a + 1;
    for (let m = cluster.length; m > 0; m--) {
      const cand = cluster.slice(cluster.length - m).map((x) => x.replace(/[012]$/, '')).join('');
      if (ONSETS.has(cand)) { cut = a + 1 + (cluster.length - m); break; }
    }
    cuts.push(cut);
  }
  const groups = [];
  let prev = 0;
  for (const c of cuts) {
    if (c <= prev) continue;
    groups.push(phones.slice(prev, c));
    prev = c;
  }
  groups.push(phones.slice(prev));
  return groups.filter((g) => g.length);
}

function textToIds(text, sym) {
  const ids = [sym['-bos-'], sym['br0']];
  let first = true, ended = false;
  const parts = text.split(/([,\.\?!;…])/);
  for (const chunk of parts) {
    if (!chunk || !chunk.trim()) continue;
    if (PUNCT[chunk]) {
      const key = PUNCT[chunk];
      if (['.', '?', '!'].includes(chunk)) {
        ids.push(sym['br4']);
        ids.push(sym[key] !== undefined ? sym[key] : 0);
        ended = true;
      } else {
        ids.push(sym['br3'] !== undefined ? sym['br3'] : 0);
        ids.push(sym[key] !== undefined ? sym[key] : 0);
        ids.push(sym['br0']);
        first = true;
      }
      continue;
    }
    const words = chunk.match(/[A-Za-z0-9']+/g) || [];
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi];
      let weak = EN_WEAK[w.toLowerCase()];
      const nxt = words[wi + 1] || '';
      if (w.toLowerCase() === 'the' && nxt && VOWEL_LETTERS.has(nxt[0].toLowerCase())) weak = ['dh', 'ih1'];
      if (w.toLowerCase() === 'a' && nxt && VOWEL_LETTERS.has(nxt[0].toLowerCase())) weak = ['ax', 'n'];
      let core = weak ? weak.map((x) => (sym[x] !== undefined ? sym[x] : (sym['<pad>'] || 0)))
                      : LEXICON[w.toLowerCase()];
      if (core) core = resolveSeq(core, sym);
      if (!core) {
        const groups = splitSyllables(wordPhones(w));
        core = [];
        groups.forEach((g, gi) => {
          if (gi > 0 && sym['-'] !== undefined) core.push(sym['-']);
          for (const p of g) core.push(sym[p] !== undefined ? sym[p] : (sym['<pad>'] || 0));
        });
      }
      if (!first) ids.push(sym['br1'] !== undefined ? sym['br1'] : sym['br0']);
      for (const id of core) ids.push(id);
      first = false;
    }
  }
  if (!ended) ids.push(sym['br4']);
  ids.push(sym['~']);
  return ids;
}

module.exports = { textToIds, wordPhones, cmuToEngine };
