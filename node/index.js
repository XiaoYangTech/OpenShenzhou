'use strict';
// index.js —— OpenShenzhou Node SDK：给语音包目录即可朗读
const path = require('path');
const { VoicePack } = require('./lib/voicepack');
const { synthesize, writeWav } = require('./lib/synth');

function splitSentences(ids, sym) {
  const END = 1;
  const bos = sym ? sym['-bos-'] : null;
  const br0 = sym ? sym['br0'] : null;
  const br4 = sym ? sym['br4'] : null;
  const puncIds = new Set();
  if (sym) {
    for (const [k, v] of Object.entries(sym)) if (k.startsWith('punc')) puncIds.add(v);
    if (sym['punc.'] !== undefined) puncIds.add(sym['punc.']);
  }
  const close = (cur) => {
    if (cur.length && cur[cur.length - 1] === END) {
      const j = cur.length - 2;
      if (j >= 0 && puncIds.has(cur[j]) && (j === 0 || cur[j - 1] !== br4)) cur.splice(j, 0, br4);
    }
    if (!cur.length || cur[cur.length - 1] !== END) cur.push(END);
    return cur;
  };
  const out = [];
  let cur = [];
  let i = 0;
  while (i < ids.length) {
    const x = ids[i];
    cur.push(x);
    i += 1;
    if (x === END) { out.push(close(cur)); cur = []; continue; }
    if (br4 !== null && x === br4) {
      while (i < ids.length && puncIds.has(ids[i])) { cur.push(ids[i]); i += 1; }
      if (i < ids.length && ids[i] !== END) {
        out.push(close(cur)); cur = [];
        if (ids[i] !== bos && bos !== null) cur = [bos, br0];
      }
      continue;
    }
  }
  if (cur.length) out.push(close(cur));
  return out;
}

class TTS {
  /**
   * @param {string} voiceDir 语音包目录（含 .bin 模型与 phones.txt）
   */
  constructor(voiceDir) {
    this.pack = new VoicePack(voiceDir);
    this.lang = this.pack.lang;
    this._g2p = null;
  }

  _g2pFn() {
    if (this._g2p) return this._g2p;
    if (this.lang === 'zh') {
      const { textToIds } = require('./lib/g2p_zh');
      this._g2p = (text) => textToIds(text, this.pack.symbols, this.pack.dir);
    } else if (this.lang === 'en') {
      const { textToIds } = require('./lib/g2p_en');
      this._g2p = (text) => textToIds(text, this.pack.symbols);
    } else if (['es', 'fr', 'de', 'pt'].includes(this.lang)) {
      const { textToIds } = require('./lib/g2p_latin');
      this._g2p = (text) => textToIds(text, this.pack.symbols, this.lang);
    } else if (this.lang === 'ja') {
      const { textToIds } = require('./lib/g2p_ja');
      this._g2p = (text) => textToIds(text, this.pack.symbols);
    } else if (this.lang === 'ko') {
      const { textToIds } = require('./lib/g2p_ko');
      this._g2p = (text) => textToIds(text, this.pack.symbols);
    } else {
      throw new Error('暂不支持的语言: ' + this.lang + '（当前支持 zh / en / es / fr / de / pt / ja / ko）');
    }
    return this._g2p;
  }

  textToIds(text) {
    return this._g2pFn()(text);
  }

  /**
   * @param {string} text 文本
   * @param {string} [outPath] 可选 WAV 输出路径
   * @param {number} [speed] 1.0 正常
   * @returns {Promise<Float32Array>} 24kHz 单声道波形
   */
  async speak(text, outPath, speed = 1.0) {
    const ids = this.textToIds(text);
    const sessions = await this.pack.sessions();
    // 引擎每句一次 encoder：多句序列逐句合成后拼接
    const chunks = splitSentences(ids, this.pack.symbols);
    if (chunks.length <= 1) return synthesize(sessions, ids, outPath, speed);
    const parts = [];
    let total = 0;
    for (const ch of chunks) {
      const a = await synthesize(sessions, ch, null, speed);
      parts.push(a);
      total += a.length;
    }
    const audio = new Float32Array(total);
    let off = 0;
    for (const a of parts) { audio.set(a, off); off += a.length; }
    if (outPath) writeWav(outPath, audio, 24000);
    return audio;
  }
}

module.exports = { TTS };
