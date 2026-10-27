'use strict';
// node_zh_ids_batch.js —— 批量导出 Node SDK 中文前端 ids（供 Python/Node 一致性检查）
// 用法: node node_zh_ids_batch.js <node-root> <voice-package> <corpus.jsonl> <out.json>
const fs = require('fs');
const path = require('path');

const [root, voice, dataFile, outFile] = process.argv.slice(2);
const { TTS } = require(path.join(root, 'node', 'index.js'));
const rows = fs.readFileSync(dataFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const tts = new TTS(voice);
const out = rows.map((r) => ({ text: r.text, ids: tts.textToIds(r.text) }));
fs.writeFileSync(outFile, JSON.stringify(out));
console.log('dumped', out.length);
