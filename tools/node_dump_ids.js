'use strict';
// node_dump_ids.js —— 批量导出 Node SDK 的音素 ID（用于 Python/Node 一致性检查）
// 用法: node node_dump_ids.js <node-root> <voice-package> <corpus.jsonl> <out.json>
const fs = require('fs');
const path = require('path');

const [root, voice, dataFile, outFile] = process.argv.slice(2);
const { TTS } = require(path.join(root, 'node', 'index.js'));
const rows = fs.readFileSync(dataFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const tts = new TTS(voice);
const out = rows.map((r) => ({ text: r.text, node: tts.textToIds(r.text) }));
fs.writeFileSync(outFile, JSON.stringify(out));
console.log('dumped', out.length);
