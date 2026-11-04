'use strict';
// node_zh_trad_ids.js —— 用 Node SDK 输出繁体句子 ids（供双端对照）
// 用法: node node_zh_trad_ids.js <voiceDir> <sentences.json>
const fs = require('fs');
const path = require('path');

const nodeRoot = process.env.OS_NODE_ROOT || path.join(__dirname, '..', '..', 'work-node');
const { TTS } = require(path.join(nodeRoot, 'node', 'index.js'));

const voiceDir = process.argv[2];
const sentPath = process.argv[3];
const sents = JSON.parse(fs.readFileSync(sentPath, 'utf8'));
const tts = new TTS(voiceDir);
const out = sents.map((s) => tts.textToIds(s));
process.stdout.write(JSON.stringify(out));
