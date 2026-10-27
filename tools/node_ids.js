// node_ids.js —— 供 check_node_parity.py 调用的 Node 前端 ids 输出
const { TTS } = require(process.argv[4] + '/index.js');
const t = new TTS(process.argv[2]);
console.log(JSON.stringify(t.textToIds(process.argv[3])));
