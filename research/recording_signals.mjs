import { readFileSync, writeFileSync } from 'node:fs';
import { analyzeCove } from '../frontend/lib/cove.ts';
import { corticalRecording } from '../frontend/lib/cortical-recording.ts';

const photo=Buffer.from(JSON.parse(readFileSync(new URL('../frontend/lib/neighborhood-rgb.json',import.meta.url),'utf8')),'base64');
const other=readFileSync(process.argv[2]);
const inputs=[photo,photo,Buffer.alloc(photo.length,128),other];
const outputs=inputs.map((image,i)=>{
  const {rows}=analyzeCove([image],24,[2,2],true,16);
  const {field}=corticalRecording(rows,24,i===1?1732:1731);
  return Buffer.from(field.buffer);
});
writeFileSync(process.argv[3],Buffer.concat(outputs));
