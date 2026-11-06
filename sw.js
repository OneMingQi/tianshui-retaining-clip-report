'use strict';
// Build 78dfab89aff9d69a. Decrypted files are held in memory only, never CacheStorage.
const BASE=self.registration.scope,STORAGE_NAME='tianshui-report-key-v1';
let key=null,keyRequest=null,manifestRequest=null;
const plain=new Map(),bytes=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const manifest=()=>manifestRequest||(manifestRequest=fetch(new URL('manifest.json',BASE),{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error('manifest');return r.json();}));
async function decrypt(path,withKey){const m=await manifest(),file=m.files[path];if(!file)return null;const response=await fetch(new URL(file.file,BASE),{cache:'no-store'});if(!response.ok)throw new Error('ciphertext');return crypto.subtle.decrypt({name:'AES-GCM',iv:bytes(file.iv),additionalData:new TextEncoder().encode(path)},withKey,await response.arrayBuffer());}
async function accept(raw){const candidate=await crypto.subtle.importKey('raw',bytes(raw),'AES-GCM',false,['decrypt']);await decrypt('index.html',candidate);key=candidate;plain.clear();return key;}
async function getKey(){
 if(key)return key;
 if(keyRequest)return keyRequest;
 keyRequest=(async()=>{const windows=await self.clients.matchAll({type:'window',includeUncontrolled:true});for(const client of windows){if(!client.url.startsWith(BASE))continue;const raw=await new Promise(resolve=>{const channel=new MessageChannel();const timer=setTimeout(()=>resolve(null),1000);channel.port1.onmessage=e=>{clearTimeout(timer);resolve(e.data?.key);};client.postMessage({type:'REQUEST_KEY'},[channel.port2]);});if(raw){try{return await accept(raw);}catch{}}}return null;})().finally(()=>{keyRequest=null;});
 return keyRequest;
}
self.addEventListener('install',event=>event.waitUntil(self.skipWaiting()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('message',event=>{
 if(event.data?.type==='UNLOCK')event.waitUntil(accept(event.data.key).then(()=>event.ports[0]?.postMessage({ok:true})).catch(()=>event.ports[0]?.postMessage({ok:false})));
 if(event.data?.type==='LOCK')event.waitUntil((async()=>{key=null;keyRequest=null;plain.clear();for(const client of await self.clients.matchAll({type:'window',includeUncontrolled:true}))if(client.url.startsWith(BASE))client.postMessage({type:'LOCKED'});})());
});
const publicFiles=new Set(['sw.js','unlock.js','unlock.css','session.js','manifest.json']);
async function respond(request,path){
 const m=await manifest();if(!m.files[path])return fetch(request);
 const activeKey=await getKey();
 if(!activeKey){if(request.mode==='navigate')return fetch(new URL('index.html',BASE),{cache:'no-store'});return new Response('请先解锁报告',{status:401,headers:{'Content-Type':'text/plain;charset=utf-8','Cache-Control':'no-store'}});}
 try{
  let data=plain.get(path);if(!data){data=await decrypt(path,activeKey);plain.set(path,data);}
  const headers=new Headers({'Content-Type':m.files[path].mime,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'strict-origin-when-cross-origin'});
  if(m.files[path].mime.startsWith('text/html')){const html=new TextDecoder().decode(data).replace(/<head([^>]*)>/i,`<head$1><script src="${new URL('session.js',BASE).href}"></script>`);return new Response(request.method==='HEAD'?null:html,{headers});}
  const range=request.headers.get('range');
  if(range){const match=/^bytes=(\d*)-(\d*)$/.exec(range);if(!match)return new Response(null,{status:416});let start=match[1]?Number(match[1]):Math.max(0,data.byteLength-Number(match[2]));let end=match[1]&&match[2]?Math.min(Number(match[2]),data.byteLength-1):data.byteLength-1;if(start>end||start>=data.byteLength)return new Response(null,{status:416,headers:{'Content-Range':`bytes */${data.byteLength}`}});headers.set('Content-Range',`bytes ${start}-${end}/${data.byteLength}`);headers.set('Accept-Ranges','bytes');headers.set('Content-Length',String(end-start+1));return new Response(request.method==='HEAD'?null:data.slice(start,end+1),{status:206,headers});}
  headers.set('Content-Length',String(data.byteLength));return new Response(request.method==='HEAD'?null:data,{headers});
 }catch{plain.delete(path);return new Response('内容无法解密，请重新解锁报告。',{status:503,headers:{'Content-Type':'text/plain;charset=utf-8','Cache-Control':'no-store'}});}
}
self.addEventListener('fetch',event=>{
 const url=new URL(event.request.url);if(!url.href.startsWith(BASE)||!['GET','HEAD'].includes(event.request.method))return;
 const path=decodeURIComponent(url.pathname.slice(new URL(BASE).pathname.length))||'index.html';
 if(publicFiles.has(path)||path.startsWith('encrypted/'))return;
 event.respondWith(respond(event.request,path));
});
