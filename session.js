'use strict';
(()=>{
 const storageName='tianshui-report-key-v1';
 const scope=new URL('/tianshui-retaining-clip-report/',location.origin);
 navigator.serviceWorker.addEventListener('message',event=>{
  if(!event.source?.scriptURL||event.source.scriptURL!==new URL('sw.js',scope).href)return;
  if(event.data?.type==='REQUEST_KEY')event.ports[0]?.postMessage({key:sessionStorage.getItem(storageName)});
  if(event.data?.type==='LOCKED'){sessionStorage.removeItem(storageName);location.replace(scope.href);}
 });
 function addLock(){const box=document.querySelector('.top-actions')||document.body;const button=document.createElement('button');button.type='button';button.className='quiet-button';button.textContent='锁定报告';button.id='lock-report';button.addEventListener('click',()=>{sessionStorage.removeItem(storageName);navigator.serviceWorker.controller?.postMessage({type:'LOCK'});location.replace(scope.href);});box.append(button);}
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',addLock,{once:true});else addLock();
})();
