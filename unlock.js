'use strict';
(async()=>{
  const form=document.getElementById('access-form'),input=document.getElementById('access-password'),status=document.getElementById('access-status'),button=document.getElementById('unlock');
  const base=new URL(document.baseURI),sessionName='tianshui-report-key-v1';
  const bytes=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
  const encode=b=>btoa(String.fromCharCode(...new Uint8Array(b)));
  document.getElementById('show-password').onclick=e=>{const show=input.type==='password';input.type=show?'text':'password';e.currentTarget.textContent=show?'隐藏':'显示';e.currentTarget.setAttribute('aria-label',show?'隐藏密码':'显示密码');e.currentTarget.setAttribute('aria-pressed',String(show));};
  if(!window.crypto?.subtle||!('serviceWorker' in navigator)){status.textContent='请使用支持加密访问的新版浏览器，并通过 HTTPS 打开网页。';button.disabled=true;return;}
  async function activate(raw){
    await navigator.serviceWorker.register(new URL('sw.js',base),{scope:base.pathname,updateViaCache:'none'});
    const registration=await navigator.serviceWorker.ready;
    const worker=registration.active;
    await new Promise((resolve,reject)=>{const channel=new MessageChannel();const timer=setTimeout(()=>reject(new Error('activation')),15000);channel.port1.onmessage=e=>{clearTimeout(timer);e.data.ok?resolve():reject(new Error('key'));};worker.postMessage({type:'UNLOCK',key:raw},[channel.port2]);});
    sessionStorage.setItem(sessionName,raw);
    input.value='';
    // First load has no controller until clients.claim; wait for it before navigating.
    if(!navigator.serviceWorker.controller)await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('controller')),10000);navigator.serviceWorker.addEventListener('controllerchange',()=>{clearTimeout(timer);resolve();},{once:true});});
    location.reload();
  }
  form.addEventListener('submit',async e=>{
    e.preventDefault();button.disabled=true;status.textContent='正在解锁…';
    let valid=false;
    try{
      const response=await fetch(new URL('manifest.json',base),{cache:'no-store'});if(!response.ok)throw new Error('network');const manifest=await response.json();
      const material=await crypto.subtle.importKey('raw',new TextEncoder().encode(input.value),'PBKDF2',false,['deriveBits']);
      const raw=await crypto.subtle.deriveBits({name:'PBKDF2',salt:bytes(manifest.salt),iterations:manifest.iterations,hash:'SHA-256'},material,256);
      const key=await crypto.subtle.importKey('raw',raw,'AES-GCM',false,['decrypt']);const file=manifest.files['index.html'];
      const encrypted=await fetch(new URL(file.file,base),{cache:'no-store'});if(!encrypted.ok)throw new Error('network');
      await crypto.subtle.decrypt({name:'AES-GCM',iv:bytes(file.iv),additionalData:new TextEncoder().encode('index.html')},key,await encrypted.arrayBuffer());
      valid=true;await activate(encode(raw));
    }catch(error){status.textContent=valid?'访问组件加载失败，请刷新后重试。':error.name==='OperationError'?'密码不正确，请重新输入。':'页面暂时无法加载，请检查网络后重试。';input.focus();input.select();button.disabled=false;}
  });
})();
