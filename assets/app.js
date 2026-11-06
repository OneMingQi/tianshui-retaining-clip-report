'use strict';
const $=id=>document.getElementById(id);
const phaseNames={endpoint_open:'开指',endpoint_withdraw:'回撤',endpoint_verify:'稳定观察',endpoint_failed:'终止',native_endpoint:'原生末态'};
const format=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'—';
$('print').addEventListener('click',()=>window.print());
function progress(){const total=document.documentElement.scrollHeight-innerHeight;$('read-progress').style.width=`${total>0?scrollY/total*100:0}%`;}
window.addEventListener('scroll',progress,{passive:true});progress();
const nav=[...document.querySelectorAll('.sidebar nav a')];
const navTargets=new Set(nav.map(a=>a.getAttribute('href').slice(1)));
const articleSections=[...document.querySelectorAll('main>section[id],main h3[id]')].filter(e=>navTargets.has(e.id));
let navFrame=0;
function updateNav(){const probe=Math.max(130,innerHeight*.22);let current=null;for(const section of articleSections){if(section.getBoundingClientRect().top<=probe)current=section;else break;}nav.forEach(a=>{const on=Boolean(current)&&a.getAttribute('href')==='#'+current.id;a.classList.toggle('active',on);a.classList.toggle('active-parent',Boolean(current)&&current.closest('section')?.id===a.getAttribute('href').slice(1)&&!on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});navFrame=0;}
function queueNav(){if(!navFrame)navFrame=requestAnimationFrame(updateNav);}
window.addEventListener('scroll',queueNav,{passive:true});window.addEventListener('resize',queueNav);updateNav();
function setMedia(el,src,poster){el.pause();el.src=src;el.poster=poster;el.load();}
$('original-size').addEventListener('change',e=>{const n=e.target.value;setMedia($('original-video'),`assets/media/upstream-lab${n}.mp4`,`assets/media/upstream-lab${n}.jpg`);});
$('v2-result').addEventListener('change',e=>{const n=e.target.value;setMedia($('v2-video'),`assets/media/v2-medium-${n}.mp4`,`assets/media/v2-medium-${n}.jpg`);});
async function readJSON(path){const r=await fetch(path);if(!r.ok)throw new Error(`${path}: HTTP ${r.status}`);return r.json();}
function cell(text,tag='td'){const c=document.createElement(tag);c.textContent=text;return c;}
async function ledger(){const report=await readJSON('data/report.json');function render(){const scope=$('ledger-filter').value;const rows=report.ledger.filter(r=>scope==='all'||(scope==='full'?r.n===36:r.n===12));const body=$('ledger-body');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');if(row.id==='c7_progress_timeout')tr.className='best';const id=row.id==='v1'?'V1':row.id==='v2'?'V2':row.id.split('_')[0].toUpperCase();tr.append(cell(id),cell(row.note),cell(`${row.success}/${row.n} · ${format(row.success/row.n*100,1)}%`),cell(`${row.jitter}/${row.n}`),cell(`${row.types.join(' / ')}（每型 n=${row.n/3}）`),cell(row.n===36?'完整开发集':'预定难例初筛'));body.append(tr);}const n=rows.reduce((a,r)=>a+r.n,0);$('ledger-count').textContent=`${rows.length} 个实验臂 / ${n} 次尝试`;}$('ledger-filter').addEventListener('change',render);render();}
const NS='http://www.w3.org/2000/svg';
function svgNode(tag,attrs={},text){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
async function traces(){const data=await readJSON('data/trajectories.json');let current;
let traceFrame=0,playing=false,startTime=0,startDataTime=0;
function stopTrace(){playing=false;cancelAnimationFrame(traceFrame);if($('trace-play')){$('trace-play').textContent='播放轨迹';$('trace-play').setAttribute('aria-pressed','false');}}
function playTrace(){if(!current)return;if(Number($('trace-time').value)>=current.t.length-1)$('trace-time').value=0;playing=true;startTime=performance.now();startDataTime=current.t[Number($('trace-time').value)];$('trace-play').textContent='暂停轨迹';$('trace-play').setAttribute('aria-pressed','true');function tick(now){if(!playing)return;const t=startDataTime+(now-startTime)/1000;let lo=0,hi=current.t.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(current.t[mid]<=t)lo=mid;else hi=mid-1;}$('trace-time').value=lo;cursor();if(t>=current.t.at(-1)){stopTrace();return;}traceFrame=requestAnimationFrame(tick);}traceFrame=requestAnimationFrame(tick);}
const svg=$('trace-svg');const left=78,right=915,width=right-left,top1=38,bottom1=218,top2=284,bottom2=440;let geom;
const colors={z:'#147d80',error:'#c8793f',lateral:'#8455a1'};
function select(){stopTrace();const campaign=$('trace-campaign').value,lab=Number($('trace-size').value);current=data.find(s=>s.campaign===campaign&&s.lab===lab);if(!current){svg.replaceChildren(svgNode('text',{x:40,y:60},'该配置无可用扩展段记录'));$('trace-source').textContent='无可用数据；不以其他案例替代。';return;}$('trace-time').max=current.t.length-1;$('trace-time').value=0;draw();cursor();}
function draw(){const s=current;svg.replaceChildren(svgNode('title',{id:'trace-chart-title'},`V2 ${s.label} / 型号 ${s.lab}，扩展段原始轨迹`));const maxT=s.t.at(-1)||1,maxZ=Math.max(110,Math.max(...s.withdrawal)*1.06),minZ=Math.min(-3,Math.min(...s.withdrawal)*1.06),maxErr=Math.max(3,Math.max(...s.clip_error)*1.12,Math.max(...s.lateral)*1.12);
geom={x:t=>left+t/maxT*width,yz:v=>bottom1-(v-minZ)/(maxZ-minZ)*(bottom1-top1),ye:v=>bottom2-v/maxErr*(bottom2-top2),maxT};
const{x,yz,ye}=geom;
for(let j=0;j<=4;j++){const t=maxT*j/4,xx=x(t);svg.append(svgNode('line',{x1:xx,x2:xx,y1:top1,y2:bottom2,stroke:'#e4ece8','stroke-width':1}));svg.append(svgNode('text',{x:xx,y:465,fill:'#6c8185','font-size':12,'text-anchor':'middle'},`${format(t,1)} s`));}
for(let j=0;j<=4;j++){const z=minZ+(maxZ-minZ)*j/4,yy=yz(z);svg.append(svgNode('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:'#e7edeb'}));svg.append(svgNode('text',{x:left-12,y:yy+4,fill:'#6c8185','font-size':12,'text-anchor':'end'},format(z,0)));const er=maxErr*j/4,ey=ye(er);svg.append(svgNode('line',{x1:left,x2:right,y1:ey,y2:ey,stroke:'#e7edeb'}));svg.append(svgNode('text',{x:left-12,y:ey+4,fill:'#6c8185','font-size':12,'text-anchor':'end'},format(er,1)));}
svg.append(svgNode('text',{x:left,y:19,fill:'#42636b','font-size':13,'font-weight':600},'末端 Z 位移 / mm（相对扩展起点）'));svg.append(svgNode('text',{x:left,y:265,fill:'#42636b','font-size':13,'font-weight':600},'固定夹目标误差、末端横向位移 / mm'));
for(const[val,map,label]of[[100,yz,'100 mm 参照'],[2,ye,'2 mm 位置参照']]){const yy=map(val);svg.append(svgNode('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:val===2?'#ad5d5d':'#7e9697','stroke-dasharray':'5 5','stroke-width':1}));svg.append(svgNode('text',{x:right-3,y:yy-6,fill:'#7a8a8d','font-size':10,'text-anchor':'end'},label));}
for(const p of s.phases){if(p.t===0)continue;const xx=x(p.t);svg.append(svgNode('line',{x1:xx,x2:xx,y1:top1,y2:bottom2,stroke:'#b5c9be','stroke-dasharray':'3 4'}));svg.append(svgNode('text',{x:Math.min(right-65,xx+4),y:top1+14,fill:'#759186','font-size':10},phaseNames[p.phase]||p.phase));}
function path(values,map,color){const stride=Math.max(1,Math.floor(values.length/700));let d='';for(let i=0;i<values.length;i+=stride)d+=(i===0?'M':'L')+format(x(s.t[i]),2)+','+format(map(values[i]),2);if((values.length-1)%stride!==0)d+='L'+format(x(s.t.at(-1)),2)+','+format(map(values.at(-1)),2);svg.append(svgNode('path',{d,fill:'none',stroke:color,'stroke-width':2.2,'stroke-linejoin':'round'}));}
path(s.withdrawal,yz,colors.z);path(s.clip_error,ye,colors.error);path(s.lateral,ye,colors.lateral);
svg.append(svgNode('line',{id:'trace-cursor',x1:left,x2:left,y1:top1,y2:bottom2,stroke:'#16333f','stroke-width':1.2}));for(const[id,c]of[['trace-dot-z',colors.z],['trace-dot-error',colors.error]])svg.append(svgNode('circle',{id,cx:left,cy:0,r:4.5,fill:c,stroke:'#fff','stroke-width':1.5}));
$('trace-source').textContent=`数据来源：${s.source} · ${s.t.length} 帧 · V2 / case 00 · 仅扩展段`;
}
function cursor(){if(!current||!geom)return;const i=Number($('trace-time').value),s=current,{x,yz,ye}=geom,xx=x(s.t[i]);const line=$('trace-cursor');line.setAttribute('x1',xx);line.setAttribute('x2',xx);$('trace-dot-z').setAttribute('cx',xx);$('trace-dot-z').setAttribute('cy',yz(s.withdrawal[i]));$('trace-dot-error').setAttribute('cx',xx);$('trace-dot-error').setAttribute('cy',ye(s.clip_error[i]));$('trace-time-label').textContent=`${format(s.t[i])} s / 帧 ${i}`;
const values=[['阶段',phaseNames[s.phase[i]]||s.phase[i]],['末端 Z 位移',`${format(s.withdrawal[i])} mm`],['固定夹目标误差',`${format(s.clip_error[i])} mm`],['末端横向位移',`${format(s.lateral[i])} mm`],['末端指令跟踪差',`${format(s.tracking[i])} mm`],['原始腕力合力',`${format(s.force[i],1)} N`]];const box=$('trace-readouts');box.replaceChildren();for(const[k,v]of values){const wrap=document.createElement('div'),dl=document.createElement('dl'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;dl.append(dt,dd);wrap.append(dl);box.append(wrap);}
}
$('trace-campaign').addEventListener('change',select);$('trace-size').addEventListener('change',select);$('trace-time').addEventListener('input',()=>{stopTrace();cursor();});
if($('trace-play'))$('trace-play').addEventListener('click',()=>playing?stopTrace():playTrace());
if($('trace-reset'))$('trace-reset').addEventListener('click',()=>{stopTrace();$('trace-time').value=0;cursor();playTrace();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopTrace();});
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stopTrace();},{threshold:0}).observe($('trajectory-explorer'));
svg.addEventListener('pointerdown',e=>{if(!current||!geom)return;stopTrace();const b=svg.getBoundingClientRect(),px=(e.clientX-b.left)/b.width*960;const t=Math.max(0,Math.min(geom.maxT,(px-left)/width*geom.maxT));$('trace-time').value=Math.min(current.t.length-1,Math.round(t/current.dt_s));cursor();});select();}
if($('ledger-body')) ledger().catch(e=>{$('ledger-body').replaceChildren();const tr=document.createElement('tr'),td=cell('数据未能加载；请通过 HTTP 预览本页，或读取下方报告数据 JSON。');td.colSpan=6;tr.append(td);$('ledger-body').append(tr);console.error(e);});
if($('trace-svg')) traces().catch(e=>{$('trace-source').textContent='交互数据未加载。请通过 HTTP 预览，静态轨迹图与 JSON 来源仍可审阅。';console.error(e);});
document.querySelectorAll('.figure > img').forEach(img=>{const a=document.createElement('a');a.href=img.getAttribute('src');a.target='_blank';a.rel='noopener';a.className='chart-link';a.setAttribute('aria-label','打开原图：'+img.alt);img.replaceWith(a);a.append(img);});

// Load archived motion only near the viewport; respect manual pause and reduced motion.
const motionPreference=matchMedia('(prefers-reduced-motion: reduce)');
const motionMobile=matchMedia('(max-width: 580px)');
const motionStates=new Map();
function gifState(el,on){el.src=on?el.dataset.src:el.dataset.poster;const button=el.closest('figure').querySelector('.motion-toggle');button.textContent=on?'暂停 GIF':'播放 GIF';button.setAttribute('aria-pressed',String(on));}
function updateMotion(el,state){const auto=state.visible&&!document.hidden&&!motionPreference.matches&&!state.manualPause;
  if(el.matches('video')){if(!el.src&&(state.visible||state.requested)){el.src=el.dataset.src;el.load();}if(auto)el.play().catch(()=>{});else el.pause();}
  else gifState(el,auto);
}
function selectMotionLayout(el){if(!el.dataset.srcMobile)return;const mobile=motionMobile.matches;const src=mobile?el.dataset.srcMobile:el.dataset.srcDesktop;const t=el.currentTime;el.dataset.src=src;el.poster=mobile?el.dataset.posterMobile:el.dataset.posterDesktop;if(el.getAttribute('src')&&el.getAttribute('src')!==src){const state=motionStates.get(el);const restore=()=>{el.currentTime=t;state.manualPause=state.wasPaused;updateMotion(el,state);};state.wasPaused=state.manualPause;el.addEventListener('loadedmetadata',restore,{once:true});el.src=src;el.load();}}
const motionObserver=new IntersectionObserver(entries=>{for(const e of entries){const s=motionStates.get(e.target);s.visible=e.isIntersecting;updateMotion(e.target,s);}},{threshold:.3});
document.querySelectorAll('[data-motion-video],[data-motion-gif]').forEach(el=>{const state={visible:false,manualPause:false,requested:false};motionStates.set(el,state);
  if(el.dataset.srcMobile){el.dataset.srcDesktop=el.dataset.src;el.dataset.posterDesktop=el.poster;selectMotionLayout(el);}
  if(el.matches('video')){el.addEventListener('pause',()=>{if(state.visible&&!document.hidden&&!motionPreference.matches)state.manualPause=true;});el.addEventListener('play',()=>{state.manualPause=false;});}
  else el.closest('figure').querySelector('.motion-toggle').addEventListener('click',()=>{const on=el.getAttribute('src')===el.dataset.src;state.manualPause=on;gifState(el,!on);});
  motionObserver.observe(el);
});
document.addEventListener('visibilitychange',()=>motionStates.forEach((s,el)=>updateMotion(el,s)));
motionPreference.addEventListener('change',()=>motionStates.forEach((s,el)=>updateMotion(el,s)));
motionMobile.addEventListener('change',()=>motionStates.forEach((s,el)=>selectMotionLayout(el)));
