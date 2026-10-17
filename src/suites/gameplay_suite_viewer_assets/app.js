const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { data: null, view: 'overview', inspectRun: null, tick: 0, mode: 'raw', playing: null, compareTick: 0, comparePlaying: null };
const accents = ['#d9ff43', '#ff6b4a', '#7fa9ff', '#efb5ff', '#ffd36a'];

const fmt = {
  number: (v, digits = 0) => v == null ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits }),
  pct: v => v == null ? '—' : `${(v * 100).toFixed(1)}%`,
  ms: v => v == null ? '—' : `${Number(v).toFixed(0)} ms`,
  date: v => v ? new Date(v).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : 'Unknown date',
  duration: (a, b) => {
    if (!a || !b) return 'Duration unavailable';
    const sec = Math.max(0, (new Date(b) - new Date(a)) / 1000);
    return sec >= 3600 ? `${Math.floor(sec / 3600)}h ${Math.round(sec % 3600 / 60)}m duration` : `${Math.round(sec / 60)}m duration`;
  }
};
const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const runLabel = run => `${run.name} · ${run.repeatIndex || 1}/${run.repeatCount || 1}`;
const currentRun = () => state.data.runs.find(run => run.id === state.inspectRun) || state.data.runs[0];

function renderClickMarker(image, marker, tick, enabled=true) {
  marker.replaceChildren();
  const target=enabled&&tick?.clickTarget;
  if(!target||!image.naturalWidth||!image.naturalHeight)return;
  marker.setAttribute('viewBox',`0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  marker.setAttribute('preserveAspectRatio','xMidYMid meet');
  const {x,y}=target, shape=state.data.suite.clickMarker||'cross';
  marker.innerHTML=shape==='circle'
    ? `<circle class="marker-stroke" cx="${x}" cy="${y}" r="16"/>`
    : `<path class="marker-stroke" d="M ${x-20} ${y} H ${x-6} M ${x+6} ${y} H ${x+20} M ${x} ${y-20} V ${y-6} M ${x} ${y+6} V ${y+20}"/>`;
}

async function start() {
  try {
    const response = await fetch('/api/suite');
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    state.data = await response.json();
    state.inspectRun = state.data.runs[0]?.id;
    hydrateOverview(); hydrateSelectors(); bindEvents(); renderInspect(); renderCompare();
    const compareNav=$('.nav-item[data-view="compare"]');
    compareNav.disabled=!state.data.suite.compareEnabled;
    compareNav.title=state.data.suite.compareEnabled?'Compare two synchronized runs':'Add at least two runs to enable Compare Studio';
    $('#loading').classList.add('hidden'); $('#overview').classList.remove('hidden');
  } catch (error) {
    $('#loading').classList.add('hidden'); $('#error').classList.remove('hidden');
    $('#error').innerHTML = `<h2>Archive unavailable</h2><p>${escapeHtml(error.message)}</p>`;
  }
}

function hydrateOverview() {
  const { suite, runs, groups } = state.data;
  $('#suite-name').textContent = suite.name.replaceAll('_', ' ');
  $('#suite-folder').textContent = suite.folderName;
  $('#suite-status').classList.toggle('failed', suite.status !== 'completed');
  $('#suite-date').textContent = fmt.date(suite.startedAt);
  $('#suite-duration').textContent = fmt.duration(suite.startedAt, suite.completedAt);
  $('#suite-config').textContent = suite.config || 'No config recorded';
  $('#open-suite span').textContent = suite.sourceType==='suite'?'Open suite folder':runs.length===1?'Open run folder':'Open runs root';
  $('#hero-run-count').textContent = runs.length;
  const totals = runs.map(r => r.summary);
  const avg = key => { const values = totals.map(s => s[key]).filter(v => v != null); return values.length ? values.reduce((a,b)=>a+b,0)/values.length : null; };
  $('#kpi-strip').innerHTML = [
    [totals.reduce((a,s)=>a+s.ticks,0), 'Observed ticks'],
    [totals.reduce((a,s)=>a+s.progress,0), 'Progress events'],
    [fmt.pct(avg('validRate')), 'Valid action rate'],
    [fmt.ms(avg('avgLatency')), 'Mean inference'],
    [fmt.pct(avg('stuckRate')), 'Stuck-state rate']
  ].map(([v,l]) => `<div class="kpi"><strong>${v}</strong><span>${l}</span></div>`).join('');
  $('#group-grid').innerHTML = groups.map((group, index) => {
    const a = group.aggregate; const color = accents[index % accents.length];
    return `<article class="group-card" style="--accent:${color}"><span class="card-index">Study ${String(index+1).padStart(2,'0')}</span><h3>${escapeHtml(group.name)}</h3><p class="model-name">${escapeHtml(group.model || 'Model not recorded')}</p><div class="group-metrics">
      ${metric(fmt.number(a.progress,1),'Mean progress')}${metric(fmt.pct(a.validRate),'Valid actions')}${metric(fmt.ms(a.avgLatency),'Mean latency')}
      ${metric(fmt.pct(a.stuckRate),'Stuck ticks')}${metric(fmt.number(a.maxActionStreak,1),'Max repeat')}${metric(a.runs,'Runs')}</div>
      <div class="group-runs">${group.runIds.map(id => { const r=runs.find(x=>x.id===id); return `<button class="mini-run" data-run="${id}"><b>Repeat ${r.repeatIndex || 1}</b>${r.summary.progress} progress</button>`; }).join('')}</div></article>`;
  }).join('');
  $('#run-grid').innerHTML = runs.map((run,index) => `<button class="run-card" data-run="${run.id}"><div class="run-card-top"><h3>${escapeHtml(run.name)}</h3><span class="run-number">No. ${String(index+1).padStart(2,'0')}</span></div><p class="run-card-meta">Repeat ${run.repeatIndex || 1} · ${run.summary.ticks} ticks · ${escapeHtml(run.status)}</p><div class="micro-bars">
    ${bar('Progress', run.summary.progress, Math.max(...runs.map(r=>r.summary.progress),1), 'var(--acid)')}
    ${bar('Stuck', run.summary.stuckRate, 1, 'var(--coral)', fmt.pct(run.summary.stuckRate))}
    ${bar('Latency', run.summary.avgLatency, Math.max(...runs.map(r=>r.summary.avgLatency || 0),1), 'var(--blue)', fmt.ms(run.summary.avgLatency))}
  </div></button>`).join('');
}
function metric(v,l){ return `<div class="group-metric"><b>${v}</b><small>${l}</small></div>`; }
function bar(label,v,max,color,shown=fmt.number(v,1)){ return `<div class="micro-row"><span>${label}</span><div class="micro-track"><i style="width:${Math.min(100,(v||0)/max*100)}%;--bar:${color}"></i></div><b>${shown}</b></div>`; }

function hydrateSelectors() {
  const options = state.data.runs.map(r => `<option value="${r.id}">${escapeHtml(runLabel(r))}</option>`).join('');
  $('#inspect-run').innerHTML = options; $('#compare-left').innerHTML = options; $('#compare-right').innerHTML = options;
  $('#compare-right').value = state.data.runs[1]?.id || state.data.runs[0]?.id;
}

function bindEvents() {
  $$('.nav-item').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
  document.addEventListener('click', event => { const target = event.target.closest('[data-run]'); if(target){ state.inspectRun=target.dataset.run; $('#inspect-run').value=state.inspectRun; state.tick=0; showView('inspect'); renderInspect(); } });
  $('#open-suite').addEventListener('click', () => openFolder()); $('#open-run').addEventListener('click', () => openFolder(state.inspectRun));
  $('#inspect-run').addEventListener('change', e => { state.inspectRun=e.target.value; state.tick=0; stopPlayback(); renderInspect(); });
  $('#inspect-scrubber').addEventListener('input', e => { state.tick=+e.target.value; renderInspect(); });
  $('#previous-tick').onclick=()=>stepInspect(-1); $('#next-tick').onclick=()=>stepInspect(1);
  $('#previous-event').onclick=()=>jumpEvent(-1); $('#next-event').onclick=()=>jumpEvent(1); $('#play').onclick=togglePlayback;
  $$('#screen-mode button').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;renderInspect();}));
  $('#inspect-image').addEventListener('load', e => { $('#image-resolution').textContent=`${e.target.naturalWidth} × ${e.target.naturalHeight}`; renderClickMarker(e.target,$('#inspect-click-marker'),currentRun()?.ticks[state.tick],state.mode!=='progress'); });
  $('#left-image').addEventListener('load', e => renderClickMarker(e.target,$('#left-click-marker'),selectedCompareRuns()[0]?.ticks[state.compareTick]));
  $('#right-image').addEventListener('load', e => renderClickMarker(e.target,$('#right-click-marker'),selectedCompareRuns()[1]?.ticks[state.compareTick]));
  $('#compare-left').addEventListener('change',renderCompare); $('#compare-right').addEventListener('change',renderCompare);
  $('#swap-runs').onclick=()=>{ const a=$('#compare-left').value; $('#compare-left').value=$('#compare-right').value; $('#compare-right').value=a; renderCompare(); };
  $('#compare-scrubber').addEventListener('input',e=>{state.compareTick=+e.target.value;renderCompare();});
  $('#compare-prev').onclick=()=>stepCompare(-1); $('#compare-next').onclick=()=>stepCompare(1); $('#compare-play').onclick=toggleComparePlayback;
  document.addEventListener('keydown', e => { if(state.view==='inspect' && !['SELECT','INPUT'].includes(e.target.tagName)){ if(e.key==='ArrowLeft')stepInspect(-1); if(e.key==='ArrowRight')stepInspect(1); if(e.key===' '){e.preventDefault();togglePlayback();} } });
}

function showView(view) {
  state.view=view; $$('.view').forEach(v=>v.classList.toggle('hidden',v.id!==view)); $$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  stopPlayback(); stopComparePlayback(); window.scrollTo({top:0,behavior:'smooth'});
}

function renderInspect() {
  const run=currentRun(); if(!run) return; const max=Math.max(0,run.ticks.length-1); state.tick=Math.min(state.tick,max); const tick=run.ticks[state.tick];
  $('#inspect-run').value=run.id; $('#inspect-scrubber').max=max; $('#inspect-scrubber').value=state.tick; $('#inspect-tick').textContent=tick?.tick ?? state.tick; $('#inspect-total').textContent=`${run.ticks.length} frames`;
  $$('#screen-mode button').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
  const source=tick?.screens?.[state.mode]; $('#inspect-image').classList.toggle('hidden',!source); $('#screen-empty').classList.toggle('hidden',!!source); if(source) $('#inspect-image').src=source; renderClickMarker($('#inspect-image'),$('#inspect-click-marker'),tick,!!source&&state.mode!=='progress');
  $('#inspect-evidence').innerHTML=evidenceHtml(run,tick); renderTimeline(run); renderDispatch(tick);
}
function evidenceHtml(run,tick){ if(!tick)return '<p>No ticks recorded.</p>'; const flags=[tick.progress.event&&['Progress event','good'],tick.state.stuck&&['Stuck','hot'],tick.error&&['Inference error','hot'],tick.skipped&&['Skipped','cool']].filter(Boolean); return `<div class="tick-title"><strong>Tick ${tick.tick}</strong><span>${fmt.ms(tick.latency)}</span></div><div class="flag-row">${flags.length?flags.map(f=>`<span class="flag ${f[1]}">${f[0]}</span>`).join(''):'<span class="flag">Nominal</span>'}</div>
  <div class="evidence-block"><small>Model response</small><div class="action">${escapeHtml(tick.action)}</div></div>
  <div class="evidence-block"><small>Telemetry</small><div class="stat-pair"><div><b>${tick.progress.count ?? '—'}</b><span>Progress count</span></div><div><b>${tick.state.repeated ?? 0}</b><span>Action repeats</span></div><div><b>${fmt.number(tick.progress.score,3)}</b><span>Region change</span></div><div><b>${fmt.number(tick.state.screenChange,3)}</b><span>Screen change</span></div></div></div>
  ${tick.state.reason?`<div class="evidence-block"><small>Stuck analysis</small><code>${escapeHtml(tick.state.reason)}</code></div>`:''}${tick.error?`<div class="evidence-block"><small>Inference error</small><code>${escapeHtml(tick.error)}</code></div>`:''}
  <div class="evidence-block"><small>Generation profile</small><code>${escapeHtml(tick.generation.key || 'Default')}\n${escapeHtml(JSON.stringify(tick.generation.effective,null,2))}</code></div>`; }
function renderTimeline(run){ const maxLatency=Math.max(...run.ticks.map(t=>t.latency||0),1); $('#timeline').innerHTML=`<div class="timeline-head"><b>Inference pulse</b><span>Acid = progress · Coral foot = stuck</span></div><div class="timeline-rail">${run.ticks.map((t,i)=>`<i class="timeline-bar ${t.progress.event?'event':''} ${t.state.stuck?'stuck':''} ${i===state.tick?'active':''}" data-tick="${i}" title="Tick ${t.tick} · ${fmt.ms(t.latency)}" style="height:${Math.max(5,(t.latency||0)/maxLatency*100)}%"></i>`).join('')}</div>`; $$('.timeline-bar',$('#timeline')).forEach(b=>b.onclick=()=>{state.tick=+b.dataset.tick;renderInspect();}); }
function renderDispatch(tick){
  const exhibit=$('#vlm-dispatch'), dispatch=tick?.dispatch;
  if(!dispatch){exhibit.innerHTML=`<div class="dispatch-empty"><b>No VLM dispatch preserved</b><span>This tick has no matching record in vlm_io.json.</span></div>`;return;}
  const messages=dispatch.messages||[], params=Object.entries(dispatch.generationParams||{});
  exhibit.innerHTML=`<header class="dispatch-head"><div class="dispatch-title"><div class="dispatch-seal">IO</div><div><h2>VLM dispatch</h2><p>The effective conversation · exactly as recorded</p></div></div><div class="dispatch-folio"><div><span>Method</span><b>${escapeHtml(dispatch.api||'unknown')}</b></div><div><span>Messages</span><b>${messages.length}</b></div><div><span>Latency</span><b>${fmt.ms(dispatch.latency)}</b></div></div></header>
    <div class="conversation">${messages.map(message=>messageHtml(message,dispatch,tick)).join('')}<div class="response-divider">Model return</div>${messageHtml({role:'assistant',index:'R',content:dispatch.response??''},dispatch,tick,true)}</div>
    <footer class="dispatch-foot"><span>Generation parameters</span>${params.length?params.map(([key,value])=>`<code class="param-tag">${escapeHtml(key)} = ${escapeHtml(value)}</code>`).join(''):'<code class="param-tag">defaults</code>'}</footer>`;
}
function messageHtml(message,dispatch,tick,isResponse=false){const role=String(message.role||'unknown').toLowerCase();const label=isResponse?'response':`message ${Number(message.index)+1}`;return `<article class="message" data-role="${escapeHtml(role)}"><div class="message-label"><i class="role-glyph">${escapeHtml(role.slice(0,1).toUpperCase()||'?')}</i>${escapeHtml(role)} · ${escapeHtml(label)}</div><div class="message-bubble">${contentHtml(message.content,dispatch,tick)}</div></article>`;}
function contentHtml(content,dispatch,tick){
  if(Array.isArray(content))return content.map(part=>partHtml(part,dispatch,tick)).join('');
  if(content&&typeof content==='object')return `<pre class="message-object">${escapeHtml(JSON.stringify(content,null,2))}</pre>`;
  return `<p>${escapeHtml(content??'').replaceAll('\n','<br>')}</p>`;
}
function partHtml(part,dispatch,tick){
  if(!part||typeof part!=='object')return `<p class="message-part">${escapeHtml(part)}</p>`;
  if(part.type==='text')return `<p class="message-part">${escapeHtml(part.text??'').replaceAll('\n','<br>')}</p>`;
  if(part.type==='image'){const current=part.isCurrent??(part.image==null||part.image==='current_vlm_image');const source=part.imageUrl||(current?(dispatch.image||tick?.screens?.vlm):null);const name=part.imageName||(current?dispatch.imageName:part.image)||'image unavailable';return `<div class="message-part message-image">${source?`<img src="${escapeHtml(source)}" alt="VLM image input">`:''}<div><span>${current?'Current':'Historical'} VLM image</span><small>${escapeHtml(name)}</small></div></div>`;}
  return `<pre class="message-part message-object">${escapeHtml(JSON.stringify(part,null,2))}</pre>`;
}
function stepInspect(delta){ const run=currentRun();state.tick=Math.max(0,Math.min(run.ticks.length-1,state.tick+delta));renderInspect(); }
function jumpEvent(dir){ const ticks=currentRun().ticks; let i=state.tick+dir; while(i>=0&&i<ticks.length){const t=ticks[i];if(t.progress.event||t.state.stuck||t.error||t.skipped){state.tick=i;renderInspect();return;}i+=dir;} }
function togglePlayback(){ if(state.playing){stopPlayback();return;} $('#play').textContent='Ⅱ';state.playing=setInterval(()=>{if(state.tick>=currentRun().ticks.length-1){stopPlayback();return;}stepInspect(1);},+$('#play-speed').value); }
function stopPlayback(){ if(state.playing)clearInterval(state.playing);state.playing=null;$('#play').textContent='▶'; }

function selectedCompareRuns(){return [state.data.runs.find(r=>r.id===$('#compare-left').value),state.data.runs.find(r=>r.id===$('#compare-right').value)];}
function renderCompare(){
  const [left,right]=selectedCompareRuns();if(!left||!right)return;
  const max=Math.max(0,Math.min(left.ticks.length,right.ticks.length)-1);state.compareTick=Math.min(state.compareTick,max);
  $('#compare-scrubber').max=max;$('#compare-scrubber').value=state.compareTick;$('#compare-tick').textContent=state.compareTick;$('#compare-total').textContent=`${max+1} aligned frames`;
  $('#left-label').textContent=runLabel(left);$('#right-label').textContent=runLabel(right);
  const lt=left.ticks[state.compareTick],rt=right.ticks[state.compareTick];
  if(lt?.screens.raw)$('#left-image').src=lt.screens.raw;if(rt?.screens.raw)$('#right-image').src=rt.screens.raw;
  renderClickMarker($('#left-image'),$('#left-click-marker'),lt);renderClickMarker($('#right-image'),$('#right-click-marker'),rt);
  $('#left-caption').innerHTML=caption(lt);$('#right-caption').innerHTML=caption(rt);
  $('#compare-deltas').innerHTML=[delta('Progress',left.summary.progress,right.summary.progress,1),delta('Valid actions',left.summary.validRate,right.summary.validRate,100,'%'),delta('Mean latency',left.summary.avgLatency,right.summary.avgLatency,1,' ms',true),delta('Stuck rate',left.summary.stuckRate,right.summary.stuckRate,100,'%',true)].join('');
}
function caption(t){return t?`<b>${escapeHtml(t.action)}</b>${fmt.ms(t.latency)} · progress ${t.progress.count??'—'}${t.state.stuck?' · <span style="color:var(--coral)">stuck</span>':''}`:'No aligned tick';}
function delta(label,a,b,mult=1,suffix='',lowerBetter=false){const diff=((b??0)-(a??0))*mult;const positive=lowerBetter?diff<0:diff>0;const cls=Math.abs(diff)<.0001?'':positive?'positive':'negative';return `<div class="delta"><span>${label} · right vs left</span><b class="${cls}">${diff>0?'+':''}${diff.toFixed(mult===100?1:0)}${suffix}</b></div>`;}
function stepCompare(delta){const [a,b]=selectedCompareRuns();const max=Math.min(a.ticks.length,b.ticks.length)-1;state.compareTick=Math.max(0,Math.min(max,state.compareTick+delta));renderCompare();}
function toggleComparePlayback(){if(state.comparePlaying){stopComparePlayback();return;}$('#compare-play').textContent='Ⅱ';state.comparePlaying=setInterval(()=>{const [a,b]=selectedCompareRuns();if(state.compareTick>=Math.min(a.ticks.length,b.ticks.length)-1){stopComparePlayback();return;}stepCompare(1);},450);}
function stopComparePlayback(){if(state.comparePlaying)clearInterval(state.comparePlaying);state.comparePlaying=null;$('#compare-play').textContent='▶';}

async function openFolder(run){try{const response=await fetch(`/api/open-folder${run?`?run=${encodeURIComponent(run)}`:''}`,{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error);toast(`Opened ${data.path}`);}catch(error){toast(error.message,true);}}
function toast(message,error=false){const el=$('#toast');el.textContent=message;el.style.background=error?'var(--coral)':'var(--paper)';el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2500);}

start();
