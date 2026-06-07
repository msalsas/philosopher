"""Live dashboard feed: a Server-Sent Events fan-out over the event bus.

`DashboardHub` subscribes to every event and pushes it to all connected SSE
clients (one asyncio.Queue per client) plus a small backlog so a freshly opened
dashboard sees recent activity. The `/dashboard` page (DASHBOARD_HTML) renders
the per-person snapshot from `/sessions` and a live event log from the stream.
"""
# ruff: noqa: E501  (the DASHBOARD_HTML string holds long CSS/JS lines)
from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any

from philosopher.events.bus import Event, EventBus


class DashboardHub:
    """Fans bus events out to connected dashboard clients (SSE)."""

    def __init__(self, history: int = 50, client_buffer: int = 100) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._client_buffer = client_buffer

    def register(self, bus: EventBus) -> None:
        bus.subscribe_all(self._on_event)

    def connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._client_buffer)
        self._clients.add(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def recent(self) -> list[dict[str, Any]]:
        return list(self._history)

    async def _on_event(self, event: Event) -> None:
        item = {"type": event.type.value, "data": event.data, "ts": event.timestamp}
        self._history.append(item)
        for q in list(self._clients):
            # Slow client: drop rather than block the bus.
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(item)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Philosopher Dashboard</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { padding: 14px 20px; background: #171a21; border-bottom: 1px solid #262b36; }
  header h1 { margin: 0; font-size: 18px; }
  .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 8px; padding: 14px; }
  .card.full { grid-column: 1 / -1; }
  .card h2 { margin: 0 0 10px; font-size: 14px; color: #9aa4b2; text-transform: uppercase; letter-spacing: .04em; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262f; vertical-align: middle; }
  th { color: #9aa4b2; font-weight: 600; }
  #log { font-family: ui-monospace, monospace; font-size: 12px; max-height: 60vh; overflow-y: auto; }
  #log div { padding: 2px 0; border-bottom: 1px solid #21262f; }
  .t { color: #6ea8fe; }
  .dim { color: #6b7280; }
  /* "Present now" chips */
  #present { display: flex; flex-wrap: wrap; gap: 10px; min-height: 28px; }
  .chip { display: flex; align-items: center; gap: 8px; background: #1d2230; border: 1px solid #2c3342;
          border-radius: 999px; padding: 5px 12px 5px 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; box-shadow: 0 0 6px #3fb95066; }
  .chip .emo { font-size: 12px; color: #9aa4b2; }
  /* Per-person emotion histogram (inline stacked bar) */
  .hist { display: flex; height: 14px; width: 160px; border-radius: 3px; overflow: hidden; background: #0f1115; }
  .hist span { display: block; height: 100%; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #9aa4b2; margin-top: 8px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
</style>
</head>
<body>
<header><h1>🧸 Philosopher Dashboard</h1></header>
<div class="wrap">
  <div class="card full">
    <h2>Present now</h2>
    <div id="present"><span class="dim">nobody around</span></div>
  </div>
  <div class="card">
    <h2>People</h2>
    <table id="people"><thead><tr>
      <th>Name</th><th>Msgs</th><th>Replies</th><th>Emotion mix</th><th>Last seen</th>
    </tr></thead><tbody></tbody></table>
    <div class="legend" id="legend"></div>
  </div>
  <div class="card">
    <h2>Live activity</h2>
    <div id="log"></div>
  </div>
  <div class="card full">
    <h2>Emotions (live, last 60s)</h2>
    <canvas id="chart" height="160"></canvas>
  </div>
</div>
<script>
// Fixed palette so the same emotion keeps its colour across the histogram and chart.
const EMO_COLORS = {happy:'#3fb950', sad:'#6ea8fe', angry:'#f85149', surprise:'#d29922',
  fear:'#a371f7', disgust:'#db61a2', neutral:'#6b7280', contempt:'#8b949e'};
function emoColor(k){ return EMO_COLORS[k] || '#8b949e'; }
function topEmotion(e){let b=null,n=-1;for(const k in (e||{})){if(e[k]>n){n=e[k];b=k;}}return b||'-';}

function histogram(emotions){
  const entries = Object.entries(emotions||{}).filter(([,n])=>n>0);
  const total = entries.reduce((a,[,n])=>a+n,0);
  if(!total) return '<span class="dim">-</span>';
  const bars = entries.map(([k,n])=>
    `<span style="width:${(100*n/total).toFixed(1)}%;background:${emoColor(k)}" title="${k}: ${n}"></span>`).join('');
  return `<div class="hist">${bars}</div>`;
}

const seenEmotions = new Set();
function renderLegend(){
  const el = document.getElementById('legend');
  el.innerHTML = [...seenEmotions].sort().map(k=>
    `<span><i style="background:${emoColor(k)}"></i>${k}</span>`).join('');
}

async function refreshPeople(){
  try{
    const r = await fetch('/sessions'); const s = await r.json();
    const tb = document.querySelector('#people tbody'); tb.innerHTML='';
    const present = [];
    for(const [fid, v] of Object.entries(s)){
      Object.keys(v.emotions||{}).forEach(k=>seenEmotions.add(k));
      const ago = Math.round(Date.now()/1000 - (v.last_seen||0));
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${v.name || '<span class=dim>unknown</span>'}</td>`+
        `<td>${v.messages}</td><td>${v.responses}</td>`+
        `<td>${histogram(v.emotions)}</td><td class=dim>${ago}s ago</td>`;
      tb.appendChild(tr);
      if(v.present) present.push(v);
    }
    renderLegend();
    renderPresent(present);
  }catch(e){}
}

function renderPresent(present){
  const el = document.getElementById('present');
  if(!present.length){ el.innerHTML = '<span class="dim">nobody around</span>'; return; }
  el.innerHTML = present.map(v=>{
    const emo = v.current_emotion || 'neutral';
    return `<span class="chip"><span class="dot" style="background:${emoColor(emo)}"></span>`+
      `${v.name || '<span class=dim>unknown</span>'} <span class="emo">${emo}</span></span>`;
  }).join('');
}

// --- Live emotion chart: rolling 60s buckets fed from SSE emotion_detected events ---
const WINDOW = 60, buckets = [];  // each: {t: epochSec, counts:{emo:n}}
function addEmotion(emo){
  const t = Math.floor(Date.now()/1000);
  let b = buckets[buckets.length-1];
  if(!b || b.t !== t){ b = {t, counts:{}}; buckets.push(b); }
  b.counts[emo] = (b.counts[emo]||0)+1;
  seenEmotions.add(emo);
}
function drawChart(){
  const cv = document.getElementById('chart');
  const w = cv.clientWidth || cv.parentNode.clientWidth - 28;
  cv.width = w; const h = cv.height;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,w,h);
  const now = Math.floor(Date.now()/1000);
  while(buckets.length && buckets[0].t < now - WINDOW) buckets.shift();
  let max = 1;
  for(const b of buckets){ const s = Object.values(b.counts).reduce((a,n)=>a+n,0); if(s>max) max=s; }
  const colW = w / WINDOW;
  for(const b of buckets){
    const x = w - (now - b.t + 1) * colW;
    let y = h;
    for(const k of Object.keys(b.counts)){
      const bh = (b.counts[k]/max) * (h - 6);
      y -= bh;
      ctx.fillStyle = emoColor(k);
      ctx.fillRect(x, y, Math.max(1, colW - 1), bh);
    }
  }
}

function startStream(){
  const log = document.getElementById('log');
  const es = new EventSource('/dashboard/stream');
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch(_) { return; }
    if(ev.type === 'emotion_detected' && ev.data && ev.data.emotion){ addEmotion(ev.data.emotion); }
    const d = document.createElement('div');
    const t = new Date((ev.ts||0)*1000).toLocaleTimeString();
    d.innerHTML = `<span class=dim>${t}</span> <span class=t>${ev.type}</span> `+
      `<span class=dim>${JSON.stringify(ev.data)}</span>`;
    log.prepend(d);
    while (log.childNodes.length > 200) log.removeChild(log.lastChild);
    refreshPeople();
  };
}
refreshPeople(); setInterval(refreshPeople, 3000); startStream();
setInterval(drawChart, 1000);
</script>
</body>
</html>
"""
