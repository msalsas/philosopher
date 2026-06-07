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
  .card h2 { margin: 0 0 10px; font-size: 14px; color: #9aa4b2; text-transform: uppercase; letter-spacing: .04em; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262f; }
  th { color: #9aa4b2; font-weight: 600; }
  #log { font-family: ui-monospace, monospace; font-size: 12px; max-height: 60vh; overflow-y: auto; }
  #log div { padding: 2px 0; border-bottom: 1px solid #21262f; }
  .t { color: #6ea8fe; }
  .dim { color: #6b7280; }
</style>
</head>
<body>
<header><h1>🧸 Philosopher Dashboard</h1></header>
<div class="wrap">
  <div class="card">
    <h2>People</h2>
    <table id="people"><thead><tr>
      <th>Name</th><th>Messages</th><th>Replies</th><th>Top emotion</th><th>Last seen</th>
    </tr></thead><tbody></tbody></table>
  </div>
  <div class="card">
    <h2>Live activity</h2>
    <div id="log"></div>
  </div>
</div>
<script>
function topEmotion(e){let b=null,n=-1;for(const k in (e||{})){if(e[k]>n){n=e[k];b=k;}}return b||'-';}
async function refreshPeople(){
  try{
    const r = await fetch('/sessions'); const s = await r.json();
    const tb = document.querySelector('#people tbody'); tb.innerHTML='';
    for(const [fid, v] of Object.entries(s)){
      const ago = Math.round(Date.now()/1000 - (v.last_seen||0));
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${v.name || '<span class=dim>unknown</span>'}</td>`+
        `<td>${v.messages}</td><td>${v.responses}</td>`+
        `<td>${topEmotion(v.emotions)}</td><td class=dim>${ago}s ago</td>`;
      tb.appendChild(tr);
    }
  }catch(e){}
}
function startStream(){
  const log = document.getElementById('log');
  const es = new EventSource('/dashboard/stream');
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch(_) { return; }
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
</script>
</body>
</html>
"""
