"""Local collaboration harness for published artifact development.

The deployed collaboration surface lives in mindshub_services/anton_services.
This router gives the desktop dev server the same `/collab/...` URL shape so
the shell and comment interactions can be tested before the Lambda stack is
deployed.
"""
from __future__ import annotations

import html
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.artifacts import (
    _iter_artifact_folders,
    _load_metadata,
    _load_published_map,
    serve_url_for,
)

router = APIRouter()

_THREADS: dict[str, dict[str, dict[str, Any]]] = {}


class _ThreadBody(BaseModel):
    body: str = ""
    anchor: dict[str, Any] = {}
    excerpt: str = ""


class _StatusBody(BaseModel):
    status: str


def local_collab_url(user: str, report_id: str) -> str:
    settings = get_app_settings()
    origin = (settings.public_base_url or f"http://{settings.host}:{settings.port}").rstrip("/")
    return f"{origin}/collab/view/{user}/{report_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _artifact_key(user: str, report_id: str) -> str:
    return f"{user}/{report_id}"


def _find_static_artifact(user: str, report_id: str) -> dict[str, Any] | None:
    expected = _artifact_key(user, report_id)
    for folder in _iter_artifact_folders(None):
        published = _load_published_map(folder)
        for name, entry in published.items():
            if not isinstance(entry, dict) or not entry.get("published", True):
                continue
            url = str(entry.get("url") or "")
            matches_id = str(entry.get("report_id") or "") == report_id
            matches_url = f"/view/{user}/{report_id}" in url
            if not (matches_id and matches_url):
                continue
            primary = (folder / name).resolve(strict=False)
            if not primary.is_file():
                continue
            meta = _load_metadata(folder) or {}
            return {
                "key": expected,
                "title": meta.get("name") or folder.name,
                "file": primary,
                "serve_url": serve_url_for(primary),
                "public_url": url,
            }
    return None


def _thread_payload(thread: dict[str, Any]) -> dict[str, Any]:
    return {**thread, "replies": [m for m in thread.get("replies", []) if not m.get("deleted_at")]}


def _list_threads(key: str, status_filter: str = "live", since: str | None = None) -> list[dict[str, Any]]:
    threads = []
    for thread in _THREADS.get(key, {}).values():
        if thread.get("status") == "deleted" and status_filter != "all":
            continue
        if status_filter != "all" and thread.get("status") != status_filter:
            continue
        if since and str(thread.get("updated_at", "")) <= since:
            continue
        threads.append(_thread_payload(thread))
    threads.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
    return threads


def _viewer() -> dict[str, str]:
    return {"sub": "local-dev", "email": "local@cowork.dev", "name": "Local Dev"}


def _shell(artifact: dict[str, Any], user: str, report_id: str) -> str:
    title = html.escape(str(artifact["title"]))
    iframe_src = html.escape(str(artifact["serve_url"]))
    public_url = html.escape(str(artifact.get("public_url") or ""))
    api_base = f"/collab/view/{html.escape(user)}/{html.escape(report_id)}"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} - Collaboration</title><style>
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1115;color:#e8eaed;height:100vh;overflow:hidden}}.top{{height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid #262b35;background:#121620}}.brand{{font-weight:650;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.brand span{{color:#81d8e8}}.wrap{{display:grid;grid-template-columns:minmax(0,1fr)380px;height:calc(100vh - 42px)}}.stage{{position:relative;min-width:0;min-height:0}}iframe{{width:100%;height:100%;border:0;background:white}}aside{{border-left:1px solid #262b35;background:#171b24;display:flex;flex-direction:column;min-width:0}}.panel-head{{height:48px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid #262b35}}.count{{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;margin-left:6px;padding:0 6px;border-radius:999px;background:#253143;color:#cfd6e3;font-size:11px;font-weight:600}}.tabs{{display:flex;gap:3px;margin:10px 14px;padding:3px;background:#10141c;border:1px solid #303847;border-radius:10px}}.tab{{flex:1;height:30px;border:0;border-radius:7px;background:transparent;color:#9aa3b2;font:inherit;font-size:12px;font-weight:500;cursor:pointer}}.tab.active{{background:#202838;color:#f5f7fb;font-weight:600}}.btn,textarea{{font:inherit}}.threads{{flex:1;overflow:auto;padding:0 14px 14px;display:grid;align-content:start;gap:10px}}.thread{{border:1px solid #303847;border-radius:10px;background:#10141c;padding:12px}}.thread.resolved{{opacity:.86}}.meta{{font-size:11px;color:#9aa3b2;margin-bottom:6px}}.msg{{font-size:13px;line-height:1.45;white-space:pre-wrap}}textarea{{width:100%;min-height:58px;resize:vertical;background:#0d1118;color:#fff;border:1px solid #303847;border-radius:8px;padding:8px}}textarea:focus{{outline:0;border-color:#1a8596;box-shadow:0 0 0 3px rgba(26,133,150,.18)}}.actions{{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}}.btn{{background:#253143;color:#f5f7fb;border:1px solid #3a4658;border-radius:8px;padding:6px 10px;cursor:pointer;text-decoration:none;font-weight:500;font-size:12px}}.btn.primary{{background:#1a8596;border-color:#1a8596;color:#061018;font-weight:600;font-size:12.5px}}.btn:disabled{{opacity:.55;cursor:not-allowed}}.composer{{border-top:1px solid #262b35;padding:10px 14px 14px}}.empty{{color:#8d97a8;font-size:13px;padding:20px;text-align:center}}.anchor-dot{{display:none;position:absolute;width:14px;height:14px;border-radius:50%;background:#1a8596;border:2px solid white;box-shadow:0 2px 10px #0008;transform:translate(-50%,-50%);z-index:4;pointer-events:none}}.anchor-dot.open,.quick-comment.open{{display:block}}.quick-comment{{display:none;position:absolute;width:min(320px,calc(100% - 24px));padding:10px;border:1px solid #3a4658;border-radius:10px;background:#171b24;box-shadow:0 18px 44px #0009;z-index:5}}.quick-comment textarea{{min-height:76px}}.quick-label{{font-size:12px;color:#9aa3b2;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600}}.quick-error{{display:none;color:#ffb4a8;font-size:12px;line-height:1.35;margin-top:8px;font-weight:600}}.quick-error.open{{display:block}}
</style></head><body><div class="top"><div class="brand"><span>MindsHub Cowork</span> Collaboration - {title}</div><a class="btn" href="{public_url}">Public view</a></div><div class="wrap"><main class="stage"><iframe id="artifact-frame" src="{iframe_src}"></iframe><div id="anchor-dot" class="anchor-dot"></div><div id="quick-comment" class="quick-comment"><div id="quick-label" class="quick-label">Comment on this spot</div><textarea id="quick-body" placeholder="Leave a comment"></textarea><div id="quick-error" class="quick-error"></div><div class="actions"><button id="quick-submit" class="btn primary">Comment</button><button id="quick-cancel" class="btn">Cancel</button></div></div></main><aside><div class="panel-head"><strong>Comments <span id="open-count" class="count">0</span></strong></div><div class="tabs"><button class="tab active" data-filter="live">Open <span id="tab-live">0</span></button><button class="tab" data-filter="resolved">Resolved <span id="tab-resolved">0</span></button><button class="tab" data-filter="all">All <span id="tab-all">0</span></button></div><div id="threads" class="threads"></div><div class="composer"><textarea id="new-body" placeholder="Write a general comment"></textarea><div class="actions"><button id="new-submit" class="btn primary">Comment</button></div></div></aside></div><script>
const apiBase='{api_base}';let anchor={{kind:'general',target_path:location.pathname}};let filter='live';const frame=document.getElementById('artifact-frame');const stage=document.querySelector('.stage');const list=document.getElementById('threads');const quick=document.getElementById('quick-comment');const quickBody=document.getElementById('quick-body');const quickLabel=document.getElementById('quick-label');const quickError=document.getElementById('quick-error');const quickSubmit=document.getElementById('quick-submit');const dot=document.getElementById('anchor-dot');
function esc(s){{return String(s||'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
async function api(path,opt={{}}){{const r=await fetch(apiBase+path,{{...opt,headers:{{'Content-Type':'application/json',...(opt.headers||{{}})}}}});if(!r.ok)throw new Error((await r.text())||r.status);return r.json();}}
function render(data){{const all=(data.threads||[]).filter(t=>t.status!=='deleted');const live=all.filter(t=>t.status==='live').length;const resolved=all.filter(t=>t.status==='resolved').length;document.getElementById('open-count').textContent=live;document.getElementById('tab-live').textContent=live;document.getElementById('tab-resolved').textContent=resolved;document.getElementById('tab-all').textContent=all.length;const threads=filter==='all'?all:all.filter(t=>t.status===filter);list.innerHTML=threads.length?'':'<div class="empty">'+(filter==='resolved'?'No resolved comments yet':'No open comments')+'</div>';threads.forEach(t=>{{const el=document.createElement('div');el.className='thread '+(t.status==='resolved'?'resolved':'');el.dataset.id=t.thread_id;el.innerHTML='<div class="meta">'+esc(t.status)+' by '+esc(t.created_by&&t.created_by.email)+'</div>'+t.replies.map(m=>'<div class="msg">'+esc(m.body)+'</div>').join('')+'<textarea placeholder="Reply"></textarea><div class="actions"><button class="btn reply">Reply</button><button class="btn resolve">'+(t.status==='resolved'?'Reopen':'Resolve')+'</button><button class="btn del">Delete</button></div>';el.querySelector('.reply').onclick=async(e)=>{{e.stopPropagation();const body=el.querySelector('textarea').value.trim();if(body){{await api('/threads/'+t.thread_id+'/replies',{{method:'POST',body:JSON.stringify({{body}})}});load(true);}}}};el.querySelector('.resolve').onclick=async(e)=>{{e.stopPropagation();await api('/threads/'+t.thread_id,{{method:'PATCH',body:JSON.stringify({{status:t.status==='resolved'?'live':'resolved'}})}});load(true);}};el.querySelector('.del').onclick=async(e)=>{{e.stopPropagation();await api('/threads/'+t.thread_id,{{method:'DELETE'}});load(true);}};el.onclick=()=>scrollAnchor(t.anchor);list.appendChild(el);}});}}
async function load(reset=false){{const data=await api('/threads?status=all');render(data);}}
function isEditing(){{const active=document.activeElement;return quick.classList.contains('open')||(active&&active.tagName==='TEXTAREA');}}
function resetAnchor(){{anchor={{kind:'general',target_path:location.pathname}};}}
function showError(message){{quickError.textContent=message;quickError.classList.add('open');}}
function clearError(){{quickError.textContent='';quickError.classList.remove('open');}}
function showQuickComment(x,y,text){{clearError();const pad=12;const left=Math.max(pad,Math.min(x,stage.clientWidth-332));const top=Math.max(pad,Math.min(y,stage.clientHeight-178));dot.style.left=x+'px';dot.style.top=y+'px';quick.style.left=left+'px';quick.style.top=top+'px';quickLabel.textContent=text?'Comment on "'+text.slice(0,80)+'"':'Comment on this spot';dot.classList.add('open');quick.classList.add('open');quickBody.focus();}}
function hideQuickComment(){{dot.classList.remove('open');quick.classList.remove('open');quickBody.value='';clearError();}}
function capture(e){{e.preventDefault();let doc=frame.contentDocument;let win=frame.contentWindow;let sel=doc.getSelection&&doc.getSelection();let text=sel?String(sel).trim():'';anchor={{kind:text?'selection':'point',target_path:win.location.pathname,text_quote:{{exact:text}},viewport:{{x:e.clientX,y:e.clientY,scroll_x:win.scrollX,scroll_y:win.scrollY,width:doc.documentElement.clientWidth,height:doc.documentElement.clientHeight}}}};showQuickComment(e.clientX,e.clientY,text);}}
function wireFrame(){{try{{frame.contentDocument.addEventListener('contextmenu',capture);}}catch(e){{}}}}
function scrollAnchor(a){{try{{const w=frame.contentWindow;if(a&&a.viewport)w.scrollTo(a.viewport.scroll_x||0,a.viewport.scroll_y||0);}}catch(e){{}}}}
async function submitThread(bodyEl,fromQuick=false){{const body=bodyEl.value.trim();if(!body){{if(fromQuick)showError('Write a comment first.');return;}}clearError();quickSubmit.disabled=true;try{{await api('/threads',{{method:'POST',body:JSON.stringify({{body,anchor,excerpt:anchor.text_quote&&anchor.text_quote.exact||''}})}});bodyEl.value='';hideQuickComment();resetAnchor();load(true);}}catch(e){{console.error('Failed to save comment',e);if(fromQuick)showError('Could not save comment. Refresh this collaboration page and try again.');}}finally{{quickSubmit.disabled=false;}}}}
document.getElementById('new-submit').onclick=async()=>submitThread(document.getElementById('new-body'));
document.getElementById('quick-submit').onclick=async()=>submitThread(quickBody,true);
document.getElementById('quick-cancel').onclick=()=>{{hideQuickComment();resetAnchor();}};
quickBody.addEventListener('keydown',e=>{{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')submitThread(quickBody,true);if(e.key==='Escape'){{hideQuickComment();resetAnchor();}}}});
document.querySelectorAll('.tab').forEach(tab=>{{tab.onclick=()=>{{filter=tab.dataset.filter;document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t===tab));load(true);}};}});
frame.onload=wireFrame;wireFrame();load(true);setInterval(()=>{{if(!document.hidden&&!isEditing())load(false).catch(()=>{{}});}},3000);
</script></body></html>"""


def _shell(artifact: dict[str, Any], user: str, report_id: str) -> str:
    title = html.escape(str(artifact["title"]))
    iframe_src = html.escape(str(artifact["serve_url"]))
    public_url = html.escape(str(artifact.get("public_url") or ""))
    api_base = f"/collab/view/{html.escape(user)}/{html.escape(report_id)}"
    shell = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - Collaboration</title>
<style>
:root{
  --bg:#080d18;--surface:#0e1626;--surface2:#131d31;--surface3:#1a2640;
  --line:#1e2a44;--line2:#2a3957;--ink:#f2f6ff;--ink2:#c7d2e5;
  --ink3:#8a97ae;--ink4:#5c6b85;--accent:#22d3ee;--accent2:#08b2cf;
  --accentBg:rgba(34,211,238,.10);--danger:#f87171;--success:#4ade80;
  --ring:0 0 0 3px rgba(34,211,238,.28);--shadow:0 18px 40px rgba(0,0,0,.58);
}
*{box-sizing:border-box}
body{margin:0;height:100vh;overflow:hidden;background:var(--bg);color:var(--ink2);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,textarea{font:inherit}
.top{height:46px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:#0b1220}
.brand{font-weight:600;font-size:13px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.brand span{color:#81d8e8}
.public{height:32px;display:inline-flex;align-items:center;border:1px solid var(--line2);border-radius:8px;background:var(--surface3);color:var(--ink);padding:0 12px;text-decoration:none;font-size:12.5px;font-weight:500}
.wrap{display:grid;grid-template-columns:minmax(0,1fr)400px;height:calc(100vh - 46px)}
.stage{position:relative;min-width:0;min-height:0;background:#050911}
iframe{width:100%;height:100%;border:0;background:white}
aside{border-left:1px solid var(--line);background:var(--bg);display:flex;flex-direction:column;min-width:0;color:var(--ink2)}
.pane-head{display:flex;align-items:center;gap:9px;padding:16px 16px 12px}
.pane-title{font-size:15.5px;line-height:1.2;font-weight:600;color:var(--ink)}
.count{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;padding:0 6px;border-radius:999px;background:var(--surface3);color:var(--ink3);font-size:11px;font-weight:600}
.live{margin-left:auto;display:flex;align-items:center;gap:6px;color:var(--ink3);font-size:12px;font-weight:500}.live-dot{width:7px;height:7px;border-radius:999px;background:var(--success);box-shadow:0 0 8px rgba(74,222,128,.55)}
.tabs{margin:0 16px 12px;display:flex;gap:3px;padding:3px;background:var(--surface2);border:1px solid var(--line);border-radius:10px}
.tab{flex:1;height:30px;border:0;border-radius:7px;background:transparent;color:var(--ink3);font-size:12px;font-weight:500;cursor:pointer}
.tab.active{background:var(--surface);color:var(--ink);box-shadow:0 1px 2px #0008;font-weight:600}.tab span{opacity:.65;font-weight:600}
.threads{flex:1;overflow:auto;padding:2px 14px 14px;display:flex;flex-direction:column;gap:11px}
.empty{margin:auto;text-align:center;max-width:250px;color:var(--ink3);font-size:13px;line-height:1.5}
.thread{background:var(--surface);border:1px solid var(--line);border-radius:13px;box-shadow:0 1px 2px #0007;padding:13px;position:relative}
.thread.focus{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 0 18px rgba(34,211,238,.24)}
.resolved-banner{display:flex;align-items:center;gap:6px;color:var(--success);font-weight:600;font-size:11px;margin-bottom:10px}
.anchor-chip{display:flex;align-items:center;gap:9px;padding:8px 10px;background:var(--accentBg);border-radius:8px;margin-bottom:11px;color:var(--accent);font-weight:600;font-size:12px}.anchor-chip small{display:block;color:var(--ink3);font-weight:500;margin-top:2px}
.author{display:flex;align-items:center;gap:9px}.avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:600;font-size:11px;background:#b98007;color:white;flex-shrink:0}.avatar.reply-avatar{width:24px;height:24px;font-size:9px;background:#1f9cb0}
.who{font-weight:600;color:var(--ink);font-size:13px}.time{font-size:11.5px;color:var(--ink4);margin-top:1px}.body{margin-top:10px;color:var(--ink2);font-size:13.5px;line-height:1.55;white-space:pre-wrap}
.replies{margin-top:12px;padding-top:12px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:12px}.reply-row{display:flex;gap:9px}.reply-body{font-size:13px;line-height:1.5;color:var(--ink2);margin-top:3px;white-space:pre-wrap}
.reply-box{display:none;margin-top:12px}.reply-box.open{display:block}
textarea{width:100%;background:#0b111c;border:1px solid var(--line2);border-radius:9px;padding:9px 11px;color:var(--ink);outline:0;resize:vertical;min-height:62px}textarea:focus{border-color:var(--accent);box-shadow:var(--ring)}
.actions{display:flex;align-items:center;gap:8px;margin-top:10px}.reply-trigger{flex:1;text-align:left;height:32px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink4);padding:0 11px;font-size:12.5px;cursor:text}
.icon-btn{width:32px;height:32px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink3);cursor:pointer}.text-btn{height:32px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink3);padding:0 12px;font-weight:500;font-size:12px;cursor:pointer}
.primary{height:32px;border:0;border-radius:8px;background:var(--accent);color:#061018;padding:0 14px;font-weight:600;font-size:12.5px;cursor:pointer}.danger{color:var(--danger)}
.composer{border-top:1px solid var(--line);padding:11px 14px 13px;background:var(--surface)}.composer-inner{background:#0b111c;border:1px solid var(--line2);border-radius:11px;padding:9px 11px}.composer textarea{border:0;background:transparent;box-shadow:none;min-height:42px;padding:0}.hint{font-size:11px;color:var(--ink4);margin-top:8px}
.quick{display:none;position:absolute;width:min(320px,calc(100% - 24px));background:var(--surface);border:1px solid var(--line2);border-radius:13px;box-shadow:var(--shadow);padding:12px;z-index:10}.quick.open{display:block}.quick-label{font-weight:600;color:var(--ink3);font-size:13px;margin-bottom:8px}.quick-error{display:none;color:#ffb4a8;font-weight:600;font-size:12px;line-height:1.35;margin-top:8px}.quick-error.open{display:block}
.anchor-dot{display:none;position:absolute;width:14px;height:14px;border-radius:50%;background:var(--accent);border:2px solid white;box-shadow:0 2px 10px #0008;transform:translate(-50%,-50%);z-index:9}.anchor-dot.open{display:block}
</style>
</head>
<body>
<div class="top"><div class="brand"><span>MindsHub Cowork</span> Collaboration - __TITLE__</div><a class="public" href="__PUBLIC_URL__">Public view</a></div>
<div class="wrap">
  <main class="stage"><iframe id="artifact-frame" src="__IFRAME_SRC__"></iframe><div id="anchor-dot" class="anchor-dot"></div><div id="quick" class="quick"><div id="quick-label" class="quick-label">Comment on this spot</div><textarea id="quick-body" placeholder="Leave a comment"></textarea><div id="quick-error" class="quick-error"></div><div class="actions"><button id="quick-submit" class="primary">Comment</button><button id="quick-cancel" class="text-btn">Cancel</button></div></div></main>
  <aside><div class="pane-head"><div class="pane-title">Comments</div><span id="open-count" class="count">0</span><div class="live"><span class="live-dot"></span>Live</div></div><div class="tabs"><button class="tab active" data-filter="live">Open <span id="tab-live">0</span></button><button class="tab" data-filter="resolved">Resolved <span id="tab-resolved">0</span></button><button class="tab" data-filter="all">All <span id="tab-all">0</span></button></div><div id="threads" class="threads"></div><div class="composer"><div class="composer-inner"><textarea id="general-body" placeholder="Right-click a metric to anchor, or write a general comment..."></textarea><div class="actions"><span class="hint">Right-click to anchor</span><button id="general-submit" class="primary" style="margin-left:auto">Comment</button></div></div></div></aside>
</div>
<script>
const apiBase="__API_BASE__";let anchor={kind:"general",target_path:location.pathname};let filter="live";
const byId=id=>document.getElementById(id), frame=byId("artifact-frame"), stage=document.querySelector(".stage"), threadsEl=byId("threads"), quick=byId("quick"), quickBody=byId("quick-body"), quickLabel=byId("quick-label"), quickError=byId("quick-error"), quickSubmit=byId("quick-submit"), dot=byId("anchor-dot");
function esc(s){return String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function initials(p){const s=(p&&(p.name||p.email)||"Local Dev").trim();return s.split(/\\s+/).map(x=>x[0]).slice(0,2).join("").toUpperCase()||"LD";}
function timeAgo(iso){const ms=Date.now()-Date.parse(iso||new Date());const m=Math.max(0,Math.floor(ms/60000));if(m<1)return"now";if(m<60)return`${m}m`;const h=Math.floor(m/60);if(h<24)return`${h}h`;return`${Math.floor(h/24)}d`;}
async function api(path,opt={}){const r=await fetch(apiBase+path,{...opt,headers:{"Content-Type":"application/json",...(opt.headers||{})}});if(!r.ok)throw new Error((await r.text())||r.status);return r.json();}
function anchorHtml(t){const a=t.anchor||{};if(a.kind==="general")return"";const label=t.excerpt||a.text_quote&&a.text_quote.exact||"Anchored comment";return`<div class="anchor-chip">🔗<div>${esc(label.slice(0,90))}<small>${esc(a.target_path||"artifact")}</small></div></div>`;}
function render(data){const all=(data.threads||[]).filter(t=>t.status!=="deleted");const live=all.filter(t=>t.status==="live").length,res=all.filter(t=>t.status==="resolved").length;byId("open-count").textContent=live;byId("tab-live").textContent=live;byId("tab-resolved").textContent=res;byId("tab-all").textContent=all.length;const list=filter==="all"?all:all.filter(t=>t.status===filter);if(!list.length){threadsEl.innerHTML=`<div class="empty"><b>${filter==="resolved"?"No resolved comments yet":"You're all caught up"}</b><br><br>${filter==="live"?"Right-click any metric to start a thread.":"Threads will appear here."}</div>`;return;}threadsEl.innerHTML=list.map(t=>{const first=t.replies&&t.replies[0]||{};const rest=(t.replies||[]).slice(1);const p=t.created_by||first.created_by||{};return `<article class="thread ${t.status==="resolved"?"":"focus"}" data-id="${esc(t.thread_id)}">${t.status==="resolved"?`<div class="resolved-banner">✓ Resolved</div>`:""}${anchorHtml(t)}<div class="author"><div class="avatar">${esc(initials(p))}</div><div><div class="who">${esc(p.name||p.email||"Local Dev")}</div><div class="time">${timeAgo(t.created_at)}</div></div></div><div class="body">${esc(first.body)}</div>${rest.length?`<div class="replies">${rest.map(r=>`<div class="reply-row"><div class="avatar reply-avatar">${esc(initials(r.created_by))}</div><div><div class="who">${esc((r.created_by&&(r.created_by.name||r.created_by.email))||"Local Dev")} <span class="time">${timeAgo(r.created_at)}</span></div><div class="reply-body">${esc(r.body)}</div></div></div>`).join("")}</div>`:""}<div class="reply-box"><textarea placeholder="Reply..."></textarea><div class="actions"><button class="primary reply-send">Reply</button><button class="text-btn reply-cancel">Cancel</button></div></div><div class="actions thread-actions"><button class="reply-trigger">Reply...</button>${t.status==="live"?`<button class="icon-btn resolve" title="Resolve">✓</button>`:`<button class="text-btn reopen">Reopen</button>`}<button class="icon-btn danger delete" title="Delete">×</button></div></article>`}).join("");}
async function load(){const data=await api("/threads?status=all");render(data);}
function isEditing(){const active=document.activeElement;return quick.classList.contains("open")||!!document.querySelector(".reply-box.open")||(active&&active.tagName==="TEXTAREA");}
function resetAnchor(){anchor={kind:"general",target_path:location.pathname};}
function showError(message){quickError.textContent=message;quickError.classList.add("open");}
function clearError(){quickError.textContent="";quickError.classList.remove("open");}
function showQuick(x,y,text){clearError();const pad=12;quick.style.left=Math.max(pad,Math.min(x,stage.clientWidth-332))+"px";quick.style.top=Math.max(pad,Math.min(y,stage.clientHeight-184))+"px";dot.style.left=x+"px";dot.style.top=y+"px";quickLabel.textContent=text?`Comment on "${text.slice(0,80)}"`:"Comment on this spot";dot.classList.add("open");quick.classList.add("open");quickBody.focus();}
function hideQuick(){dot.classList.remove("open");quick.classList.remove("open");quickBody.value="";clearError();}
function capture(e){e.preventDefault();const doc=frame.contentDocument, win=frame.contentWindow, sel=doc.getSelection&&doc.getSelection();const text=sel?String(sel).trim():"";anchor={kind:text?"selection":"point",target_path:win.location.pathname,text_quote:{exact:text},viewport:{x:e.clientX,y:e.clientY,scroll_x:win.scrollX,scroll_y:win.scrollY,width:doc.documentElement.clientWidth,height:doc.documentElement.clientHeight}};showQuick(e.clientX,e.clientY,text);}
function wireFrame(){try{frame.contentDocument.addEventListener("contextmenu",capture);}catch(e){}}
function scrollAnchor(a){try{const w=frame.contentWindow;if(a&&a.viewport)w.scrollTo(a.viewport.scroll_x||0,a.viewport.scroll_y||0);}catch(e){}}
async function submit(bodyEl,fromQuick=false){const body=bodyEl.value.trim();if(!body){if(fromQuick)showError("Write a comment first.");return;}clearError();quickSubmit.disabled=true;try{await api("/threads",{method:"POST",body:JSON.stringify({body,anchor,excerpt:anchor.text_quote&&anchor.text_quote.exact||""})});bodyEl.value="";hideQuick();resetAnchor();await load();}catch(e){console.error(e);if(fromQuick)showError("Could not save comment. Refresh this collaboration page and try again.");}finally{quickSubmit.disabled=false;}}
threadsEl.addEventListener("click",async e=>{const card=e.target.closest(".thread");if(!card)return;const id=card.dataset.id;if(e.target.closest(".reply-trigger")){card.querySelector(".reply-box").classList.add("open");card.querySelector(".thread-actions").style.display="none";card.querySelector("textarea").focus();return;}if(e.target.closest(".reply-cancel")){card.querySelector(".reply-box").classList.remove("open");card.querySelector(".thread-actions").style.display="flex";return;}if(e.target.closest(".reply-send")){const ta=card.querySelector(".reply-box textarea");const body=ta.value.trim();if(body){await api(`/threads/${id}/replies`,{method:"POST",body:JSON.stringify({body})});await load();}return;}if(e.target.closest(".resolve")){await api(`/threads/${id}`,{method:"PATCH",body:JSON.stringify({status:"resolved"})});await load();return;}if(e.target.closest(".reopen")){await api(`/threads/${id}`,{method:"PATCH",body:JSON.stringify({status:"live"})});filter="live";document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.filter==="live"));await load();return;}if(e.target.closest(".delete")){await api(`/threads/${id}`,{method:"DELETE"});await load();return;}const data=await api("/threads?status=all");const t=(data.threads||[]).find(x=>x.thread_id===id);if(t)scrollAnchor(t.anchor);});
document.querySelectorAll(".tab").forEach(tab=>tab.onclick=()=>{filter=tab.dataset.filter;document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t===tab));load();});
byId("quick-submit").onclick=()=>submit(quickBody,true);byId("quick-cancel").onclick=hideQuick;byId("general-submit").onclick=()=>{anchor={kind:"general",target_path:location.pathname};submit(byId("general-body"));};
quickBody.addEventListener("keydown",e=>{if((e.metaKey||e.ctrlKey)&&e.key==="Enter")submit(quickBody,true);if(e.key==="Escape")hideQuick();});
frame.onload=wireFrame;wireFrame();load();setInterval(()=>{if(!document.hidden&&!isEditing())load().catch(()=>{})},3000);
</script>
</body>
</html>"""
    return (
        shell.replace("__TITLE__", title)
        .replace("__IFRAME_SRC__", iframe_src)
        .replace("__PUBLIC_URL__", public_url)
        .replace("__API_BASE__", api_base)
    )


@router.get("/collab/view/{user}/{report_id}", response_class=HTMLResponse)
async def collab_shell(user: str, report_id: str):
    artifact = _find_static_artifact(user, report_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return HTMLResponse(_shell(artifact, user, report_id))


@router.get("/collab/view/{user}/{report_id}/threads")
async def list_threads(user: str, report_id: str, status: str = "live", since: str | None = None):
    return {"threads": _list_threads(_artifact_key(user, report_id), status, since)}


@router.post("/collab/view/{user}/{report_id}/threads")
async def create_thread(user: str, report_id: str, body: _ThreadBody):
    text = body.body.strip()
    if not text:
        raise HTTPException(status_code=400, detail="body is required")
    key = _artifact_key(user, report_id)
    now = _now()
    thread_id = uuid.uuid4().hex
    thread = {
        "thread_id": thread_id,
        "status": "live",
        "anchor": body.anchor,
        "excerpt": body.excerpt[:500],
        "created_by": _viewer(),
        "created_at": now,
        "updated_at": now,
        "replies": [{
            "message_id": uuid.uuid4().hex,
            "body": text,
            "created_by": _viewer(),
            "created_at": now,
        }],
    }
    _THREADS.setdefault(key, {})[thread_id] = thread
    return JSONResponse({"thread": _thread_payload(thread)}, status_code=201)


@router.post("/collab/view/{user}/{report_id}/threads/{thread_id}/replies")
async def reply(user: str, report_id: str, thread_id: str, body: _ThreadBody):
    thread = _THREADS.get(_artifact_key(user, report_id), {}).get(thread_id)
    if not thread or thread.get("status") == "deleted":
        raise HTTPException(status_code=404, detail="thread not found")
    text = body.body.strip()
    if not text:
        raise HTTPException(status_code=400, detail="body is required")
    now = _now()
    thread.setdefault("replies", []).append({
        "message_id": uuid.uuid4().hex,
        "body": text,
        "created_by": _viewer(),
        "created_at": now,
    })
    thread["updated_at"] = now
    return JSONResponse({"thread": _thread_payload(thread)}, status_code=201)


@router.patch("/collab/view/{user}/{report_id}/threads/{thread_id}")
async def patch_thread(user: str, report_id: str, thread_id: str, body: _StatusBody):
    thread = _THREADS.get(_artifact_key(user, report_id), {}).get(thread_id)
    if not thread or thread.get("status") == "deleted":
        raise HTTPException(status_code=404, detail="thread not found")
    if body.status not in {"live", "resolved"}:
        raise HTTPException(status_code=400, detail="status must be live or resolved")
    thread["status"] = body.status
    thread["updated_at"] = _now()
    return {"thread": _thread_payload(thread)}


@router.delete("/collab/view/{user}/{report_id}/threads/{thread_id}")
async def delete_thread(user: str, report_id: str, thread_id: str):
    thread = _THREADS.get(_artifact_key(user, report_id), {}).get(thread_id)
    if not thread or thread.get("status") == "deleted":
        raise HTTPException(status_code=404, detail="thread not found")
    thread["status"] = "deleted"
    thread["deleted_at"] = _now()
    thread["updated_at"] = thread["deleted_at"]
    return {"ok": True}
