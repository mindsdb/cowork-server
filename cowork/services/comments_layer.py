"""On-artifact comment marker layer for the in-app preview iframe.

Ported from mindshub_services' ``comments_ui.py`` (the published-artifact
shell). There, the wrapper and the content iframe are same-origin, so the
wrapper injects the layer via ``contentDocument`` and the layer calls the
comments API directly. In Cowork the preview iframe is served by
cowork-server while the renderer lives on a different origin (``file://`` in
Electron), so:

  * we inject the layer here, at serve time, into the artifact's HTML — the
    only way to get a script into a cross-origin iframe; and
  * the layer is credential-free: it owns DOM concerns only (markers, hover
    highlight, CSS-path anchoring, on-artifact popovers) and exchanges data
    with the renderer purely over ``postMessage``. The renderer holds the
    MindsHub creds and performs every comments API call (see
    ``useArtifactComments`` + ``ArtifactCommentLayer`` on the client).

postMessage contract (both sides tag messages ``source:"anton-comments"``):
  layer -> parent : {type:"ready"}
                    {type:"create", selector, text}
                    {type:"reply", id, text}
                    {type:"status", id, status}
                    {type:"count", count}
                    {type:"mode", active}
  parent -> layer : {type:"list", comments:[...]}   // full normalized set
                    {type:"enter-mode"|"exit-mode"}
                    {type:"focus", commentId}
                    {type:"hl-on"|"hl-off", commentId}

The normalized comment shape the layer renders (parent produces it):
  {id, selector, status, author, text, created_at (epoch s),
   replies:[{author, text, created_at}]}
"""

import re

# The layer only arms itself when the renderer opts in via this query flag on
# the iframe's entry-document URL, so ordinary previews are untouched.
ACTIVATION_PARAM = "__antonComments"

LAYER_JS = r"""
(function(){
  if (window.__antonCommentsLayer) return;
  window.__antonCommentsLayer = true;
  var mode = false, comments = [], hoverEl = null, pop = null;

  var css = document.createElement('style');
  css.textContent = ""
    // The layer is injected INTO the artifact document, so the page's own
    // global rules (e.g. `button{padding:10px 18px}`) bleed into our controls
    // and can crush/restyle them. Neutralize the risky inherited/box props on
    // everything we render; !important beats host rules regardless of specificity.
    + ".ac-pop,.ac-pop *,.ac-marker{box-sizing:border-box!important;}"
    + ".ac-pop button,.ac-pop input,.ac-pop textarea{margin:0!important;padding:0;"
    + "font-family:inherit;line-height:normal!important;min-width:0!important;min-height:0!important;}"
    + ".ac-pop svg,.ac-marker svg{flex:0 0 auto!important;}"
    + ".ac-hl{position:absolute;pointer-events:none;border:2px solid #00e5ff;"
    + "background:rgba(0,229,255,0.08);z-index:2147483640;border-radius:3px;}"
    + "@keyframes ac-popIn{0%{opacity:0;transform:translate(-50%,-100%) scale(.6)}"
    + "100%{opacity:1;transform:translate(-50%,-100%) scale(1)}}"
    + "@keyframes ac-popIn-pop{0%{opacity:0;transform:scale(.94)}100%{opacity:1;transform:scale(1)}}"
    + ".ac-marker{position:absolute;z-index:2147483641;width:28px;height:28px;"
    + "border-radius:50% 50% 50% 3px;background:#00e5ff;border:2px solid #fff;"
    + "cursor:pointer;transform:translate(-50%,-100%);"
    + "display:flex;align-items:center;justify-content:center;"
    + "color:#04222a;font:800 12px -apple-system,monospace;"
    + "box-shadow:0 4px 14px rgba(0,0,0,.55),0 0 0 5px rgba(34,211,238,.30);"
    + "transition:box-shadow .12s ease,transform .12s ease;"
    + "animation:ac-popIn .3s ease;}"
    + ".ac-marker:hover{box-shadow:0 6px 18px rgba(0,0,0,.6),0 0 0 7px rgba(34,211,238,.5);"
    + "transform:translate(-50%,-100%) scale(1.1);}"
    + ".ac-pop{position:absolute;z-index:2147483642;width:max-content;"
    + "min-width:300px;max-width:420px;background:#0c141d;"
    + "border:1.5px solid rgba(0,229,255,0.45);border-radius:12px;padding:14px;"
    + "box-shadow:0 16px 50px rgba(0,0,0,.5),0 0 0 1px rgba(0,229,255,0.06);"
    + "font:13px 'Inter',-apple-system,sans-serif;color:#e7eef3;animation:ac-popIn-pop .18s ease;}"
    + ".ac-pop .ac-top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;}"
    + ".ac-pop .ac-author{display:flex;align-items:flex-start;gap:10px;min-width:0;}"
    + ".ac-pop .ac-avatar{flex-shrink:0;width:32px;height:32px;border-radius:999px;display:flex;"
    + "align-items:center;justify-content:center;color:#0a0a0f;font-size:12px;font-weight:700;}"
    + ".ac-pop .ac-name{font-size:13px;font-weight:700;color:#fff;line-height:1.15;}"
    + ".ac-pop .ac-time{font-size:11px;color:#6c7a87;margin-top:2px;}"
    + ".ac-pop .ac-ctext{font-size:13px;color:#e7eef3;line-height:1.45;margin-top:12px;"
    + "white-space:pre-wrap;word-break:break-word;}"
    + ".ac-pop .ac-actions{display:flex;align-items:center;gap:6px;flex-shrink:0;}"
    + ".ac-pop .ac-replies{margin-top:14px;display:flex;flex-direction:column;gap:12px;"
    + "max-height:220px;overflow:auto;}"
    + ".ac-pop .ac-reply{display:flex;gap:10px;align-items:flex-start;}"
    + ".ac-pop .ac-reply-avatar{flex-shrink:0;width:24px;height:24px;border-radius:999px;display:flex;"
    + "align-items:center;justify-content:center;color:#0a0a0f;font-size:10px;font-weight:700;}"
    + ".ac-pop .ac-reply-body{min-width:0;}"
    + ".ac-pop .ac-reply-head{font-size:12px;color:#6c7a87;}"
    + ".ac-pop .ac-reply-name{font-weight:700;color:#e7eef3;}"
    + ".ac-pop .ac-reply-txt{font-size:12px;color:#e7eef3;margin-top:2px;white-space:pre-wrap;"
    + "word-break:break-word;}"
    + ".ac-pop .ac-pill{display:inline-flex;align-items:center;gap:5px;height:26px;padding:0 10px;"
    + "border-radius:999px;border:1px solid #2a3a48;background:transparent;color:#9fb0bd;"
    + "font:600 11px 'Inter',sans-serif;cursor:pointer;white-space:nowrap;"
    + "transition:color .15s,background .15s,border-color .15s;}"
    + ".ac-pop .ac-pill:hover{color:#c7d3dd;border-color:#3a4a58;}"
    + ".ac-pop .ac-pill.resolved{border-color:rgba(74,201,126,0.35);"
    + "background:rgba(74,201,126,0.12);color:#66d39a;}"
    + ".ac-pop .ac-pill svg{width:13px;height:13px;}"
    + ".ac-pop .ac-badge{display:inline-flex;align-items:center;height:22px;padding:0 9px;"
    + "border-radius:999px;font:600 11px 'Inter',sans-serif;color:#9fb0bd;background:#16202b;"
    + "border:1px solid #2a3a48;}"
    + ".ac-pop .ac-reply-row{display:flex;gap:8px;align-items:center;margin-top:14px;}"
    + ".ac-pop .ac-reply-input{flex:1;min-width:0;height:34px;padding:0 12px;background:#0b121a;"
    + "border:1px solid #283845;border-radius:8px;color:#e7eef3;font:13px 'Inter',sans-serif;"
    + "outline:none;transition:border-color .15s;}"
    + ".ac-pop .ac-reply-input:focus{border-color:#00e5ff;}"
    + ".ac-pop .ac-reply-input::placeholder{color:#5b6b78;}"
    + ".ac-pop .ac-icon-btn{flex-shrink:0;width:34px;height:34px;border-radius:8px;"
    + "background:transparent;display:flex;align-items:center;justify-content:center;cursor:pointer;"
    + "transition:color .15s,background .15s,border-color .15s;}"
    + ".ac-pop .ac-icon-btn.ac-send{border:1px solid #2a3a48;color:#cfdae2;}"
    + ".ac-pop .ac-icon-btn.ac-send:hover{border-color:#00e5ff;color:#00e5ff;}"
    + ".ac-pop .ac-icon-btn.ac-cancel{border:1px solid #4a2530;color:#e0556a;}"
    + ".ac-pop .ac-icon-btn.ac-cancel:hover{background:rgba(224,85,106,0.12);}"
    + ".ac-pop .ac-icon-btn svg{width:16px;height:16px;}"
    + ".ac-pop textarea{width:100%;min-height:60px;background:#0b121a;border:1px solid #283845;"
    + "border-radius:8px;color:#e7eef3;padding:9px 12px;font:13px 'Inter',sans-serif;"
    + "resize:none;outline:none;box-sizing:border-box;line-height:1.45;transition:border-color .15s;}"
    + ".ac-pop textarea:focus{border-color:#00e5ff;}"
    + ".ac-pop textarea::placeholder{color:#5b6b78;}"
    + ".ac-pop .ac-row{display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:nowrap;}"
    + ".ac-pop button.ac-text-btn{font:600 12px 'Inter',sans-serif;padding:0 13px;height:30px;"
    + "white-space:nowrap;border-radius:8px;cursor:pointer;border:1px solid #2a3a48;"
    + "background:transparent;color:#9fb0bd;}"
    + ".ac-pop button.ac-text-btn:hover{color:#c7d3dd;border-color:#3a4a58;}"
    + ".ac-pop button.ac-primary{border:none;background:#00e5ff;color:#04121a;}"
    + ".ac-pop button.ac-primary:hover{background:#33eaff;}"
    + ".ac-pop button.ac-primary:disabled{opacity:.45;cursor:default;}"
    + ".ac-hint{font-size:10.5px;color:#6c7a87;margin-right:auto;}"
    + "body.ac-mode, body.ac-mode *{cursor:crosshair !important;}";
  document.head.appendChild(css);

  var hl = document.createElement('div');
  hl.className = 'ac-hl'; hl.style.display = 'none';
  document.body.appendChild(hl);

  // Cross-origin: the parent (renderer) lives on a different origin than this
  // iframe, so we can't target a specific origin — post to '*' and let both
  // sides gate on the source tag.
  function send(msg){ msg.source = 'anton-comments'; try { parent.postMessage(msg, '*'); } catch (e) {} }
  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
  function displayName(a){ a = String(a || ''); var i = a.indexOf('@');
    return i > 0 ? a.slice(0, i) : (a || 'Anonymous'); }
  function initials(a){
    var parts = displayName(a).split(/[^A-Za-z0-9]+/).filter(Boolean);
    if (!parts.length) return '?';
    return (parts[0].charAt(0) + (parts.length > 1 ? parts[1].charAt(0) : '')).toUpperCase();
  }
  var AVATAR_COLORS = ['#d99a1c','#1a8596','#5f8ad9','#c46fb0','#5fb87a','#d97a5f','#8f7fd9'];
  function avatarColor(a){ a = String(a || ''); var h = 0;
    for (var i = 0; i < a.length; i++) h = (h * 31 + a.charCodeAt(i)) >>> 0;
    return AVATAR_COLORS[h % AVATAR_COLORS.length]; }
  function timeAgo(ts){
    if (!ts) return '';
    var diff = Math.floor(Date.now() / 1000) - ts;
    if (diff < 45) return 'now';
    if (diff < 3600) return Math.round(diff / 60) + 'm';
    if (diff < 86400) return Math.round(diff / 3600) + 'h';
    if (diff < 604800) return Math.round(diff / 86400) + 'd';
    try { return new Date(ts * 1000).toLocaleDateString(); } catch (e) { return ''; }
  }
  var CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    + 'stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14">'
    + '</path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
  var SEND_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    + 'stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12">'
    + '</polyline></svg>';
  var CANCEL_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
    + 'stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line>'
    + '<line x1="6" y1="6" x2="18" y2="18"></line></svg>';

  function cssPath(el){
    if (!(el instanceof Element)) return null;
    if (el.id) return '#' + CSS.escape(el.id);
    var parts = [];
    while (el && el.nodeType === 1 && el !== document.body) {
      var sel = el.nodeName.toLowerCase();
      var parent = el.parentNode;
      if (!parent) break;
      if (el.id) { parts.unshift('#' + CSS.escape(el.id)); break; }
      var sibs = Array.prototype.filter.call(parent.children, function(c){
        return c.nodeName === el.nodeName; });
      if (sibs.length > 1) sel += ':nth-of-type(' + (sibs.indexOf(el) + 1) + ')';
      parts.unshift(sel);
      el = parent;
    }
    return parts.length ? parts.join(' > ') : null;
  }
  function rectOf(el){
    var r = el.getBoundingClientRect();
    return {top: r.top + window.scrollY, left: r.left + window.scrollX,
            right: r.right + window.scrollX, width: r.width, height: r.height};
  }

  function isClosed(c){ return c.status === 'resolved' || c.status === 'dismissed'; }

  function clearMarkers(){
    Array.prototype.forEach.call(document.querySelectorAll('.ac-marker'),
      function(m){ m.remove(); });
  }
  function placeMarkers(){
    clearMarkers();
    comments.forEach(function(c){
      if (!c.selector || isClosed(c)) return;
      var el; try { el = document.querySelector(c.selector); } catch (e) { el = null; }
      if (!el) return;
      var r = rectOf(el);
      var m = document.createElement('div');
      m.className = 'ac-marker';
      m.style.top = r.top + 'px'; m.style.left = r.right + 'px';
      m.textContent = String((c.replies || []).length + 1);
      m.addEventListener('click', function(ev){ ev.stopPropagation(); openThread(c, r); });
      document.body.appendChild(m);
    });
  }

  var markersRAF = 0;
  function scheduleMarkers(){
    if (markersRAF) return;
    markersRAF = requestAnimationFrame(function(){ markersRAF = 0; placeMarkers(); });
  }

  function closePop(){ if (pop) { pop.remove(); pop = null; } }
  function popAt(top, left){
    closePop();
    pop = document.createElement('div');
    pop.className = 'ac-pop';
    var maxLeft = window.scrollX + document.documentElement.clientWidth - 430;
    pop.style.top = top + 'px';
    pop.style.left = Math.max(window.scrollX + 4, Math.min(left, maxLeft)) + 'px';
    document.body.appendChild(pop);
    return pop;
  }
  function popUnderMarker(r){ return popAt(r.top + 8, r.right - 20); }

  function openAdd(el, r){
    setMode(false);
    var selector = cssPath(el);
    var p = popUnderMarker(r);
    p.innerHTML = '<textarea placeholder="Add a comment…"></textarea>'
      + '<div class="ac-row"><span class="ac-hint">⌘↵ to comment</span>'
      + '<button class="ac-text-btn ac-cancel">Cancel</button>'
      + '<button class="ac-text-btn ac-primary ac-save" disabled>Comment</button></div>';
    var ta = p.querySelector('textarea');
    var saveBtn = p.querySelector('.ac-save');
    ta.addEventListener('input', function(){ saveBtn.disabled = !ta.value.trim(); });
    ta.addEventListener('keydown', function(e){
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); saveBtn.click(); }
    });
    ta.focus();
    p.querySelector('.ac-cancel').onclick = closePop;
    saveBtn.onclick = function(){
      var text = ta.value.trim(); if (!text) return;
      send({type: 'create', selector: selector, text: text});
      closePop();
    };
  }

  function openThread(c, r){
    var p = popUnderMarker(r);
    var replies = (c.replies || []).map(function(rep){
      return '<div class="ac-reply">'
        + '<span class="ac-reply-avatar" style="background:' + avatarColor(rep.author) + '">'
        + esc(initials(rep.author)) + '</span>'
        + '<div class="ac-reply-body"><div class="ac-reply-head">'
        + '<span class="ac-reply-name">' + esc(displayName(rep.author)) + '</span> '
        + esc(timeAgo(rep.created_at)) + '</div>'
        + '<div class="ac-reply-txt">' + esc(rep.text) + '</div></div></div>';
    }).join('');
    var repliesHtml = replies ? '<div class="ac-replies">' + replies + '</div>' : '';
    var actions;
    if (c.status === 'resolved') {
      actions = '<button class="ac-pill ac-status resolved" data-to="open">'
        + CHECK_SVG + '<span>Resolved</span></button>';
    } else if (c.status === 'dismissed') {
      actions = '<span class="ac-badge">Dismissed</span>'
        + '<button class="ac-pill ac-status" data-to="open">Reopen</button>';
    } else {
      actions = '<button class="ac-pill ac-status" data-to="resolved">'
        + CHECK_SVG + '<span>Resolve</span></button>'
        + '<button class="ac-pill ac-status" data-to="dismissed">Dismiss</button>';
    }
    p.innerHTML = '<div class="ac-top"><div class="ac-author">'
      + '<span class="ac-avatar" style="background:' + avatarColor(c.author) + '">'
      + esc(initials(c.author)) + '</span>'
      + '<div style="min-width:0;"><div class="ac-name">' + esc(displayName(c.author)) + '</div>'
      + '<div class="ac-time">' + esc(timeAgo(c.created_at)) + '</div></div></div>'
      + '<div class="ac-actions">' + actions + '</div></div>'
      + '<div class="ac-ctext">' + esc(c.text) + '</div>'
      + repliesHtml
      + '<div class="ac-reply-row">'
      + '<input class="ac-reply-input" type="text" placeholder="Reply...">'
      + '<button class="ac-icon-btn ac-send" title="Send reply">' + SEND_SVG + '</button>'
      + '<button class="ac-icon-btn ac-cancel" title="Close">' + CANCEL_SVG + '</button></div>';
    var input = p.querySelector('.ac-reply-input');
    p.querySelector('.ac-cancel').onclick = closePop;
    Array.prototype.forEach.call(p.querySelectorAll('.ac-status'), function(btn){
      btn.onclick = function(){ send({type: 'status', id: c.id, status: btn.getAttribute('data-to')}); closePop(); };
    });
    function submit(){
      var text = (input.value || '').trim(); if (!text) return;
      input.disabled = true;
      send({type: 'reply', id: c.id, text: text});
      closePop();
    }
    p.querySelector('.ac-send').onclick = submit;
    input.addEventListener('keydown', function(e){
      if (e.key === 'Enter') { e.preventDefault(); submit(); } });
    input.focus();
  }

  function setMode(on){
    mode = on;
    document.body.classList.toggle('ac-mode', on);
    if (!on) hl.style.display = 'none';
    send({type:'mode', active: on});
  }

  document.addEventListener('mousemove', function(e){
    if (!mode) return;
    var el = e.target;
    if (!el || el === hl || (el.closest && el.closest('.ac-marker'))) return;
    if (el.closest && el.closest('.ac-pop')) return;
    hoverEl = el;
    var r = rectOf(el);
    hl.style.display = 'block';
    hl.style.top = r.top + 'px'; hl.style.left = r.left + 'px';
    hl.style.width = r.width + 'px'; hl.style.height = r.height + 'px';
  }, true);

  document.addEventListener('click', function(e){
    if (!mode) return;
    if (e.target.closest && (e.target.closest('.ac-pop') || e.target.closest('.ac-marker'))) return;
    e.preventDefault(); e.stopPropagation();
    var el = hoverEl || e.target;
    openAdd(el, rectOf(el));
  }, true);

  document.addEventListener('click', function(e){
    if (!pop) return;
    var t = e.target;
    if (t.closest && (t.closest('.ac-pop') || t.closest('.ac-marker'))) return;
    closePop();
  });

  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape') { if (pop) closePop(); else if (mode) setMode(false); }
  });
  window.addEventListener('scroll', scheduleMarkers, true);
  window.addEventListener('resize', scheduleMarkers);

  function focusComment(id, openThreadToo){
    var c = comments.filter(function(x){ return x.id === id; })[0];
    if (!c || !c.selector) return;
    var el; try { el = document.querySelector(c.selector); } catch (e) { el = null; }
    if (!el) return;
    el.scrollIntoView({behavior:'smooth', block:'center'});
    var r = rectOf(el);
    hl.style.display = 'block';
    hl.style.top = r.top + 'px'; hl.style.left = r.left + 'px';
    hl.style.width = r.width + 'px'; hl.style.height = r.height + 'px';
    setTimeout(function(){ hl.style.display = 'none'; }, 1200);
    if (openThreadToo) openThread(c, r);
  }

  window.addEventListener('message', function(ev){
    var d = ev.data || {};
    if (d.source !== 'anton-comments') return;
    if (d.type === 'list') {
      comments = Array.isArray(d.comments) ? d.comments : [];
      placeMarkers();
      send({type:'count', count: comments.length});
    }
    else if (d.type === 'enter-mode') setMode(true);
    else if (d.type === 'exit-mode') setMode(false);
    else if (d.type === 'focus') focusComment(d.commentId, true);
    else if (d.type === 'hl-on') {
      if (mode) return;
      var hc = comments.filter(function(x){ return x.id === d.commentId; })[0];
      if (!hc || !hc.selector) { hl.style.display = 'none'; return; }
      var hel; try { hel = document.querySelector(hc.selector); } catch (e) { hel = null; }
      if (!hel) { hl.style.display = 'none'; return; }
      var hr = rectOf(hel);
      hl.style.display = 'block';
      hl.style.top = hr.top + 'px'; hl.style.left = hr.left + 'px';
      hl.style.width = hr.width + 'px'; hl.style.height = hr.height + 'px';
    }
    else if (d.type === 'hl-off') { if (!mode) hl.style.display = 'none'; }
  });

  // Announce readiness so the parent pushes the current comment list.
  send({type:'ready'});
})();
"""


# Precomputed once: the JS is a constant blob, and a literal ``</script>`` in it
# would break out of the injected tag.
_SCRIPT_TAG = "<script>%s</script>" % LAYER_JS.replace("</script>", "<\\/script>")
# Case-insensitive so we anchor to the real body close in mixed-case documents,
# without allocating a full lowercased copy of the (possibly large) HTML.
_BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)


def inject_layer(html: str) -> str:
    """Append the comment marker layer to an artifact's HTML document.

    Inserts before the last ``</body>`` when present (so the layer's absolutely
    positioned nodes live inside the body), else falls back to appending.
    """
    matches = list(_BODY_CLOSE_RE.finditer(html))
    if matches:
        idx = matches[-1].start()
        return html[:idx] + _SCRIPT_TAG + html[idx:]
    return html + _SCRIPT_TAG
