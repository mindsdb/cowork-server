/*
 * In-frame preview shim. Injected as the first script of every HTML artifact
 * preview (see preview_shim.py).
 *
 * Two independent halves:
 *   1. An error reporter, installed ALWAYS. It is the only way a page's own
 *      failure reaches the user or the agent — the viewer never sees the
 *      frame's console. Report types: error (uncaught), resource (a file
 *      that failed to load), csp (a policy violation), rejection (an
 *      unhandled promise rejection), console (console.error with an Error or
 *      "XxxError:" text), http (a fetch or XHR answered 5xx).
 *   2. Storage substitutes, installed ONLY when the native ones throw. The
 *      preview frame runs at an opaque origin, where localStorage,
 *      sessionStorage, document.cookie, indexedDB.open(), navigator.
 *      serviceWorker and caches all raise SecurityError. A generated page
 *      typically reads storage during top-level initialisation, so the throw
 *      aborts its script before any listener is bound and every control on
 *      the rendered page is dead.
 *
 * Verified by hand against a real browser. tests/test_preview_shim.py runs
 * the reporters in node against stand-in browser objects, which proves the
 * wiring but not a real sandbox, so re-check manually after any edit here.
 */
(function () {
  'use strict';
  // Captured once, before the page's own script runs: unlike window.top, an
  // assignment to window.parent sticks, so a page could otherwise repoint it
  // later and silently swallow every diagnostic this shim reports.
  var PARENT = window.parent, IS_TOP = window.parent === window;
  var TAG = 'anton-preview';
  var LINE_OFFSET = __LINE_OFFSET__;
  var MAX_REPORTS = 20;
  /* What the new report types carry at most. The app and the server cut a
     message to 300 characters again; this only keeps one huge console line
     or error body from crossing postMessage whole. */
  var MAX_TEXT = 1000;
  var MAX_BODY = 300;
  var seen = Object.create(null);
  var sent = 0;
  /* A report names what the agent can open, and never a draft link's token.
     On the draft host the document's address starts with /_t/v1.<token>/, so
     that segment goes, with or without its origin (a page's own message can
     carry location.pathname). Anywhere else the document's own folder goes (a
     desktop draft's loopback path), and a /_t/ folder inside an artifact is
     just a folder. about:srcdoc has no folder, and nothing is cut there. */
  var HREF = String(window.location.href);
  /* The folder comes from the path alone: a query or hash such as
     ?next=/a or #/route would otherwise move it. */
  var PATH_ONLY = HREF.split(/[?#]/)[0];
  var BASE = PATH_ONLY.slice(0, PATH_ONLY.lastIndexOf('/') + 1);
  var TOKEN_PATH = /^[a-z][a-z0-9+.-]*:\/\/[^\/?#]+\/_t\/v1\.[^\/?#]+\//.test(HREF) ?
    /(?:https?:\/\/[^\/\s]+)?\/_t\/v1\.[^\/?#\s]+\//g : null;

  function local(value) {
    var text = String(value || '');
    if (TOKEN_PATH) return text.replace(TOKEN_PATH, '');
    if (!BASE) return text;
    return text.split(BASE).join('');
  }

  function post(payload) {
    try {
      if (IS_TOP) return;
      payload.source = TAG;
      PARENT.postMessage(payload, '*');
    } catch (e) { /* frame detached mid-report */ }
  }

  /* One report per key for the life of the document: a page that fails the
     same way in a loop (a polling fetch, a render that logs each frame)
     sends it once, and MAX_REPORTS bounds the distinct ones. */
  function report(key, payload) {
    if (sent >= MAX_REPORTS || seen[key]) return;
    seen[key] = true;
    sent += 1;
    post(payload);
  }

  function clip(value, limit) {
    return String(value == null ? '' : value).replace(/\s+/g, ' ').trim().slice(0, limit);
  }

  /* Cross-realm safe: an Error from another frame fails instanceof. */
  function isError(value) {
    if (value instanceof Error) return true;
    return !!value && typeof value === 'object' &&
      typeof value.message === 'string' && typeof value.stack === 'string';
  }

  function describe(value) {
    if (isError(value)) return (value.name || 'Error') + ': ' + value.message;
    if (typeof value === 'string') return value;
    try { return String(value); } catch (e) { return ''; }
  }

  /* console.error is the page's own opinion that something broke. Report it
     only when it carries an Error, or text shaped like one ("TypeError: x");
     framework chatter (React's warnings, Chrome's [Violation] notes) is not
     breakage and would bury the one report that matters. */
  var ERROR_TEXT = /\b[A-Z][A-Za-z0-9]*(Error|Exception):/;
  var CONSOLE_NOISE = /\[violation\]|^\s*warning:|download the react devtools/i;

  function reportConsole(args) {
    var error = null, parts = [];
    for (var i = 0; i < args.length; i++) {
      if (!error && isError(args[i])) error = args[i];
      parts.push(describe(args[i]));
    }
    var text = parts.join(' ');
    if (CONSOLE_NOISE.test(text)) return;
    if (!error && !ERROR_TEXT.test(text)) return;
    var message = clip(local(text), MAX_TEXT);
    if (!message) return;
    report('k|' + message, {
      type: 'console',
      message: message,
      file: '',
      line: 0,
      stack: error && error.stack ? local(error.stack) : ''
    });
  }

  /* A server error a page asked for. The page may never surface it (a
     dashboard that shows an empty chart on a 500), so it is reported here,
     with the start of the body the server sent: usually the traceback line
     that says what broke. */
  function reportHttp(method, url, status, body) {
    var where = local(url);
    var verb = String(method || 'GET').toUpperCase();
    report('h|' + verb + '|' + where.split('?')[0] + '|' + status, {
      type: 'http',
      message: clip(verb + ' ' + where + ' \u2192 ' + status + ': ' + clip(body, MAX_BODY), MAX_TEXT),
      file: '',
      line: 0,
      url: where,
      method: verb,
      status: status
    });
  }

  post({ type: 'document-start' });

  try {
    /* capture = true: resource failures (<script src>, <link>, <img>) do not
       bubble to window, so a plain listener never sees them. They are the
       direct diagnosis for a CDN blocked by the shell's CSP. A load the
       policy blocks fires here AND in the securitypolicyviolation listener
       below, so one blocked file arrives at the renderer as two reports. */
    window.addEventListener('error', function (ev) {
      try {
        if (ev instanceof ErrorEvent) {
          var file = ev.filename || '';
          var line = typeof ev.lineno === 'number' ? ev.lineno : 0;
          if (file === HREF) {
            /* Only this document's own inline scripts shifted; an external
               file carries its own coordinates. lineno is 0 for "Script
               error." and for some parse failures, hence the guard. */
            if (line > LINE_OFFSET) line -= LINE_OFFSET;
            /* The document's own address names nothing the agent can open —
               about:srcdoc on web, a signed draft URL with a query on
               desktop and on the draft host. An empty file tells the server
               to print the artifact's source path instead. */
            file = '';
          } else if (file) {
            /* An external <script src> under the same document (e.g. a
               desktop draft's "static/app.js") still carries the full
               address, which names nothing the agent can open either. What
               is left after the cut is the path the agent actually edits. */
            file = local(file);
          }
          report('e|' + ev.message + '|' + file + '|' + line, {
            type: 'error',
            message: local(ev.message),
            file: file,
            line: line,
            col: typeof ev.colno === 'number' ? ev.colno : 0,
            stack: ev.error && ev.error.stack ? local(ev.error.stack) : ''
          });
          return;
        }
        var node = ev.target || {};
        var tagName = node.tagName || '';
        var url = local(node.src || node.href || '');
        report('r|' + tagName + '|' + url, {
          type: 'resource',
          tagName: String(tagName),
          url: url
        });
      } catch (inner) { /* a reporter must never break the page */ }
    }, true);

    window.addEventListener('unhandledrejection', function (ev) {
      try {
        var reason = ev.reason;
        var message = reason && reason.message ? reason.message : String(reason);
        report('p|' + message, {
          type: 'rejection',
          message: 'Unhandled promise rejection: ' + local(message),
          file: '',
          line: 0,
          col: 0,
          stack: reason && reason.stack ? local(reason.stack) : ''
        });
      } catch (inner) { /* a reporter must never break the page */ }
    });

    /* CSP3 dispatches this on the element or the document; document is the
       specified subscription point. */
    document.addEventListener('securitypolicyviolation', function (ev) {
      try {
        report('c|' + ev.violatedDirective + '|' + ev.blockedURI, {
          type: 'csp',
          violatedDirective: String(ev.violatedDirective || ''),
          blockedURI: local(ev.blockedURI)
        });
      } catch (inner) { /* a reporter must never break the page */ }
    });
  } catch (e) { /* no listeners available: the page still runs */ }

  /* The wrappers below call the page's own function first-class and only
     look at what came back: a page must behave exactly as without them. */
  try {
    var CONSOLE = window.console;
    var nativeConsoleError = CONSOLE && CONSOLE.error;
    if (typeof nativeConsoleError === 'function') {
      CONSOLE.error = function () {
        try { reportConsole(arguments); } catch (inner) { /* never break the page */ }
        return nativeConsoleError.apply(this, arguments);
      };
    }
  } catch (e) { /* console not writable: no console reports */ }

  try {
    var nativeFetch = window.fetch;
    if (typeof nativeFetch === 'function') {
      window.fetch = function (input, init) {
        var method = 'GET', url = '';
        try {
          method = (init && init.method) || (input && typeof input === 'object' && input.method) || 'GET';
          url = typeof input === 'string' ? input : (input && input.url) || String(input);
        } catch (inner) { /* an exotic input: report with what we have */ }
        /* The page gets the chained promise: it settles exactly as the native
           one does, and the page's own handlers run after ours, so the clone
           below is taken before the page reads the body. A rejection passes
           through untouched, unhandled where the page leaves it unhandled. */
        return nativeFetch.apply(this, arguments).then(function (response) {
          try {
            if (response && response.status >= 500) {
              var status = response.status, where = response.url || url;
              response.clone().text().then(function (body) {
                reportHttp(method, where, status, body);
              }, function () {
                reportHttp(method, where, status, '');
              });
            }
          } catch (inner) { /* never break the page */ }
          return response;
        });
      };
    }
  } catch (e) { /* fetch not writable: no fetch reports */ }

  try {
    var XHR = window.XMLHttpRequest;
    if (XHR && XHR.prototype && typeof XHR.prototype.open === 'function') {
      var nativeOpen = XHR.prototype.open, nativeSend = XHR.prototype.send;
      var requests = typeof WeakMap === 'function' ? new WeakMap() : null;
      XHR.prototype.open = function (method, url) {
        try { if (requests) requests.set(this, [method, url]); } catch (inner) { /* never break the page */ }
        return nativeOpen.apply(this, arguments);
      };
      XHR.prototype.send = function () {
        try {
          var xhr = this;
          xhr.addEventListener('loadend', function () {
            try {
              if (!(xhr.status >= 500)) return;
              var request = (requests && requests.get(xhr)) || ['GET', ''];
              var body = '';
              /* responseText throws for any other responseType. */
              try {
                if (!xhr.responseType || xhr.responseType === 'text') body = xhr.responseText;
              } catch (inner) { /* no text body */ }
              reportHttp(request[0], xhr.responseURL || String(request[1] || ''), xhr.status, body);
            } catch (inner) { /* never break the page */ }
          });
        } catch (inner) { /* never break the page */ }
        return nativeSend.apply(this, arguments);
      };
    }
  } catch (e) { /* XMLHttpRequest not patchable: no XHR reports */ }

  /* View reporter. The app's inline editor is a separate frame built from the
     source, so it opens on the first slide or the top of the page. This tells
     the app where the user is in the preview, so the editor can open there
     too: the hash, the window's scroll and the last scrolled container, the
     user's own navigation inputs since load (to replay in a deck that keeps
     its current slide in script state), and the heading on screen (to check
     the replay landed on the same view).

     Only trusted inputs count, and only ones the page reacted to: a key or
     click is kept when the DOM changed within SETTLE_MS after it. That keeps
     scrolling with the arrow keys and clicks on plain text out of the list.
     Inputs land on the page's own elements, named by a path of tag:index
     steps from <body> (or from the nearest element with an id). Every value
     is capped; the app validates them again, as untrusted data. */
  try {
    var NAV_KEYS = {
      ArrowRight: 1, ArrowLeft: 1, ArrowUp: 1, ArrowDown: 1,
      PageDown: 1, PageUp: 1, Home: 1, End: 1, ' ': 1
    };
    /* The same list as the editor's (htmlVisualEditorRuntime.js INTERACTIVE). */
    var CONTROL = 'a[href], button, summary, label, select, [role="button"], [role="tab"], ' +
      '[role="link"], [role="menuitem"], [role="option"], [onclick]';
    var FIELD = 'input, textarea, select, [contenteditable]:not([contenteditable="false"])';
    /* The comment layer's own UI (comments_layer.py). */
    var LAYER_UI = '#act-mk, .act-hl, .act-pop, .act-menu';
    var MAX_INPUTS = 50, MAX_STEPS = 32, MAX_PATH = 300, MAX_HASH = 512, MAX_HEADING = 120;
    var MAX_LABEL = 80, SETTLE_MS = 250;
    var ID_RE = /^[A-Za-z][\w.:-]{0,99}$/;
    var viewInputs = [], viewOverflow = false, scrolled = null, lastView = '';
    /* True while a restore (below) replays a view: no report goes out and the
       user's own input waits, so the page settles where the app asked. */
    var replaying = false;
    var viewTimers = [], scrollTimer = 0;

    var inLayer = function (el) {
      return !!(el && el.closest && el.closest(LAYER_UI));
    };
    var isLayer = function (el) {
      return !!(el && el.matches && el.matches(LAYER_UI));
    };

    /* tag:index steps up to <body>, the index counting same-tag siblings only
       (the comment layer's elements skipped), so the editor's copy of the
       page resolves the same element. An element with a unique id anchors
       the path. '' for anything outside <body>. */
    var pathOf = function (el) {
      var steps = [];
      while (el && el.nodeType === 1 && el !== document.body) {
        if (steps.length >= MAX_STEPS) return '';
        var id = el.id;
        if (id && ID_RE.test(id) && document.getElementById(id) === el) {
          steps.unshift('#' + id);
          break;
        }
        var tag = el.localName, index = 0;
        for (var s = el.previousElementSibling; s; s = s.previousElementSibling) {
          if (s.localName === tag && !isLayer(s)) index += 1;
        }
        steps.unshift(tag + ':' + index);
        el = el.parentElement;
      }
      if (!el || (el !== document.body && steps[0].charAt(0) !== '#')) return '';
      var path = steps.join('/');
      return path.length <= MAX_PATH ? path : '';
    };

    /* The topmost heading on screen. The editor runs the same rule
       (htmlVisualEditorRuntime.js visibleHeading). */
    var visibleHeading = function () {
      var best = null, bestTop = 0, bestLeft = 0;
      var width = window.innerWidth, height = window.innerHeight;
      var nodes = document.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"]');
      for (var i = 0; i < nodes.length && i < 500; i++) {
        var node = nodes[i];
        if (inLayer(node)) continue;
        var r = node.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0 || r.bottom <= 0 || r.right <= 0 ||
          r.top >= height || r.left >= width) continue;
        if (node.checkVisibility &&
          !node.checkVisibility({ opacityProperty: true, visibilityProperty: true })) continue;
        if (!best || r.top < bestTop - 1 || (Math.abs(r.top - bestTop) <= 1 && r.left < bestLeft)) {
          best = node; bestTop = r.top; bestLeft = r.left;
        }
      }
      return best ? clip(best.textContent, MAX_HEADING) : '';
    };

    var sendView = function () {
      if (replaying) return;
      var view = {
        type: 'view',
        hash: String(window.location.hash || '').slice(0, MAX_HASH),
        scrollX: Math.round(window.scrollX || 0),
        scrollY: Math.round(window.scrollY || 0),
        container: null,
        inputs: viewOverflow ? [] : viewInputs.slice(),
        overflow: viewOverflow,
        heading: visibleHeading()
      };
      if (scrolled && scrolled.isConnected) {
        var where = pathOf(scrolled);
        if (where) {
          view.container = {
            path: where,
            top: Math.round(scrolled.scrollTop || 0),
            left: Math.round(scrolled.scrollLeft || 0)
          };
        }
      }
      var text = JSON.stringify(view);
      if (text === lastView) return;
      lastView = text;
      post(view);
    };

    /* Twice: once the page has reacted, and again once a slide transition is
       likely over, so the heading is the one the user ends up looking at. */
    var scheduleView = function () {
      for (var t = 0; t < viewTimers.length; t++) clearTimeout(viewTimers[t]);
      viewTimers = [setTimeout(sendView, 150), setTimeout(sendView, 900)];
    };

    /* Kept only when the page changed its DOM in response. */
    var record = function (input) {
      if (viewOverflow || typeof MutationObserver !== 'function' || !document.body) return;
      var changed = false;
      var observer = new MutationObserver(function (list) {
        for (var i = 0; i < list.length; i++) {
          if (!inLayer(list[i].target)) { changed = true; return; }
        }
      });
      observer.observe(document.body, {
        attributes: true, childList: true, subtree: true, characterData: true
      });
      setTimeout(function () {
        try {
          observer.disconnect();
          if (!changed) return;
          if (viewInputs.length >= MAX_INPUTS) {
            viewOverflow = true;
            viewInputs = [];
          } else {
            viewInputs.push(input);
          }
          scheduleView();
        } catch (inner) { /* never break the page */ }
      }, SETTLE_MS);
    };

    window.addEventListener('keydown', function (ev) {
      try {
        if (replaying || !ev.isTrusted || ev.ctrlKey || ev.metaKey || ev.altKey || !NAV_KEYS[ev.key]) return;
        var target = ev.target;
        /* Typing in a field is not navigation, and Space or Enter on a
           control also fires a click, which is recorded on its own. */
        if (target && target.closest && (target.closest(FIELD) || target.closest(CONTROL))) return;
        record({ type: 'key', key: ev.key, shiftKey: !!ev.shiftKey });
      } catch (inner) { /* never break the page */ }
    }, true);

    window.addEventListener('click', function (ev) {
      try {
        if (replaying || !ev.isTrusted || ev.button !== 0) return;
        var target = ev.target;
        if (!target || !target.closest || inLayer(target)) return;
        /* A click in comment mode places a comment; the page never sees it. */
        if (document.body && document.body.classList.contains('act-mode')) return;
        var control = target.closest(CONTROL) || target;
        var where = pathOf(control);
        if (!where) return;
        record({ type: 'click', path: where, label: clip(control.textContent, MAX_LABEL) });
      } catch (inner) { /* never break the page */ }
    }, true);

    window.addEventListener('hashchange', function () {
      try { scheduleView(); } catch (inner) { /* never break the page */ }
    });

    /* capture: a container's scroll does not bubble. */
    window.addEventListener('scroll', function (ev) {
      try {
        var target = ev.target;
        if (target && target.nodeType === 1 && !inLayer(target)) scrolled = target;
        clearTimeout(scrollTimer);
        scrollTimer = setTimeout(sendView, 200);
      } catch (inner) { /* never break the page */ }
    }, { capture: true, passive: true });

    /* Keeping the place across a reload the app made (ENG-4383). A save, a
       quick edit or an agent change reloads the preview, which would open on
       the first slide again. The app keeps the last view this document
       reported and, once the reloaded document says document-start, sends it
       back as a `restore`; this replays it the way the app's inline editor
       does (htmlVisualEditorRuntime.js applyView): the inputs as untrusted
       events, then the hash and the scroll, then the heading check.

       Only the parent window may ask, with its own tag, once per document,
       and the view is checked again here with the same caps as a report.
       After a replay that landed, the recorded inputs ARE the replayed ones,
       so the next report (and the next save's restore) carries the whole way
       there. One that did not land reloads the document, which then stays
       at its start: somewhere random is worse than the beginning. */
    var PARENT_TAG = 'anton-preview-parent';
    var KEY_CODES = {
      ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40,
      PageUp: 33, PageDown: 34, End: 35, Home: 36, ' ': 32
    };
    var PATH_STEP = /^(?:#[A-Za-z][A-Za-z0-9_.:-]{0,99}|[A-Za-z][A-Za-z0-9-]{0,63}:[0-9]{1,4})$/;
    var REPLAY_BUDGET_MS = 8000, STEP_WAIT_MS = 1000, MAX_OFFSET = 1e7;
    var restoreTaken = false;

    var cleanPath = function (path) {
      if (typeof path !== 'string' || !path || path.length > MAX_PATH) return '';
      var steps = path.split('/');
      if (steps.length > MAX_STEPS) return '';
      for (var i = 0; i < steps.length; i++) {
        if (!PATH_STEP.test(steps[i]) || (i > 0 && steps[i].charAt(0) === '#')) return '';
      }
      return path;
    };
    var cleanOffset = function (value) {
      return typeof value === 'number' && isFinite(value) && value >= 0 ?
        Math.min(Math.round(value), MAX_OFFSET) : 0;
    };

    /* The app's restore, rebuilt from known fields; null when any part is off. */
    var cleanView = function (raw) {
      if (!raw || typeof raw !== 'object') return null;
      var rawInputs = raw.inputs == null ? [] : raw.inputs;
      if (!Array.isArray(rawInputs) || rawInputs.length > MAX_INPUTS) return null;
      var inputs = [];
      for (var i = 0; i < rawInputs.length; i++) {
        var item = rawInputs[i];
        if (!item || typeof item !== 'object') return null;
        if (item.type === 'key' && typeof item.key === 'string' &&
          Object.prototype.hasOwnProperty.call(KEY_CODES, item.key)) {
          inputs.push({ type: 'key', key: item.key, shiftKey: item.shiftKey === true });
        } else if (item.type === 'click' && cleanPath(item.path)) {
          inputs.push({ type: 'click', path: item.path, label: clip(typeof item.label === 'string' ? item.label : '', MAX_LABEL) });
        } else {
          return null;
        }
      }
      var container = null;
      if (raw.container && typeof raw.container === 'object' && cleanPath(raw.container.path)) {
        container = {
          path: raw.container.path,
          top: cleanOffset(raw.container.top),
          left: cleanOffset(raw.container.left)
        };
      }
      return {
        hash: typeof raw.hash === 'string' && raw.hash.charAt(0) === '#' && raw.hash.length <= MAX_HASH ? raw.hash : '',
        scrollX: cleanOffset(raw.scrollX),
        scrollY: cleanOffset(raw.scrollY),
        container: container,
        inputs: inputs,
        heading: clip(typeof raw.heading === 'string' ? raw.heading : '', MAX_HEADING)
      };
    };

    /* pathOf in reverse; the comment layer's elements are skipped the same way. */
    var resolvePath = function (path) {
      if (!cleanPath(path) || !document.body) return null;
      var steps = path.split('/'), el = document.body;
      for (var i = 0; i < steps.length && el; i++) {
        var step = steps[i];
        if (step.charAt(0) === '#') { el = document.getElementById(step.slice(1)); continue; }
        var colon = step.lastIndexOf(':'), tag = step.slice(0, colon), index = Number(step.slice(colon + 1));
        var seenCount = -1, found = null;
        for (var c = 0; c < el.children.length; c++) {
          var child = el.children[c];
          if (child.localName !== tag || isLayer(child)) continue;
          seenCount += 1;
          if (seenCount === index) { found = child; break; }
        }
        el = found;
      }
      return el || null;
    };

    var clickTarget = function (input) {
      var el = resolvePath(input.path);
      if (el) return el;
      if (!input.label) return null;
      var match = null, nodes = document.querySelectorAll(CONTROL);
      for (var i = 0; i < nodes.length; i++) {
        if (inLayer(nodes[i]) || clip(nodes[i].textContent, MAX_LABEL) !== input.label) continue;
        if (match) return null;
        match = nodes[i];
      }
      return match;
    };

    var replayKey = function (input) {
      var code = KEY_CODES[input.key], target = document.body || document.documentElement;
      var types = ['keydown', 'keyup'];
      for (var i = 0; i < types.length; i++) {
        var ev = new KeyboardEvent(types[i], {
          key: input.key, code: input.key === ' ' ? 'Space' : input.key, shiftKey: input.shiftKey,
          bubbles: true, cancelable: true, composed: true, view: window
        });
        /* The constructor drops keyCode; decks written against it still read it. */
        try {
          Object.defineProperty(ev, 'keyCode', { get: function () { return code; } });
          Object.defineProperty(ev, 'which', { get: function () { return code; } });
        } catch (inner) { /* key alone */ }
        target.dispatchEvent(ev);
      }
      return true;
    };

    var replayClick = function (input) {
      /* In comment mode a click places a comment instead of reaching the page. */
      if (document.body && document.body.classList.contains('act-mode')) return false;
      var el = clickTarget(input);
      if (!el) return false;
      var r = el.getBoundingClientRect();
      var init = {
        bubbles: true, cancelable: true, composed: true, view: window, button: 0,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2
      };
      var Pointer = typeof PointerEvent === 'function' ? PointerEvent : MouseEvent;
      var pointer = function (type) {
        var extra = { pointerId: 1, isPrimary: true };
        for (var k in init) extra[k] = init[k];
        return new Pointer(type, extra);
      };
      el.dispatchEvent(pointer('pointerdown'));
      el.dispatchEvent(new MouseEvent('mousedown', init));
      el.dispatchEvent(pointer('pointerup'));
      el.dispatchEvent(new MouseEvent('mouseup', init));
      el.dispatchEvent(new MouseEvent('click', init));
      return true;
    };

    /* A frame for the page to react, then any short transition it started
       (a deck often ignores input mid-transition), at most STEP_WAIT_MS. */
    var settleThen = function (next) {
      var waited = 0;
      var check = function () {
        var busy = false;
        try {
          var running = document.getAnimations ? document.getAnimations() : [];
          for (var i = 0; i < running.length && !busy; i++) {
            var a = running[i];
            if (a.playState !== 'running') continue;
            var timing = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
            var left = Number(timing && timing.endTime) - (Number(a.currentTime) || 0);
            busy = isFinite(left) && left <= 1500;
          }
        } catch (inner) { busy = false; }
        if (busy && waited < STEP_WAIT_MS) { waited += 50; setTimeout(check, 50); return; }
        setTimeout(next, 30);
      };
      setTimeout(check, 60);
    };

    var whenLoaded = function (next) {
      if (document.readyState === 'complete') { next(); return; }
      var done = false;
      var go = function () { if (!done) { done = true; next(); } };
      window.addEventListener('load', go);
      setTimeout(go, 5000);
    };

    /* An edit to the very heading on screen (a common ask: "retitle this
       slide") must not count as landing elsewhere. A heading that is gone
       from the page cannot be checked; one still there but not on screen
       means the replay went somewhere else. */
    var headingOnPage = function (text) {
      var nodes = document.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"]');
      for (var i = 0; i < nodes.length && i < 500; i++) {
        if (!inLayer(nodes[i]) && clip(nodes[i].textContent, MAX_HEADING) === text) return true;
      }
      return false;
    };

    var finishRestore = function (status, inputs) {
      replaying = false;
      post({ type: 'view-restored', status: status });
      if (status !== 'ok') {
        try { window.location.reload(); } catch (inner) { /* stays where it is */ }
        return;
      }
      viewInputs = inputs;
      viewOverflow = false;
      lastView = '';
      sendView();
    };

    var restoreView = function (view) {
      replaying = true;
      var timedOut = false, moved = false, missed = false, done = [];
      var budget = setTimeout(function () { timedOut = true; }, REPLAY_BUDGET_MS);
      var check = function () {
        clearTimeout(budget);
        try {
          if (view.container) {
            var box = resolvePath(view.container.path);
            if (box) { box.scrollTop = view.container.top; box.scrollLeft = view.container.left; }
          }
          try { window.scrollTo(view.scrollX, view.scrollY); } catch (inner) { /* no scrolling */ }
          var status = 'ok';
          if (view.heading && (moved || missed)) {
            if (visibleHeading() !== view.heading && (missed || headingOnPage(view.heading))) status = 'mismatch';
          } else if (missed) {
            status = 'mismatch';
          }
          finishRestore(status, done);
        } catch (inner) {
          finishRestore('mismatch', []);
        }
      };
      var hashThen = function () {
        if (view.hash && view.hash !== window.location.hash) {
          window.location.hash = view.hash;
          moved = true;
          settleThen(check);
        } else {
          check();
        }
      };
      var step = function (i) {
        try {
          if (i >= view.inputs.length) { hashThen(); return; }
          if (timedOut) { missed = true; hashThen(); return; }
          var input = view.inputs[i];
          var ok = input.type === 'key' ? replayKey(input) : replayClick(input);
          if (!ok) { missed = true; hashThen(); return; }
          moved = true;
          done.push(input);
          settleThen(function () { step(i + 1); });
        } catch (inner) {
          finishRestore('mismatch', []);
        }
      };
      whenLoaded(function () { settleThen(function () { step(0); }); });
    };

    window.addEventListener('message', function (ev) {
      try {
        var data = ev.data;
        if (IS_TOP || !ev.isTrusted || ev.source !== PARENT || !data || typeof data !== 'object') return;
        if (data.source !== PARENT_TAG || data.type !== 'restore' || restoreTaken) return;
        restoreTaken = true;
        var view = cleanView(data.view);
        if (!view) { post({ type: 'view-restored', status: 'invalid' }); return; }
        restoreView(view);
      } catch (inner) { /* never break the page */ }
    });

    /* While a restore replays, the user's own input waits (the shim's
       listeners are the first on window), and a replayed click on a link to
       another page does not take the preview off the document. */
    var held = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'keydown'];
    for (var h = 0; h < held.length; h++) {
      window.addEventListener(held[h], function (ev) {
        try {
          if (!replaying || !ev.isTrusted) return;
          ev.preventDefault();
          ev.stopImmediatePropagation();
        } catch (inner) { /* never break the page */ }
      }, true);
    }
    window.addEventListener('click', function (ev) {
      try {
        if (!replaying || ev.isTrusted || !ev.target || !ev.target.closest) return;
        var link = ev.target.closest('a[href], area[href]');
        if (link && (link.getAttribute('href') || '').charAt(0) !== '#') ev.preventDefault();
      } catch (inner) { /* never break the page */ }
    });
  } catch (e) { /* no view reports: the page still runs */ }

  var nativeStorageWorks = true;
  try { void window.localStorage; } catch (e) { nativeStorageWorks = false; }
  /* One probe stands for the whole opaque-origin condition: where localStorage
     works, so do cookies, serviceWorker and caches. A browser that disables
     storage by setting also lands here, which costs nothing — the page keeps
     running on the in-memory substitutes. */
  if (nativeStorageWorks) return;

  function makeStorage() {
    var map = new Map();
    var api = {
      getItem: function (k) { k = String(k); return map.has(k) ? map.get(k) : null; },
      setItem: function (k, v) { map.set(String(k), String(v)); },
      removeItem: function (k) { map.delete(String(k)); },
      clear: function () { map.clear(); },
      key: function (i) {
        var keys = Array.from(map.keys());
        return i >= 0 && i < keys.length ? keys[i] : null;
      }
    };
    /* configurable is not optional here: a Proxy's ownKeys trap must report
       every non-configurable own key of its target, and a plain accessor is
       non-configurable. Without this flag Object.keys(localStorage) and
       JSON.stringify(localStorage) throw "trap result did not include
       'length'" — trading the old SecurityError for a new TypeError in the
       two calls the trap below exists to serve. */
    Object.defineProperty(api, 'length', {
      get: function () { return map.size; },
      configurable: true
    });
    /* ownKeys and getOwnPropertyDescriptor are load-bearing: without them
       Object.keys(localStorage) and JSON.stringify(localStorage) come back
       empty, which pages use to enumerate their own saved state. */
    return new Proxy(api, {
      get: function (t, p) {
        if (p in t) return t[p];
        return typeof p === 'string' && map.has(p) ? map.get(p) : undefined;
      },
      set: function (t, p, v) {
        if (typeof p === 'string' && !(p in t)) { map.set(p, String(v)); return true; }
        /* `length` is an accessor with no setter, and this whole file is
           'use strict', so `t[p] = v` below would throw TypeError for it —
           trading the SecurityError this shim exists to avoid for a new
           throw from inside the trap. Native Storage just drops that write,
           so an accessor-only own member does the same here. A page
           reassigning a method (e.g. `localStorage.clear = fn`) hits a
           plain writable data property instead, which falls through to the
           assignment below and shadows it, matching native Storage. */
        var desc = Object.getOwnPropertyDescriptor(t, p);
        if (desc && desc.get && !desc.set) return true;
        t[p] = v;
        return true;
      },
      has: function (t, p) { return (p in t) || (typeof p === 'string' && map.has(p)); },
      deleteProperty: function (t, p) { map.delete(p); return true; },
      ownKeys: function () { return Array.from(map.keys()); },
      getOwnPropertyDescriptor: function (t, p) {
        if (map.has(p)) {
          return { value: map.get(p), writable: true, enumerable: true, configurable: true };
        }
        return Object.getOwnPropertyDescriptor(t, p);
      }
    });
  }

  function define(target, name, value) {
    /* configurable on purpose: a page shipping its own polyfill must be able
       to replace ours. An own property shadows the throwing accessor that
       lives on Window.prototype. */
    try {
      Object.defineProperty(target, name, {
        value: value, configurable: true, writable: true
      });
    } catch (e) { /* frozen target: leave it as it was */ }
  }

  define(window, 'localStorage', makeStorage());
  define(window, 'sessionStorage', makeStorage());
  /* indexedDB exists in the sandbox and throws on open(), so plain feature
     detection does not save the page; the same holds for the two below, which
     throw on property access itself. */
  define(window, 'indexedDB', undefined);
  define(window, 'caches', undefined);
  define(window.navigator, 'serviceWorker', undefined);

  /* Cookies get their own store and grammar: `set` takes
     "name=value; path=/; expires=…" and `get` returns "a=1; b=2". Sharing the
     storage Map would leak cookie strings into Object.keys(localStorage). */
  var cookies = new Map();

  function cookieExpired(text) {
    if (/;\s*max-age\s*=\s*(0|-\d+)/i.test(text)) return true;
    var match = /;\s*expires\s*=\s*([^;]+)/i.exec(text);
    if (!match) return false;
    var when = Date.parse(match[1]);
    return !isNaN(when) && when <= Date.now();
  }

  /* The page's own height, for the app to fit a page that is taller than its
     frame and scrolls nowhere: a deck sized from the frame's width with its
     overflow hidden is cut off below, out of reach. `content` is the page's
     full height, `locked` whether the root or body hide their overflow (so
     nothing scrolls). Sent after load and whenever the page or frame resizes,
     only when it changed. */
  try {
    var lastSize = '', sizeTimer = 0;
    var hidesOverflow = function (el) {
      if (!el) return false;
      var value = window.getComputedStyle(el).overflowY;
      return value === 'hidden' || value === 'clip';
    };
    /* How tall what is on screen runs: the lowest bottom of the visible
       leaf elements (text, images, media) that start inside the frame. A deck
       that hides its other slides below the visible one has a page far taller
       than the frame, but those slides start below it and don't count, so a
       deck that fits isn't shrunk. Only content that starts on screen and
       runs past the bottom (the page really is cut off) makes it taller. */
    var MAX_SCANNED = 4000;
    var MEDIA = { IMG: 1, SVG: 1, CANVAS: 1, VIDEO: 1, IFRAME: 1, svg: 1 };
    var visibleBottom = function (frameHeight) {
      var all = document.body.getElementsByTagName('*');
      var count = Math.min(all.length, MAX_SCANNED);
      var lowest = 0;
      for (var i = 0; i < count; i++) {
        var el = all[i];
        if (el.children.length > 0 && !MEDIA[el.tagName]) continue;
        var box = el.getBoundingClientRect();
        if (box.width < 2 || box.height < 2) continue;
        if (box.top >= frameHeight - 1 || box.bottom <= 0) continue;
        var style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;
        if (box.bottom > lowest) lowest = box.bottom;
      }
      return Math.ceil(lowest + (window.scrollY || 0));
    };
    var sendSize = function () {
      var root = document.documentElement, body = document.body;
      if (!root || !body) return;
      var height = Math.round(window.innerHeight || 0);
      var locked = hidesOverflow(root) || hidesOverflow(body);
      var size = {
        type: 'size',
        width: Math.round(window.innerWidth || 0),
        height: height,
        // A page that scrolls fits as it is; only a locked one is measured.
        content: locked
          ? visibleBottom(height)
          : Math.ceil(Math.max(root.scrollHeight || 0, body.scrollHeight || 0)),
        locked: locked
      };
      var text = JSON.stringify(size);
      if (text === lastSize) return;
      lastSize = text;
      post(size);
    };
    var scheduleSize = function () {
      clearTimeout(sizeTimer);
      sizeTimer = setTimeout(function () {
        try { sendSize(); } catch (inner) { /* never break the page */ }
      }, 150);
    };
    window.addEventListener('load', scheduleSize);
    window.addEventListener('resize', scheduleSize);
    if (typeof ResizeObserver === 'function') {
      var watchBody = function () {
        if (document.body) new ResizeObserver(scheduleSize).observe(document.body);
      };
      if (document.body) watchBody();
      else document.addEventListener('DOMContentLoaded', watchBody);
    }
  } catch (e) { /* size reports are best effort */ }

  try {
    Object.defineProperty(document, 'cookie', {
      configurable: true,
      get: function () {
        var out = [];
        cookies.forEach(function (v, k) { out.push(k + '=' + v); });
        return out.join('; ');
      },
      set: function (raw) {
        var text = String(raw);
        var pair = text.split(';')[0];
        var eq = pair.indexOf('=');
        if (eq < 0) return;
        var name = pair.slice(0, eq).trim();
        if (!name) return;
        if (cookieExpired(text)) { cookies.delete(name); return; }
        cookies.set(name, pair.slice(eq + 1).trim());
      }
    });
  } catch (e) { /* cookie accessor is not redefinable here */ }
})();
