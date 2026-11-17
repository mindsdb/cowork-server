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
