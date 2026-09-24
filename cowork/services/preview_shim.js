/*
 * In-frame preview shim. Injected as the first script of every HTML artifact
 * preview (see preview_shim.py).
 *
 * Two independent halves:
 *   1. An error reporter, installed ALWAYS. It is the only way a page's own
 *      failure reaches the user or the agent — the viewer never sees the
 *      frame's console.
 *   2. Storage substitutes, installed ONLY when the native ones throw. The
 *      preview frame runs at an opaque origin, where localStorage,
 *      sessionStorage, document.cookie, indexedDB.open(), navigator.
 *      serviceWorker and caches all raise SecurityError. A generated page
 *      typically reads storage during top-level initialisation, so the throw
 *      aborts its script before any listener is bound and every control on
 *      the rendered page is dead.
 *
 * Verified by hand against a real browser; no automated test exercises this
 * file in a sandbox, so re-check manually after any edit here.
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
  var seen = Object.create(null);
  var sent = 0;

  function post(payload) {
    try {
      if (IS_TOP) return;
      payload.source = TAG;
      PARENT.postMessage(payload, '*');
    } catch (e) { /* frame detached mid-report */ }
  }

  function report(key, payload) {
    if (sent >= MAX_REPORTS || seen[key]) return;
    seen[key] = true;
    sent += 1;
    post(payload);
  }

  post({ type: 'document-start' });

  try {
    /* capture = true: resource failures (<script src>, <link>, <img>) do not
       bubble to window, so a plain listener never sees them. They are the
       direct diagnosis for a CDN blocked by the shell's CSP. */
    window.addEventListener('error', function (ev) {
      try {
        if (ev instanceof ErrorEvent) {
          var file = ev.filename || '';
          var line = typeof ev.lineno === 'number' ? ev.lineno : 0;
          var href = String(window.location.href);
          if (file === href) {
            /* Only this document's own inline scripts shifted; an external
               file carries its own coordinates. lineno is 0 for "Script
               error." and for some parse failures, hence the guard. */
            if (line > LINE_OFFSET) line -= LINE_OFFSET;
            /* The document's own address names nothing the agent can open —
               about:srcdoc on web, a signed draft URL with a query on
               desktop. An empty file tells the server to print the artifact's
               source path instead. */
            file = '';
          } else if (file) {
            /* An external <script src> under the same document (e.g. a
               desktop draft's "static/app.js") still carries the loopback
               origin and full server path, which names nothing the agent can
               open either. Strip the document's own directory so what is left
               is the path the agent actually edits. */
            var dir = href.slice(0, href.lastIndexOf('/') + 1);
            if (dir && file.indexOf(dir) === 0) file = file.slice(dir.length);
          }
          report('e|' + ev.message + '|' + file + '|' + line, {
            type: 'error',
            message: String(ev.message || ''),
            file: file,
            line: line,
            col: typeof ev.colno === 'number' ? ev.colno : 0,
            stack: ev.error && ev.error.stack ? String(ev.error.stack) : ''
          });
          return;
        }
        var node = ev.target || {};
        var tagName = node.tagName || '';
        var url = node.src || node.href || '';
        report('r|' + tagName + '|' + url, {
          type: 'resource',
          tagName: String(tagName),
          url: String(url)
        });
      } catch (inner) { /* a reporter must never break the page */ }
    }, true);

    window.addEventListener('unhandledrejection', function (ev) {
      try {
        var reason = ev.reason;
        var message = reason && reason.message ? reason.message : String(reason);
        report('p|' + message, {
          type: 'error',
          message: 'Unhandled promise rejection: ' + message,
          file: '',
          line: 0,
          col: 0,
          stack: reason && reason.stack ? String(reason.stack) : ''
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
          blockedURI: String(ev.blockedURI || '')
        });
      } catch (inner) { /* a reporter must never break the page */ }
    });
  } catch (e) { /* no listeners available: the page still runs */ }

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
