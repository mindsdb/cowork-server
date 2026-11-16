/* PDFs embedded in an artifact page inside a sandboxed frame.
 *
 * Browsers do not run their built-in PDF viewer in a sandboxed document, and
 * no sandbox keyword turns it back on. This points every <embed>, <object>
 * and <iframe> that shows a PDF of the page's own host at a document that
 * renders it with pdf.js (viewer.mjs next to this file).
 *
 * The page keeps the element it holds a reference to, so its scripts and
 * frameworks go on working with it:
 * - <embed> stays, with type text/html and a blob: URL of the viewer document;
 * - <iframe> stays: it gets srcdoc and is taken out and put back in place,
 *   which drops a navigation to the PDF that is already under way;
 * - <object> stays but hidden, and a new <object> goes right after it: Chrome
 *   never loads a frame into an <object> with fallback content that was
 *   inserted as something else, whatever its attributes become later.
 *
 * On a published artifact it also saves PDFs the page links with
 * <a download> itself (see onClick): Firefox would get the viewer page.
 *
 * Spliced as source text into the frame runtime (mindshub_services) and the
 * preview shim (cowork-server); the folder is copied byte for byte between the
 * two repositories (manifest.txt). Keep it free of a closing script tag, of
 * placeholder-like names and of timers.
 */
function installPdfEmbeds(w, options) {
  'use strict';

  var doc = w.document;
  if (!doc || typeof w.MutationObserver !== 'function' || w.location.origin === 'null') return;
  if (!options.assets()) return;

  var PDF_TYPE = 'application/pdf';
  var VIEWER_TYPE = 'text/html';
  var TITLE = 'PDF document';
  var SELECTOR = 'embed, object, iframe';
  // Shown when viewer.mjs never starts (blocked by the page's own CSP, a failed
  // download): the document must say so without running a script.
  var FALLBACK_STYLE =
    'html,body{margin:0;height:100%}' +
    '.anton-pdf-fallback{visibility:hidden;animation:anton-pdf-reveal 0s 8s forwards;' +
    'padding:16px;color:#333;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}' +
    '@keyframes anton-pdf-reveal{to{visibility:visible}}';

  var states = new WeakMap();
  var ours = new WeakSet();
  var hidden = new Set();

  function report(err) {
    if (typeof options.report === 'function') options.report(err);
  }

  function escapeAttr(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // The absolute URL of the PDF an element shows, or ''. For an element
  // already pointed at the viewer, `type` is the type the page gave it.
  function pdfUrl(raw, type) {
    if (!raw) return '';
    var url;
    try {
      url = new w.URL(raw, doc.baseURI);
    } catch (e) {
      return '';
    }
    if ((url.protocol !== 'https:' && url.protocol !== 'http:') || url.origin !== w.location.origin) return '';
    var byType = String(type || '').trim().toLowerCase() === PDF_TYPE;
    return byType || /\.pdf$/i.test(url.pathname) ? url.href : '';
  }

  // No literal document tags anywhere in this file, comments included: the
  // text sits in an inline script of every preview page, and other code
  // searches the page HTML for the opening head tag and the closing body tag.
  // The parser adds head and body on its own.
  function viewerDocument(url) {
    var assets = options.assets();
    var link = options.draft
      ? '<a href="' + escapeAttr(url) + '" target="_blank" rel="noopener noreferrer">Open the PDF in a new tab</a>'
      : '<a href="' + escapeAttr(url) + '" download target="_blank" rel="noopener noreferrer">Download the PDF</a>';
    return '<' + '!DOCTYPE html><' + 'html lang="en" data-anton-pdf="' + escapeAttr(url) + '"' +
      (options.draft ? ' data-anton-pdf-draft' : '') + '><meta charset="utf-8">' +
      '<meta name="viewport" content="width=device-width, initial-scale=1"><title>PDF</title>' +
      '<style>' + FALLBACK_STYLE + '</style>' +
      '<link rel="stylesheet" href="' + escapeAttr(assets + 'viewer.css') + '">' +
      '<script type="module" src="' + escapeAttr(assets + 'viewer.mjs') + '"><\/script>' +
      '<div class="anton-pdf-fallback" id="anton-pdf-fallback">' +
      '<p>The PDF viewer could not start.</p><p>' + link + '</p></div>';
  }

  function viewerBlob(url) {
    return w.URL.createObjectURL(new w.Blob([viewerDocument(url)], { type: VIEWER_TYPE }));
  }

  // <embed>: the same element, pointed at the viewer document.
  function swapEmbed(el) {
    var state = states.get(el);
    var raw = el.getAttribute('src');
    if (state && raw === state.blob) return;
    var current = el.getAttribute('type');
    var type = state && String(current).toLowerCase() === VIEWER_TYPE ? state.type : current;
    var url = pdfUrl(raw, type);
    if (!url) {
      if (state) restoreEmbed(el, state);
      return;
    }
    var next = { type: type, blob: viewerBlob(url), title: state ? state.title : !el.hasAttribute('title') };
    states.set(el, next);
    if (next.title) el.setAttribute('title', TITLE);
    el.setAttribute('data-anton-pdf', url);
    el.setAttribute('type', VIEWER_TYPE);
    el.setAttribute('src', next.blob);
  }

  function restoreEmbed(el, state) {
    states.delete(el);
    if (String(el.getAttribute('type')).toLowerCase() === VIEWER_TYPE) {
      if (state.type === null) el.removeAttribute('type');
      else el.setAttribute('type', state.type);
    }
    el.removeAttribute('data-anton-pdf');
    if (state.title) el.removeAttribute('title');
  }

  // <iframe>: the same element with srcdoc, taken out and put back in place.
  function swapIframe(el) {
    var state = states.get(el);
    var raw = el.getAttribute('src');
    if (state && raw === state.src && el.hasAttribute('srcdoc')) return;
    var url = pdfUrl(raw, el.getAttribute('type'));
    if (!url) {
      if (state) {
        states.delete(el);
        el.removeAttribute('srcdoc');
        el.removeAttribute('data-anton-pdf');
        if (state.title) el.removeAttribute('title');
      }
      return;
    }
    var next = { src: raw, title: state ? state.title : !el.hasAttribute('title') };
    states.set(el, next);
    if (next.title) el.setAttribute('title', TITLE);
    el.setAttribute('data-anton-pdf', url);
    el.setAttribute('srcdoc', viewerDocument(url));
    var parent = el.parentNode;
    var after = el.nextSibling;
    parent.removeChild(el);
    parent.insertBefore(el, after);
  }

  // <object>: the original stays, hidden; a new <object> right after it
  // shows the viewer.
  function swapObject(el) {
    if (ours.has(el)) return;
    var state = states.get(el);
    var raw = el.getAttribute('data');
    var type = el.getAttribute('type');
    if (state && raw === state.data && type === state.type) {
      follow(el, state);
      return;
    }
    if (state) restoreObject(el, state);
    var url = pdfUrl(raw, type);
    if (!url) return;
    var copy = doc.createElement('object');
    var attrs = el.attributes;
    for (var i = 0; i < attrs.length; i += 1) copy.setAttribute(attrs[i].name, attrs[i].value);
    if (!copy.hasAttribute('title')) copy.setAttribute('title', TITLE);
    copy.setAttribute('data-anton-pdf', url);
    // type and data before the copy enters the document: Chrome decides then.
    copy.setAttribute('type', VIEWER_TYPE);
    copy.setAttribute('data', viewerBlob(url));
    ours.add(copy);
    var next = {
      data: raw,
      type: type,
      name: el.getAttribute('name'),
      display: el.style.getPropertyValue('display'),
      priority: el.style.getPropertyPriority('display'),
      copy: copy
    };
    states.set(el, next);
    hidden.add(el);
    if (next.name !== null) el.removeAttribute('name');
    el.style.setProperty('display', 'none', 'important');
    el.parentNode.insertBefore(copy, el.nextSibling);
  }

  function restoreObject(el, state) {
    states.delete(el);
    hidden.delete(el);
    if (state.copy.parentNode) state.copy.parentNode.removeChild(state.copy);
    if (state.display) el.style.setProperty('display', state.display, state.priority);
    else el.style.removeProperty('display');
    if (state.name !== null) el.setAttribute('name', state.name);
  }

  function follow(el, state) {
    if (el.isConnected && el.nextSibling !== state.copy) el.parentNode.insertBefore(state.copy, el.nextSibling);
  }

  // After removals: a hidden original that left the document takes its copy
  // with it; one that moved gets its copy back right after it.
  function checkHidden() {
    hidden.forEach(function (el) {
      var state = states.get(el);
      if (!el.isConnected) restoreObject(el, state);
      else follow(el, state);
    });
  }

  function insideHidden(el) {
    for (var node = el.parentNode; node; node = node.parentNode) {
      if (hidden.has(node)) return true;
    }
    return false;
  }

  function swap(el) {
    try {
      if (!el.isConnected || insideHidden(el)) return;
      if (el.tagName === 'EMBED') swapEmbed(el);
      else if (el.tagName === 'IFRAME') swapIframe(el);
      else if (el.tagName === 'OBJECT') swapObject(el);
    } catch (err) {
      report(err);
    }
  }

  function scan(node) {
    if (node.nodeType !== 1) return;
    swap(node);
    if (typeof node.querySelectorAll !== 'function') return;
    var found = node.querySelectorAll(SELECTOR);
    for (var i = 0; i < found.length; i += 1) swap(found[i]);
  }

  new w.MutationObserver(function (records) {
    var removed = false;
    for (var i = 0; i < records.length; i += 1) {
      var record = records[i];
      if (record.type === 'attributes') {
        swap(record.target);
        continue;
      }
      if (record.removedNodes.length) removed = true;
      for (var j = 0; j < record.addedNodes.length; j += 1) scan(record.addedNodes[j]);
    }
    if (removed && hidden.size) {
      try {
        checkHidden();
      } catch (err) {
        report(err);
      }
    }
  }).observe(doc, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'data', 'type'] });

  if (doc.documentElement) scan(doc.documentElement);

  // A click on <a download> that points at a PDF of this host. Firefox sends
  // such a download with the headers of a frame navigation, which the host
  // answers with the viewer page, so the file is read with fetch (on a
  // published artifact the runtime's, which carries the token) and saved
  // from a blob: URL. A draft's sandbox refuses downloads: nothing to do there.
  function fileName(url) {
    var last = new w.URL(url).pathname.split('/').pop();
    try {
      last = decodeURIComponent(last);
    } catch (e) {
      // Keep the encoded name.
    }
    return last || 'document.pdf';
  }

  function onClick(event) {
    if (event.defaultPrevented || event.button !== 0) return;
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    var target = event.target;
    var link = target && typeof target.closest === 'function' ? target.closest('a[download]') : null;
    if (!link) return;
    var url = pdfUrl(link.getAttribute('href'), link.getAttribute('type'));
    if (!url) return;
    event.preventDefault();
    var name = link.getAttribute('download') || fileName(url);
    w.fetch(url).then(function (response) {
      if (!response.ok) throw new Error('The PDF download failed with status ' + response.status + '.');
      return response.blob();
    }).then(function (blob) {
      var save = doc.createElement('a');
      save.setAttribute('href', w.URL.createObjectURL(blob));
      save.setAttribute('download', name);
      var parent = doc.body || doc.documentElement;
      parent.appendChild(save);
      save.click();
      parent.removeChild(save);
    }).catch(report);
  }

  if (!options.draft && typeof w.fetch === 'function' && typeof doc.addEventListener === 'function') {
    doc.addEventListener('click', onClick);
  }
}
