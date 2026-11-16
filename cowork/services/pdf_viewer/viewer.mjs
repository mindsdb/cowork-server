/* The pdf.js viewer of a PDF embedded in a sandboxed artifact frame.
 *
 * Runs in the document embed.js builds (<html data-anton-pdf> names the PDF)
 * or in page.html, which a frame that navigates to a PDF gets instead of the
 * file (the PDF is then this document's own address). It starts once the
 * document is on screen and has a size, and reads every file through the
 * artifact page's fetch when that page shares this origin, so the frame
 * runtime's token and its refresh apply.
 */

const LIMITS = Object.freeze({
  maxBytes: 33554432,
  maxPages: 500,
  maxCanvasPixels: 16777216,
  maxCanvasPixelsTouch: 4194304,
  maxImageSize: 67108864,
});
const ASSETS_HOOK = Symbol.for('anton.pdf.assets');
const MESSAGES = {
  fetch: 'The PDF could not be loaded.',
  invalid: 'This file is damaged or is not a PDF.',
  password: 'This PDF is protected with a password and cannot be shown here.',
  'too-large': 'This PDF is too large to show here.',
  assets: 'The PDF viewer could not load. Reload the page to try again.',
};

class ViewerError extends Error {
  constructor(kind, cause) {
    super(kind, { cause });
    this.kind = kind;
  }
}

const root = document.documentElement;
const draft = root.hasAttribute('data-anton-pdf-draft');
const embedded = root.hasAttribute('data-anton-pdf');
const pdfUrl = embedded ? root.getAttribute('data-anton-pdf') : location.href;

function setState(value) {
  root.setAttribute('data-anton-pdf-state', value);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// The artifact page's fetch when it shares this origin (the frame runtime's
// wrapper on a published artifact); a parent of another origin throws.
function parentFetch() {
  if (window.parent === window) return null;
  try {
    const parent = window.parent;
    return typeof parent.fetch === 'function' ? parent.fetch.bind(parent) : null;
  } catch {
    return null;
  }
}
const fetchFile = parentFetch() || window.fetch.bind(window);

async function read(url, limit = 0) {
  let response;
  try {
    response = await fetchFile(url, { credentials: 'same-origin' });
  } catch (error) {
    throw new ViewerError('fetch', error);
  }
  if (!response.ok) throw new ViewerError('fetch');
  if (limit && Number(response.headers.get('Content-Length')) > limit) {
    response.body?.cancel().catch(() => {});
    throw new ViewerError('too-large');
  }
  // Wrapped in this realm's Uint8Array: the response may come from the
  // parent's realm, and pdf.js checks the type with instanceof.
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (limit && bytes.byteLength > limit) throw new ViewerError('too-large');
  return bytes;
}

// The asset folder: the frame runtime's current address (it changes when the
// token is refreshed), else the folder this module was loaded from.
function assetsBase() {
  for (const candidate of [window, window.parent]) {
    try {
      const hook = candidate[ASSETS_HOOK];
      const base = typeof hook === 'function' ? hook() : '';
      if (base) return new URL(base);
    } catch {
      // A parent of another origin has no hook to read.
    }
  }
  return new URL('./', import.meta.url);
}

async function loadLibrary(base) {
  const pdfjs = await import(new URL('pdf.min.mjs', base).href);
  globalThis.pdfjsLib = pdfjs;
  const components = await import(new URL('pdf_viewer.mjs', base).href);
  pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdf.worker.min.mjs', base).href;
  const stylesheet = document.getElementById('anton-pdf-components') || element('link');
  stylesheet.id = 'anton-pdf-components';
  stylesheet.rel = 'stylesheet';
  stylesheet.href = new URL('pdf_viewer.css', base).href;
  document.head.append(stylesheet);
  return { pdfjs, components, base };
}

// pdf.js reads CMaps, standard fonts and wasm through this factory, so those
// reads take the same path as the PDF. Same interface as pdf.js's own
// BaseBinaryDataFactory, which the library does not export.
class ReadBinaryDataFactory {
  constructor({ cMapUrl = null, standardFontDataUrl = null, wasmUrl = null }) {
    this.cMapUrl = cMapUrl;
    this.standardFontDataUrl = standardFontDataUrl;
    this.wasmUrl = wasmUrl;
  }

  async fetch({ kind, filename }) {
    const base = this[kind];
    if (!base) throw new Error(`No ${kind} is set.`);
    return read(`${base}${filename}`);
  }
}

// Resolves once the element is in the viewport and has a size: a PDF in a
// hidden tab or below the fold loads nothing until then.
function whenShown(target) {
  return new Promise((resolve) => {
    let visible = false;
    let sized = false;
    const done = () => {
      if (!visible || !sized) return;
      intersection.disconnect();
      resize.disconnect();
      resolve();
    };
    const intersection = new IntersectionObserver((entries) => {
      // The last entry is the current state; earlier ones in the batch are stale.
      visible = entries[entries.length - 1].isIntersecting;
      done();
    });
    const resize = new ResizeObserver(() => {
      sized = target.clientWidth > 0 && target.clientHeight > 0;
      done();
    });
    intersection.observe(target);
    resize.observe(target);
  });
}

function buildUi() {
  document.getElementById('anton-pdf-fallback')?.remove();
  const toolbar = element('div', 'anton-pdf-toolbar');
  const pageInput = element('input', 'anton-pdf-page');
  pageInput.type = 'number';
  pageInput.min = '1';
  pageInput.value = '1';
  pageInput.setAttribute('aria-label', 'Page number');
  const pageCount = element('span', 'anton-pdf-count', '/ …');
  const zoomOut = element('button', 'anton-pdf-button', '−');
  zoomOut.type = 'button';
  zoomOut.setAttribute('aria-label', 'Zoom out');
  const zoomIn = element('button', 'anton-pdf-button', '+');
  zoomIn.type = 'button';
  zoomIn.setAttribute('aria-label', 'Zoom in');
  toolbar.append(pageInput, pageCount, zoomOut, zoomIn, element('span', 'anton-pdf-spacer'));
  // A draft frame's sandbox refuses downloads, so there is no button there.
  let download = null;
  if (!draft) {
    download = element('a', 'anton-pdf-button', 'Download');
    download.hidden = true;
    download.setAttribute('aria-label', 'Download the PDF');
    toolbar.append(download);
  }
  const container = element('div', 'anton-pdf-container');
  const viewer = element('div', 'pdfViewer');
  container.append(viewer);
  const status = element('div', 'anton-pdf-status', 'Loading…');
  document.body.append(toolbar, container, status);
  return { pageInput, pageCount, zoomOut, zoomIn, download, container, viewer, status };
}

function showError(kind) {
  setState(`error:${kind}`);
  const box = element('div', 'anton-pdf-error');
  box.append(element('p', '', MESSAGES[kind] || MESSAGES.fetch));
  // A draft opens the PDF in a new tab, which gets the viewer page; on the
  // viewer page of a draft that tab would be this page again.
  if (!draft || embedded) {
    const link = element('a', '', draft ? 'Open the PDF in a new tab' : 'Download the PDF');
    link.href = pdfUrl;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    if (!draft) link.setAttribute('download', '');
    const line = element('p');
    line.append(link);
    box.append(line);
  }
  document.body.replaceChildren(box);
}

function failureKind(error) {
  if (error instanceof ViewerError) return error.kind;
  if (error && error.name === 'PasswordException') return 'password';
  return 'invalid';
}

function requestedPage() {
  const match = /(?:^|&)page=(\d+)/.exec(new URL(pdfUrl).hash.slice(1));
  return match ? Number(match[1]) : 0;
}

function fileName() {
  try {
    return decodeURIComponent(new URL(pdfUrl).pathname.split('/').pop()) || 'document.pdf';
  } catch {
    return 'document.pdf';
  }
}

async function render({ pdfjs, components, base }, data, ui) {
  const worker = new pdfjs.PDFWorker();
  try {
    await worker.promise;
  } catch (error) {
    throw new ViewerError('assets', error);
  }
  const realWorker = typeof Worker === 'function' && worker.port instanceof Worker;
  root.setAttribute('data-anton-pdf-worker', realWorker ? 'worker' : 'main-thread');
  // Before getDocument: pdf.js takes over the bytes it is given.
  if (ui.download) {
    ui.download.href = URL.createObjectURL(new Blob([data], { type: 'application/pdf' }));
    ui.download.download = fileName();
  }
  const pdf = await pdfjs.getDocument({
    data,
    worker,
    isEvalSupported: false,
    enableXfa: false,
    useWorkerFetch: false,
    cMapUrl: new URL('cmaps/', base).href,
    cMapPacked: true,
    standardFontDataUrl: new URL('standard_fonts/', base).href,
    wasmUrl: new URL('wasm/', base).href,
    maxImageSize: LIMITS.maxImageSize,
    BinaryDataFactory: ReadBinaryDataFactory,
  }).promise;
  if (pdf.numPages > LIMITS.maxPages) {
    await pdf.destroy();
    throw new ViewerError('too-large');
  }

  const { EventBus, GenericL10n, LinkTarget, PDFLinkService, PDFViewer } = components;
  const eventBus = new EventBus();
  const linkService = new PDFLinkService({
    eventBus,
    externalLinkTarget: LinkTarget.BLANK,
    externalLinkRel: 'noopener noreferrer',
  });
  const touch = window.matchMedia('(pointer: coarse)').matches;
  const viewer = new PDFViewer({
    container: ui.container,
    viewer: ui.viewer,
    eventBus,
    linkService,
    textLayerMode: 1,
    annotationMode: pdfjs.AnnotationMode.ENABLE,
    imageResourcesPath: new URL('images/', base).href,
    maxCanvasPixels: touch ? LIMITS.maxCanvasPixelsTouch : LIMITS.maxCanvasPixels,
    // The built-in English strings: no locale file is fetched.
    l10n: new GenericL10n(),
  });
  linkService.setViewer(viewer);

  let zoomed = false;
  eventBus.on('pagesinit', () => {
    viewer.currentScaleValue = 'page-width';
    const page = requestedPage();
    if (page >= 1 && page <= pdf.numPages) viewer.currentPageNumber = page;
    // PDFViewer's own resize handling keeps the scale; follow the width here.
    new ResizeObserver(() => {
      if (!zoomed) viewer.currentScaleValue = 'page-width';
      viewer.update();
    }).observe(ui.container);
  });
  eventBus.on('pagerendered', () => {
    if (root.getAttribute('data-anton-pdf-state') !== 'loading') return;
    ui.status.remove();
    setState('ready');
  });
  eventBus.on('pagechanging', ({ pageNumber }) => {
    ui.pageInput.value = String(pageNumber);
  });
  ui.pageCount.textContent = `/ ${pdf.numPages}`;
  ui.pageInput.max = String(pdf.numPages);
  ui.pageInput.addEventListener('change', () => {
    const page = Number(ui.pageInput.value);
    if (Number.isInteger(page) && page >= 1 && page <= pdf.numPages) viewer.currentPageNumber = page;
    else ui.pageInput.value = String(viewer.currentPageNumber);
  });
  ui.zoomIn.addEventListener('click', () => {
    zoomed = true;
    viewer.increaseScale();
  });
  ui.zoomOut.addEventListener('click', () => {
    zoomed = true;
    viewer.decreaseScale();
  });

  viewer.setDocument(pdf);
  linkService.setDocument(pdf, null);
  if (ui.download) ui.download.hidden = false;
}

async function main() {
  const ui = buildUi();
  setState('waiting');
  await whenShown(ui.container);
  setState('loading');
  const bytes = read(pdfUrl, LIMITS.maxBytes);
  bytes.catch(() => {});
  let library;
  try {
    library = await loadLibrary(assetsBase());
  } catch {
    // On a published artifact the PDF read goes through the frame runtime,
    // which refreshes an expired token; once that read has succeeded the
    // modules are tried once more at the address the runtime gives now. A
    // failed read ends here with its own error.
    await bytes;
    try {
      library = await loadLibrary(assetsBase());
    } catch (error) {
      throw new ViewerError('assets', error);
    }
  }
  await render(library, await bytes, ui);
}

main().catch((error) => showError(failureKind(error)));
