"""Trusted pages served by the edge before any live app code can run."""
import json

from cowork.services.live_app_preview import app_origin, cowork_origins, viewer_sign_in_url


def page(slot: int, *, authorize: bool) -> str:
    config = json.dumps({"cowork": sorted(cowork_origins()), "app": app_origin(slot), "signIn": viewer_sign_in_url(),
                         "authorize": authorize}).replace("<", "\\u003c")
    return '''<!doctype html><meta charset="utf-8"><title>App preview</title>
<p id="status">Connecting app preview…</p><script>
'use strict';
const config = CONFIG;
const [nonce, launchId, parentOrigin = ''] = location.hash.slice(1).split('/');
// The opening page names its own origin, which must be one this server serves.
const named = (() => { try { return decodeURIComponent(parentOrigin); } catch { return ''; } })();
const cowork = config.cowork.includes(named) ? named : '';
const peer = config.authorize ? window.opener : window.parent;
const prefix = '/_live_app/preview/';
let stage = 'starting';
let previewIntent;
const validId = value => typeof value === 'string' && /^[a-f0-9-]{36}$/.test(value);
const send = (event, extra = {}) => peer.postMessage({type:'live-app-preview', event, nonce, launchId, ...extra}, cowork);
const deadline = async promise => {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Timed out')), 8000);
  })]); } finally { clearTimeout(timer); }
};
const call = async (name, body) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const r = await fetch(prefix + name, {method:'POST', credentials:'same-origin', signal:controller.signal,
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), cache:'no-store', redirect:'error'});
    if (!r.ok) {
      const error = new Error('Preview unavailable'); error.status = r.status;
      try { error.detail = (await r.json()).detail; } catch {}
      throw error;
    }
    return r.status === 204 ? {} : await r.json();
  } finally { clearTimeout(timer); }
};
const connected = async intent => {
  await call('connect', {token:intent});
  stage = 'done'; send('connected'); location.replace('/');
};
async function storageAvailable() {
  const key = '__preview_probe_' + nonce;
  localStorage.setItem(key, '1'); localStorage.removeItem(key);
  sessionStorage.setItem(key, '1'); sessionStorage.removeItem(key);
  await deadline(caches.open(key)); await deadline(caches.delete(key));
  await deadline(new Promise((resolve, reject) => {
    const request = indexedDB.open(key, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => { request.result.close(); const removal = indexedDB.deleteDatabase(key);
      removal.onsuccess = resolve; removal.onerror = () => reject(removal.error); };
  }));
}
window.addEventListener('message', async event => {
  const data = event.data;
  if (event.origin !== cowork || event.source !== peer || !data || data.type !== 'live-app-preview'
      || data.nonce !== nonce || data.launchId !== launchId) return;
  if (data.event === 'intent' && stage === 'ready' && /^[a-f0-9]{64}$/.test(data.token || '')) {
    stage = 'checking';
    try {
      if (config.authorize) {
        const result = await call('authorize', {token:data.token});
        stage = 'done'; send('grant', {grant:result.grant}); window.close();
      } else {
        previewIntent = data.token;
        await connected(data.token);
      }
    } catch (error) {
      if (error.status === 401) {
        if (config.authorize) {
          const back = config.app + prefix + 'authorize#' + nonce + '/' + launchId + '/' + encodeURIComponent(cowork);
          location.replace(config.signIn + '?rd=' + encodeURIComponent(back));
        } else { stage = 'sign-in'; send('sign-in'); }
      } else {
        // The sign-in window says why the server refused, such as another account.
        stage = 'failed';
        send('failed', config.authorize && typeof error.detail === 'string' ? {message:error.detail} : {});
      }
    }
  } else if (data.event === 'grant' && stage === 'sign-in' && /^[a-f0-9]{64}$/.test(data.grant || '')) {
    stage = 'checking';
    try {
      await call('exchange', {token:data.grant});
      try { await connected(previewIntent); }
      catch (error) { stage = 'failed'; send(error.status === 401 ? 'cookie-blocked' : 'failed'); }
    } catch { stage = 'failed'; send('failed'); }
  }
});
(async () => {
  if (!validId(nonce) || !validId(launchId) || !cowork || !config.signIn || !peer || (config.authorize ? window.top !== window : window.top === window)) return;
  if (!config.authorize) {
    try { await storageAvailable(); }
    catch { stage = 'failed'; send('storage-blocked'); return; }
  }
  stage = 'ready'; send('ready');
})().catch(() => { stage = 'failed'; send('failed'); });
</script>'''.replace('CONFIG', config)
