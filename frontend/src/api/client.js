const BASE = '/api';

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`Health check failed (${r.status})`);
  return r.json();
}

export async function universalRestore(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(`${BASE}/universal-restoration`, { method: 'POST', body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Error ${r.status}`); }
  return r.json();
}

export async function hardRouting(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(`${BASE}/hard-routing`, { method: 'POST', body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Error ${r.status}`); }
  return r.json();
}

export async function softMixture(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(`${BASE}/soft-mixture`, { method: 'POST', body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Error ${r.status}`); }
  return r.json();
}

export async function faceToSketch(file, style) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('style', style);
  const r = await fetch(`${BASE}/face-to-sketch`, { method: 'POST', body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Error ${r.status}`); }
  return r.json();
}
