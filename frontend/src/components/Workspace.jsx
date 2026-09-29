import React from 'react';
export default function Workspace({ title, details }) {
  return <section className="rounded-xl border border-slate-200 bg-white p-6"><h2 className="text-xl font-semibold">{title}</h2><p className="mt-3 text-slate-600">{details}</p><p className="mt-6 text-sm">Pending research, model training, and ONNX integration.</p></section>;
}
