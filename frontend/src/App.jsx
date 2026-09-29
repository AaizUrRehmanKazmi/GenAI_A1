import React, { useEffect, useState } from 'react';
import { getHealth } from './api/client';
import UniversalRestoration from './pages/UniversalRestoration';
import HardRoutedRestoration from './pages/HardRoutedRestoration';
import SoftMoERestoration from './pages/SoftMoERestoration';
import FaceToSketch from './pages/FaceToSketch';

const pages = [['Universal Restoration', UniversalRestoration], ['Hard-Routed Restoration', HardRoutedRestoration], ['Soft Mixture-of-Experts Restoration', SoftMoERestoration], ['Face-to-Sketch Generator', FaceToSketch]];
export default function App() {
  const [active, setActive] = useState(0);
  const [status, setStatus] = useState('Checking API…');
  useEffect(() => { getHealth().then(() => setStatus('API connected • models pending')).catch(() => setStatus('API unavailable')); }, []);
  const Page = pages[active][1];
  return <main className="mx-auto max-w-5xl px-6 py-12"><p className="text-sm text-slate-500">Assignment infrastructure</p><h1 className="mt-2 text-3xl font-bold">GenAI Research Workspace</h1><p role="status" className="mt-3">{status}</p><p className="mt-3 text-sm text-slate-600">Structural placeholder. Complete the Google Stitch design before developing the final interface.</p><nav aria-label="Workspaces" className="my-8 flex flex-wrap gap-2">{pages.map(([title], index) => <button key={title} aria-pressed={active === index} onClick={() => setActive(index)} className={`rounded-lg px-4 py-3 text-sm ${active === index ? 'bg-slate-900 text-white' : 'bg-slate-200'}`}>{title}</button>)}</nav><Page /></main>;
}
