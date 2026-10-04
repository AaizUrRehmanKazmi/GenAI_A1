import React, { useEffect, useState, useRef } from 'react';
import { getHealth } from './api/client';
import UniversalRestoration from './pages/UniversalRestoration';
import HardRoutedRestoration from './pages/HardRoutedRestoration';
import SoftMoERestoration from './pages/SoftMoERestoration';
import FaceToSketch from './pages/FaceToSketch';

const pages = [
  ['Universal Restoration', UniversalRestoration, '🔧'],
  ['Hard-Routed', HardRoutedRestoration, '🎯'],
  ['Soft MoE', SoftMoERestoration, '🧬'],
  ['Face → Sketch', FaceToSketch, '✏️'],
];

export default function App() {
  const [active, setActive] = useState(0);
  const [status, setStatus] = useState('checking');
  const [tasks, setTasks] = useState({});

  useEffect(() => {
    getHealth()
      .then(d => {
        setTasks(d.tasks || {});
        setStatus(d.inference_ready ? 'ready' : 'partial');
      })
      .catch(() => setStatus('error'));
  }, []);

  const Page = pages[active][1];

  return (
    <>
      <div className="app-bg" />
      <div className="app-container">
        <header className="app-header">
          <h1>GenAI Research Workspace</h1>
          <p className="subtitle">Image Restoration & Style-Conditioned Sketch Generation</p>
          <div className={`status-badge ${status}`}>
            <span className="dot" />
            {status === 'ready' && 'All models loaded'}
            {status === 'partial' && 'Some models available'}
            {status === 'checking' && 'Connecting to API…'}
            {status === 'error' && 'API unavailable'}
          </div>
          {Object.keys(tasks).length > 0 && (
            <div className="health-tasks">
              {Object.entries(tasks).map(([name, ready]) => (
                <span key={name} className="health-task">
                  <span className={`indicator ${ready ? 'ready' : 'unavailable'}`} />
                  {name}
                </span>
              ))}
            </div>
          )}
        </header>

        <nav className="tab-nav" aria-label="Workspaces">
          {pages.map(([title, _, icon], index) => (
            <button
              key={title}
              className={`tab-btn ${active === index ? 'active' : ''}`}
              onClick={() => setActive(index)}
              aria-pressed={active === index}
            >
              {icon} {title}
            </button>
          ))}
        </nav>

        <div className="fade-in" key={active}>
          <Page />
        </div>
      </div>
    </>
  );
}
