import React, { useState, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import { hardRouting } from '../api/client';

const CLASSES = ['clean', 'salt', 'blur', 'occlusion'];

export default function HardRoutedRestoration() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFile = useCallback((f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleRestore = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await hardRouting(file);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const download = useCallback(() => {
    if (!result) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${result.image_base64}`;
    a.download = 'hard_routed_restored.png';
    a.click();
  }, [result]);

  return (
    <div className="card">
      <h2 className="card-title">🎯 Hard-Routed Restoration</h2>
      <p className="card-desc">
        A CNN classifier identifies the corruption type, then routes the image to the
        matching specialist autoencoder. Clean images bypass restoration entirely.
      </p>

      <ImageUploader onFile={handleFile} />

      {preview && (
        <>
          <div className="action-row">
            <button className="btn btn-primary" onClick={handleRestore} disabled={loading}>
              {loading ? <><span className="spinner" /> Classifying & Restoring…</> : '🎯 Classify & Restore'}
            </button>
            {result && (
              <button className="btn btn-secondary" onClick={download}>
                💾 Download
              </button>
            )}
          </div>

          {result && result.probabilities && (
            <>
              <div className="prob-bars">
                {CLASSES.map(cls => {
                  const prob = result.probabilities[cls] || 0;
                  const isSelected = result.selected_expert === cls;
                  return (
                    <div key={cls} className={`prob-row ${isSelected ? 'selected' : ''}`}>
                      <span className="name">{cls}</span>
                      <div className="prob-bar-track">
                        <div
                          className={`prob-bar-fill ${cls}`}
                          style={{ width: `${Math.max(prob * 100, 2)}%` }}
                        >
                          {(prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="metrics-row">
                <div className="metric-badge">
                  <span className="label">Selected Expert</span>
                  <span className="value" style={{ textTransform: 'capitalize' }}>{result.selected_expert}</span>
                </div>
                <div className="metric-badge">
                  <span className="label">Inference</span>
                  <span className="value">{result.inference_ms.toFixed(1)} ms</span>
                </div>
              </div>
            </>
          )}

          <div className="preview-row">
            <div className="preview-box">
              <div className="label">Input Image</div>
              <img src={preview} alt="Input" />
            </div>
            <div className="preview-box">
              <div className="label">Restored Output</div>
              {result ? (
                <img src={`data:image/png;base64,${result.image_base64}`} alt="Restored" />
              ) : (
                <div style={{ aspectRatio: '1', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  {loading ? 'Processing…' : 'Click Classify & Restore'}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {error && <div className="error-msg">⚠ {error}</div>}
    </div>
  );
}
