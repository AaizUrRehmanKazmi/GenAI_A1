import React, { useState, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import { universalRestore } from '../api/client';

export default function UniversalRestoration() {
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
      const data = await universalRestore(file);
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
    a.download = 'restored.png';
    a.click();
  }, [result]);

  return (
    <div className="card">
      <h2 className="card-title">🔧 Universal Restoration</h2>
      <p className="card-desc">
        A single autoencoder that restores images corrupted by salt-and-pepper noise,
        Gaussian blur, or rectangular occlusion — without knowing the corruption type.
      </p>

      <ImageUploader onFile={handleFile} />

      {preview && (
        <>
          <div className="action-row">
            <button className="btn btn-primary" onClick={handleRestore} disabled={loading}>
              {loading ? <><span className="spinner" /> Restoring…</> : '✨ Restore Image'}
            </button>
            {result && (
              <button className="btn btn-secondary" onClick={download}>
                💾 Download Result
              </button>
            )}
          </div>

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
                  {loading ? 'Processing…' : 'Click Restore to see output'}
                </div>
              )}
            </div>
          </div>

          {result && (
            <div className="metrics-row">
              <div className="metric-badge">
                <span className="label">Inference</span>
                <span className="value">{result.inference_ms.toFixed(1)} ms</span>
              </div>
            </div>
          )}
        </>
      )}

      {error && <div className="error-msg">⚠ {error}</div>}
    </div>
  );
}
