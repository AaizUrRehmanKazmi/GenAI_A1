import React, { useState, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import { softMixture } from '../api/client';

const EXPERTS = ['clean', 'salt', 'blur', 'occlusion'];

export default function SoftMoERestoration() {
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
      const data = await softMixture(file);
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
    a.download = 'soft_moe_restored.png';
    a.click();
  }, [result]);

  const dominantIdx = result?.routing_weights
    ? EXPERTS.indexOf(Object.entries(result.routing_weights).sort((a, b) => b[1] - a[1])[0][0])
    : -1;

  return (
    <div className="card">
      <h2 className="card-title">🧬 Soft Mixture-of-Experts</h2>
      <p className="card-desc">
        A differentiable gate blends outputs from all four expert branches using continuous
        routing weights. Unlike hard routing, every expert contributes to the final output.
      </p>

      <ImageUploader onFile={handleFile} />

      {preview && (
        <>
          <div className="action-row">
            <button className="btn btn-primary" onClick={handleRestore} disabled={loading}>
              {loading ? <><span className="spinner" /> Blending Experts…</> : '🧬 Restore with MoE'}
            </button>
            {result && (
              <button className="btn btn-secondary" onClick={download}>
                💾 Download
              </button>
            )}
          </div>

          {result && result.routing_weights && (
            <>
              <div className="routing-weights">
                {EXPERTS.map((name, idx) => {
                  const w = result.routing_weights[name] || 0;
                  return (
                    <div key={name} className={`weight-card ${idx === dominantIdx ? 'dominant' : ''}`}>
                      <div className="expert-name">{name}</div>
                      <div className="weight-value">{(w * 100).toFixed(1)}%</div>
                      <div className="weight-bar">
                        <div className="weight-bar-inner" style={{ width: `${w * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="metrics-row">
                <div className="metric-badge">
                  <span className="label">Dominant Expert</span>
                  <span className="value" style={{ textTransform: 'capitalize' }}>
                    {Object.entries(result.routing_weights).sort((a, b) => b[1] - a[1])[0][0]}
                  </span>
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
              <div className="label">Blended Restoration</div>
              {result ? (
                <img src={`data:image/png;base64,${result.image_base64}`} alt="Restored" />
              ) : (
                <div style={{ aspectRatio: '1', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  {loading ? 'Processing…' : 'Click Restore to see MoE output'}
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
