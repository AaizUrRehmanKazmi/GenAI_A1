import React, { useState, useCallback, useRef, useEffect } from 'react';
import ImageUploader from '../components/ImageUploader';
import { faceToSketch } from '../api/client';

const STYLES = [
  { id: 1, name: 'Style 1', icon: '✏️', desc: 'Fine lines' },
  { id: 2, name: 'Style 2', icon: '🖊️', desc: 'Bold strokes' },
  { id: 3, name: 'Style 3', icon: '🖌️', desc: 'Artistic' },
];

export default function FaceToSketch() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [style, setStyle] = useState(1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleFile = useCallback((f) => {
    stopWebcam();
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setWebcamActive(true);
      setError(null);
    } catch (e) {
      setError('Could not access webcam: ' + e.message);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setWebcamActive(false);
  }, []);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      const f = new File([blob], 'webcam_capture.png', { type: 'image/png' });
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      stopWebcam();
    }, 'image/png');
  }, [stopWebcam]);

  useEffect(() => {
    return () => stopWebcam();
  }, [stopWebcam]);

  const handleGenerate = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await faceToSketch(file, style);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [file, style]);

  const download = useCallback(() => {
    if (!result) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${result.image_base64}`;
    a.download = `sketch_style${style}.png`;
    a.click();
  }, [result, style]);

  return (
    <div className="card">
      <h2 className="card-title">✏️ Face-to-Sketch Generator</h2>
      <p className="card-desc">
        Upload a facial photograph or capture one with your webcam, select an artist style,
        and generate a sketch using a style-conditioned cGAN.
      </p>

      {!webcamActive ? (
        <>
          <ImageUploader onFile={handleFile} />
          <div className="action-row" style={{ marginTop: '0.75rem' }}>
            <button className="btn btn-webcam" onClick={startWebcam}>
              📷 Use Webcam
            </button>
          </div>
        </>
      ) : (
        <div>
          <div className="webcam-container">
            <video ref={videoRef} autoPlay playsInline muted />
          </div>
          <div className="action-row">
            <button className="btn btn-primary" onClick={capturePhoto}>
              📸 Capture Photo
            </button>
            <button className="btn btn-secondary" onClick={stopWebcam}>
              ✕ Cancel
            </button>
          </div>
        </div>
      )}

      <div className="style-selector">
        {STYLES.map(s => (
          <button
            key={s.id}
            className={`style-card ${style === s.id ? 'selected' : ''}`}
            onClick={() => { setStyle(s.id); setResult(null); }}
          >
            <div className="style-icon">{s.icon}</div>
            <div className="style-name">{s.name}</div>
          </button>
        ))}
      </div>

      {preview && (
        <>
          <div className="action-row">
            <button className="btn btn-primary" onClick={handleGenerate} disabled={loading}>
              {loading ? <><span className="spinner" /> Generating…</> : '✏️ Generate Sketch'}
            </button>
            {result && (
              <button className="btn btn-secondary" onClick={download}>
                💾 Download Sketch
              </button>
            )}
          </div>

          <div className="preview-row">
            <div className="preview-box">
              <div className="label">Original Photo</div>
              <img src={preview} alt="Original" />
            </div>
            <div className="preview-box">
              <div className="label">Generated Sketch — {STYLES.find(s => s.id === style)?.name}</div>
              {result ? (
                <img src={`data:image/png;base64,${result.image_base64}`} alt="Sketch" />
              ) : (
                <div style={{ aspectRatio: '1', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  {loading ? 'Generating…' : 'Select a style and click Generate'}
                </div>
              )}
            </div>
          </div>

          {result && (
            <div className="metrics-row">
              <div className="metric-badge">
                <span className="label">Style</span>
                <span className="value">{STYLES.find(s => s.id === style)?.name}</span>
              </div>
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
