import React, { useRef, useState, useCallback } from 'react';

export default function ImageUploader({ onFile, accept = 'image/png,image/jpeg' }) {
  const inputRef = useRef();
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((file) => {
    if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
      onFile(file);
    }
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  return (
    <div
      className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <div className="icon">📁</div>
      <div className="label">Drop an image here or click to browse</div>
      <div className="hint">PNG or JPEG • Max 10 MB</div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  );
}
