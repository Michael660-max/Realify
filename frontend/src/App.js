import "./App.css";
import React, { useRef, useState } from "react";
import Canvas from "./components/Canvas";
import UploadButton from "./components/UploadButton";
import MeshViewer from "./components/MeshViewer"

function App() {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const [isDrawing, setDrawing] = useState(false);
  const [fillMode, setFillMode] = useState(false);
  const [meshUrl, setMeshUrl] = useState(null);

  const uploadAndReconstruct = async (file) => {
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("http://localhost:8000/reconstruct_3d", {
      method: "POST",
      body: fd,
    });
    const { meshUrl: path } = await res.json();
    setMeshUrl(`http://localhost:8000${path}`);
  };

  return (
    <div style={{ width: "800px", height: "600px" }}>
      <Canvas
        canvasRef={canvasRef}
        contextRef={contextRef}
        isDrawing={isDrawing}
        setDrawing={setDrawing}
        fillMode={fillMode}
      />
      <UploadButton canvasRef={canvasRef} />
      <button onClick={() => setFillMode((m) => !m)}>
        {fillMode ? "Draw" : "Fill"}
      </button>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => uploadAndReconstruct(e.target.files[0])}
      />

      {meshUrl && <MeshViewer meshUrl={meshUrl}/>}
    </div>
  );
}

export default App;
