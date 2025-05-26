import "./App.css";
import React, { useRef, useState } from "react";
import Canvas from "./components/Canvas";
import UploadButton from "./components/UploadButton";
import MeshViewer from "./components/MeshViewer";

function App() {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const [isDrawing, setDrawing] = useState(false);
  const [fillMode, setFillMode] = useState(false);
  const [meshUrl, setMeshUrl] = useState(null);
  const [drawMode, toggleDraw] = useState(false);

  const uploadAndReconstruct = async (file) => {
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("http://localhost:8000/reconstruct_3d", {
      method: "POST",
      body: fd,
    });
    const { meshUrl } = await res.json();
    setMeshUrl(`http://localhost:8000${meshUrl}`);
  };

  return (
    <div>
      <div className="container">
        <div className="canvas-wrap">
          <Canvas
            className="box"
            canvasRef={canvasRef}
            contextRef={contextRef}
            isDrawing={isDrawing}
            setDrawing={setDrawing}
            fillMode={fillMode}
            drawMode={drawMode}
          />
          <button
            className="btn-backdrop"
            onClick={() => {
              toggleDraw((prev) => !prev);
            }}
          >
            <img src="/icons/pencil.png" alt="Draw" />
          </button>
        </div>

        <div className="box" />
      </div>

      <UploadButton canvasRef={canvasRef} />
      <button onClick={() => setFillMode((m) => !m)}>
        {fillMode ? "Draw" : "Fill"}
      </button>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => uploadAndReconstruct(e.target.files[0])}
      />

      {meshUrl && <MeshViewer meshUrl={meshUrl} />}
    </div>
  );
}

export default App;
