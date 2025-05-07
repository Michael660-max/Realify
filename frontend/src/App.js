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
  const [model, setModel] = useState(null);

  const uploadAndReconstruct = async (file) => {
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("http://localhost:8000/reconstruct_3d", {
      method: "POST",
      body: fd,
    });
    const data = await res.json();
    setModel(data);
  };

  return (
    <div>
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
        accept="image/png"
        onChange={(e) => uploadAndReconstruct(e.target.files[0])}
      />

      {model && (
        <MeshViewer
          objUrl={model.obj}
          mtlUrl={model.mtl}
          textureUrl={model.albedo}
        />
      )}
    </div>
  );
}

export default App;
