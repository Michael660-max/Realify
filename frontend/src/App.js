import "./App.css";
import React, { useRef, useState } from "react";
import Canvas from "./components/Canvas";
import UploadButton from "./components/UploadButton";

function App() {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const [isDrawing, setDrawing] = useState(false);
  const [fillMode, setFillMode] = useState(false);

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
    </div>
  );
}

export default App;
