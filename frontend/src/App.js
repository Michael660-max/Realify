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
  const [stage, setStage] = useState("idle");
  const [imgUrl, setImgUrl] = useState("");

  const uploadAndReconstruct = async (url) => {
    const response = await fetch(url);
    const blob = await response.blob();
    const file = new File([blob], "realified.png", { type: blob.type });

    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("http://localhost:8000/reconstruct_3d", {
      method: "POST",
      body: fd,
    });
    const { meshUrl } = await res.json();
    setMeshUrl(`http://localhost:8000${meshUrl}`);
    setStage("3d_ready");
  };

  return (
    <div>
      <div className="container">
        <div className="canvas-wrap">
          {/*---- LEFT BOX ----*/}
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
            className="btn-backdrop-left"
            onClick={() => {
              toggleDraw((prev) => !prev);
            }}
          >
            <img src="/icons/pencil.png" alt="Draw" />
          </button>
          <button
            className="btn-backdrop-left-bottom"
            onClick={() => setFillMode((m) => !m)}
          >
            <img src="/icons/fill.png" alt="Draw" />
          </button>
          <UploadButton
            canvasRef={canvasRef}
            currStage={stage}
            onDone={(url) => {
              setImgUrl(`http://localhost:8000${url}?t=${Date.now()}`);
              setStage("2d_done");
            }}
          />
          {stage === "2d_done" && (
            <button
              className="btn-backdrop-middle-second"
              onClick={(e) => uploadAndReconstruct(imgUrl)}
            >
              REALIFY
            </button>
          )}
        </div>
        {/*---- RIGHT BOX ----*/}
        {stage === "2d_done" && imgUrl && (
          <div className="box box-2">
            <img src={imgUrl} alt="" />
          </div>
        )}

        {stage === "3d_ready" && meshUrl && (
          <div className="box">
            <MeshViewer className="mesh-container" meshUrl={meshUrl} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
