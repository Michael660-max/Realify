import "./App.css";
import React, { useRef, useState, useEffect } from "react";
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
  const [msg, setMsg] = useState("");
  const messages = [
    "Go ahead, draw a face in the black box! I’ll try my best to realify it—bet you can’t stump me 👀🎨",
    "Ready to see some pure magic? Tap ✨realify✨ and watch your sketch come alive in vivid 2D! 🧙‍♂️",
    "Feeling extremely bold today? Hit 🌀REALIFY🌀 and watch your drawing leap into interactive 3D space! 🌐",
    "Want to try again? Simply erase and quickly draw a new face, then tap realify to restart the fun! 🔄✏️",
  ];
  const [fullText, setFullText] = useState(messages[0]);

  useEffect(() => {
    setMsg("");
    const chars = Array.from(fullText);
    let i = 0;
    const id = setInterval(() => {
      if (i < chars.length) {
        setMsg(fullText.substring(0, ++i));
      } else {
        clearInterval(id);
      }
    }, 50);
    return () => clearInterval(id);
  }, [fullText]);

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
    setFullText(messages[3]);
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
            className={`btn-backdrop-left ${drawMode ? "toggled" : ""}`}
            onClick={() => {
              toggleDraw((prev) => !prev);
              setFullText(messages[1]);
            }}
          >
            <img src="/icons/pencil.png" alt="Draw" />
          </button>
          <button
            className={`btn-backdrop-left-bottom ${fillMode ? "toggled" : ""}`}
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
              setFullText(messages[2]);
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

      {/*---- BOT ----*/}
      <div className="bot-speech">
        <span id="msg">{msg}</span>
      </div>

      <div className="bot-container">
        <img src="icons/robot.png" alt="Robot Guide"></img>
      </div>
    </div>
  );
}

export default App;
