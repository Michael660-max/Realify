import React, { useState } from "react";

function UploadButton({ canvasRef }) {
  const [prompt, setPrompt] = useState("");

  const uploadDrawing = async () => {
    const canvas = canvasRef.current;

    // Sending image and prompt to backend
    canvas.toBlob(async (blob) => {
      if (!blob) return console.error("Canvas export failed");
      const formData = new FormData();
      const file = new File([blob], "drawing.png", { type: "image/png" });
      formData.append("file", file);
      formData.append("prompt", prompt);
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      console.log(result);
    }, "image/png");
  };

  return (
    <div>
      <input
        type="text"
        placeholder="american woman, bangs"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      ></input>

      <button onClick={uploadDrawing}>Generate 2D</button>
    </div>
  );
}

export default UploadButton;
