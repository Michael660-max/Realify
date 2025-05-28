import React from "react";

export default function UploadButton({ canvasRef, currStage, onDone }) {
  const uploadDrawing = () => {
    return new Promise((resolve, reject) => {
      canvasRef.current.toBlob(async (blob) => {
        if (!blob) return reject("Export Failed");
        const formData = new FormData();
        const file = new File([blob], "drawing.png", { type: "image/png" });
        formData.append("file", file);
        formData.append("prompt", "");

        const response = await fetch("http://localhost:8000/generate_2d", {
          method: "POST",
          body: formData,
        });

        const { url } = await response.json();
        resolve(url);
      }, "image/png");
    });
  };

  const handleClick = async () => {
    try {
      const url = await uploadDrawing();
      onDone(url);
    } catch (err) {
      console.error("Upload fail " + err);
    }
  };

  return currStage === "idle" ? (
    <button className="btn-backdrop-middle" onClick={handleClick}>
      Realify
    </button>
  ) : null;
}
