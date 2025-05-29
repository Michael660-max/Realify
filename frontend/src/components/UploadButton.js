import React, { useState } from "react";

export default function UploadButton({ canvasRef, currStage, onDone }) {
  const [loading, setLoading] = useState(false);

  const uploadDrawing = () => {
    setLoading(true);
    return new Promise((resolve, reject) => {
      canvasRef.current.toBlob(async (blob) => {
        if (!blob) return reject("Export Failed");

        try {
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
          setLoading(false);
        } catch (err) {
          reject(err);
        }
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
    <button
      disabled={loading}
      className="btn-backdrop-middle"
      onClick={handleClick}
    >
      {loading ? "loading" : "realify"}
    </button>
  ) : null;
}
