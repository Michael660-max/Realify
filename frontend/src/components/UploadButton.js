import React from "react";

function UploadButton({ canvasRef }) {
  const uploadDrawing = async () => {
    const canvas = canvasRef.current;
    const dataUrl = canvas.toDataURL("image/png");

    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const file = new File([blob], "drawing.png", { type: "image/png" });

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json()
    console.log(result)
  };

  return (
    <button className="btn" onClick={uploadDrawing}>
      Upload Drawing
    </button>
  );
}

export default UploadButton;
