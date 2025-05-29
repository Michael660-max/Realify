import React, { useEffect } from "react";

function Canvas({
  canvasRef,
  contextRef,
  setDrawing,
  isDrawing,
  fillMode,
  drawMode,
}) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const scale = window.devicePixelRatio || 1;
    const cssW = 620;
    const cssH = 620;

    canvas.width = Math.floor(cssW * scale);
    canvas.height = Math.floor(cssH * scale);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.style.borderRadius = "15px";
    // canvas.style.marginBottom = "25px"

    const context = canvas.getContext("2d");
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.scale(scale, scale);
    context.lineCap = "round";
    context.strokeStyle = "black";
    context.lineWidth = 10;
    contextRef.current = context;
  }, [canvasRef, contextRef]);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (!context) return;

    context.resetTransform();
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const scale = window.devicePixelRatio || 1;
    context.scale(scale, scale);
  };

  // Stack based flood fill alg
  const fill = (x, y) => {
    if (!contextRef.current) return;
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const { width, height } = canvas;
    const img = context.getImageData(0, 0, width, height);
    const data = img.data;

    const targetIdx = (x + width * y) * 4;
    const targetColour = data.slice(targetIdx, targetIdx + 4);

    const fillColour = [0, 0, 0, 255];
    if (targetColour.every((v, i) => v === fillColour[i])) return;

    const stack = [[x, y]];
    while (stack.length) {
      const [cx, cy] = stack.pop();
      const idx = (cx + width * cy) * 4;
      const currColour = data.slice(idx, idx + 4);

      if (!currColour.every((v, i) => v === targetColour[i])) continue;

      data[idx] = fillColour[0];
      data[idx + 1] = fillColour[1];
      data[idx + 2] = fillColour[2];
      data[idx + 3] = fillColour[3];

      if (cx > 0) stack.push([cx - 1, cy]);
      if (cy > 0) stack.push([cx, cy - 1]);
      if (cx < width - 1) stack.push([cx + 1, cy]);
      if (cy < height - 1) stack.push([cx, cy + 1]);
    }
    context.putImageData(img, 0, 0);
  };

  const startDrawing = ({ nativeEvent }) => {
    if (!drawMode && !fillMode) return;

    const { offsetX, offsetY } = nativeEvent;
    if (fillMode) {
      fill(offsetX * 2, offsetY * 2);
      return;
    }
    setDrawing(true);
    contextRef.current.beginPath();
    contextRef.current.moveTo(offsetX, offsetY);
  };

  const stopDrawing = () => {
    setDrawing(false);
    contextRef.current.closePath();
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing || fillMode) {
      return;
    }
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.lineTo(offsetX, offsetY);
    contextRef.current.stroke();
  };

  return (
    <div>
      <canvas
        onMouseDown={drawMode || fillMode ? startDrawing : undefined}
        onMouseUp={drawMode || fillMode ? stopDrawing : undefined}
        onMouseMove={drawMode || fillMode ? draw : undefined}
        ref={canvasRef}
      />
      <button onClick={clearCanvas} className={"btn-backdrop-right"}>
        <img src="/icons/clear.png" alt="Clear" />
      </button>
    </div>
  );
}

export default Canvas;
