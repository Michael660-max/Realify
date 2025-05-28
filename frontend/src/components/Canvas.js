import React, { useEffect } from "react";

function Canvas({ canvasRef, contextRef, setDrawing, isDrawing, fillMode, drawMode }) {

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
    context.fillStyle = "#2D2D2D";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.scale(scale, scale);
    context.lineCap = "round";
    context.strokeStyle = "white";
    context.lineWidth = 10;
    contextRef.current = context;
  }, [canvasRef, contextRef]);

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

    const fillColour = [255, 255, 255, 255];
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
    if (!drawMode) return

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
    <canvas
      onMouseDown={drawMode ? startDrawing : undefined}
      onMouseUp={drawMode ? stopDrawing : undefined}
      onMouseMove={drawMode ? draw : undefined}
      ref={canvasRef}
    />
  );
}

export default Canvas;
