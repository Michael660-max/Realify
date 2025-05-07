import React, { useRef, useEffect } from "react";
import * as THREE from "three";
import { MTLLoader, OBJLoader } from "three-stdlib";

function MeshViewer(objUrl, mtlUrl, textureUrl) {
  const mountRef = useRef();

  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(512, 512);
    mountRef.current.appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 2).normalize();
    scene.add(light);

    new MTLLoader().load(mtlUrl, (materials) => {
      materials.preload();
      new OBJLoader().setMaterials(materials).load(objUrl, (obj) => {
        const tex = new THREE.TextureLoader().load(textureUrl);
        obj.traverse((c) => {
          if (c.isMesh) c.material.map = tex;
        });
        scene.add(obj);
        renderer.render(scene, camera);
      });
    });

    return () => {
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, [objUrl, mtlUrl, textureUrl]);

  return <div ref={mountRef} />;
}

export default MeshViewer;
