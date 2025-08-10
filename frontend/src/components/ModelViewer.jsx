// Example: src/components/ModelViewer.jsx
import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment } from "@react-three/drei";

function Model({ url }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} scale={1.2} />;
}

export default function ModelViewer({ url="/models/your-model.glb", height=320 }) {
  return (
    <div style={{ width:"100%", height, borderRadius:16, overflow:"hidden" }}>
      <Canvas camera={{ position: [1.5, 1.2, 1.8], fov: 55 }}>
        <ambientLight intensity={0.6} />
        <Environment preset="city" />
        <Suspense fallback={null}>
          <Model url={url} />
        </Suspense>
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
