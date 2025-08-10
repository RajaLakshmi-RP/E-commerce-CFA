import React, { Suspense, useEffect, useState, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF } from "@react-three/drei";

const MODEL_URL = "/models/designer-dress.glb";

/* ---------- Fallback orb (never fails) ---------- */
function OrbMesh({ color = "#ffb3ff" }) {
  const ref = useRef();
  useFrame((_, d) => {
    ref.current.rotation.y += d * 0.6;
    ref.current.rotation.x += d * 0.2;
  });
  return (
    <mesh ref={ref} castShadow receiveShadow>
      <torusKnotGeometry args={[1, 0.32, 180, 24]} />
      <meshStandardMaterial color={color} metalness={0.6} roughness={0.25} />
    </mesh>
  );
}

/* ---------- Dress model ---------- */
function DressModel({ url = MODEL_URL }) {
  const group = useRef();
  const { scene } = useGLTF(url);
  useFrame((_, d) => { if (group.current) group.current.rotation.y += d * 0.35; });
  return <primitive ref={group} object={scene} position={[0, -0.8, 0]} scale={1.3} />;
}

export default function DressViewer({ height = 300 }) {
  const [hasModel, setHasModel] = useState(true);

  useEffect(() => {
    let alive = true;
    // Check quickly if the file is there; if not, show fallback
    fetch(MODEL_URL, { method: "HEAD" })
      .then((r) => alive && setHasModel(r.ok))
      .catch(() => alive && setHasModel(false));
    return () => { alive = false; };
  }, []);

  return (
    <div style={{ width: "100%", height, borderRadius: 16, overflow: "hidden" }}>
      <Canvas camera={{ position: [1.6, 1.7, 2.4], fov: 55 }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[4, 6, 2]} intensity={1.2} />
        <Suspense fallback={null}>
          {hasModel ? <DressModel /> : <OrbMesh color="#FF10F0" />}
          <Environment preset="city" />
        </Suspense>
        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>
    </div>
  );
}

// Optional: prefetch if present
// useGLTF.preload(MODEL_URL);
