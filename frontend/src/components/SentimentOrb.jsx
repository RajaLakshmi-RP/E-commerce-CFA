import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

function OrbMesh({ sentiment = "neutral" }) {
  const ref = useRef();
  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.6;
    ref.current.rotation.x += delta * 0.2;
  });

  const { color, emissive } = useMemo(() => {
    if (sentiment === "positive") return { color: "#22c55e", emissive: "#14532d" };
    if (sentiment === "negative") return { color: "#ef4444", emissive: "#7f1d1d" };
    return { color: "#eab308", emissive: "#713f12" };
  }, [sentiment]);

  return (
    <mesh ref={ref} castShadow receiveShadow>
      <torusKnotGeometry args={[1, 0.32, 180, 24]} />
      <meshStandardMaterial color={color} emissive={emissive} metalness={0.6} roughness={0.2} />
    </mesh>
  );
}

export default function SentimentOrb({ sentiment, height = 260 }) {
  return (
    <div style={{ width: "100%", height, borderRadius: 16, overflow: "hidden" }}>
      <Canvas shadows camera={{ position: [3, 2, 3], fov: 55 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 6, 2]} intensity={1.2} castShadow />
        <OrbitControls enableZoom={false} enablePan={false} />
        <OrbMesh sentiment={sentiment} />
      </Canvas>
    </div>
  );
}
