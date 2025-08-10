import { motion } from "framer-motion";

export default function HeroHeader() {
  return (
    <div className="hero">
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Review Sentiment + Reasoning
      </motion.h1>
      <motion.p
        className="sub"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        Paste a product review—get sentiment, reasons, a detailed explanation, and a brand-friendly rephrase.
      </motion.p>
    </div>
  );
}
