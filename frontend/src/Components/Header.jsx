/**
 * Header Component
 * 
 * Application header with title and navigation.
 */

import React from 'react';
import { motion } from 'framer-motion';
import animeLogo from '../assets/L-deathNote.jfif'
function Header() {
  return (
    <header className="relative py-12 px-4 text-center overflow-hidden">
      {/* Decorative blur element */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[200px] bg-gradient-to-b from-sakura-500/20 to-transparent blur-3xl" />
      
      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative"
      >
        {/* Logo / Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className="inline-block mb-4"
        >
          <div className="w-20 h-20 rounded-2xl overflow-hidden shadow-xl shadow-sakura-500/30">
            <img
            src={animeLogo}
            alt="Anime Face Generator"
            className="w-full h-full object-cover"
            />
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="font-display text-5xl md:text-6xl font-bold mb-4"
        >
          <span className="text-gradient">Anime Face</span>
          <br />
          <span className="text-white">Generator</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="text-midnight-300 text-lg md:text-xl max-w-2xl mx-auto"
        >
          Create unique anime character faces using Deep Convolutional 
          Generative Adversarial Networks (DCGAN)
        </motion.p>

        {/* Tech badges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="flex flex-wrap justify-center gap-3 mt-6"
        >
          {['PyTorch', 'ONNX', 'FastAPI', 'React'].map((tech, index) => (
            <span
              key={tech}
              className="px-3 py-1 rounded-full text-sm bg-white/5 border border-white/10 text-midnight-300"
            >
              {tech}
            </span>
          ))}
        </motion.div>
      </motion.div>
    </header>
  );
}

export default Header;