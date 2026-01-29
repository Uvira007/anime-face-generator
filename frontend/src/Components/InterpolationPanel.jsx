/**
 * Interpolation Panel Component
 * 
 * Controls and display for latent space interpolation between two faces.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function InterpolationPanel({ onInterpolate, isLoading, images, generationTime, onDownload }) {
  const [seed1, setSeed1] = useState(42);
  const [seed2, setSeed2] = useState(123);
  const [numSteps, setNumSteps] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    onInterpolate(seed1, seed2, numSteps);
  };

  const randomizeSeed = (setter) => {
    setter(Math.floor(Math.random() * 100000));
  };

  return (
    <div className="space-y-8">
      {/* Control Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-2xl p-6 md:p-8"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
            <span className="text-xl">🔄</span>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">Latent Space Interpolation</h2>
            <p className="text-sm text-midnight-400">Morph between two anime faces</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Seed Inputs */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Seed 1 */}
            <div>
              <label className="block text-sm font-medium text-midnight-300 mb-2">
                Starting Face (Seed 1)
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed1}
                  onChange={(e) => setSeed1(parseInt(e.target.value) || 0)}
                  className="input-field flex-1"
                  placeholder="Seed for face A"
                />
                <button
                  type="button"
                  onClick={() => randomizeSeed(setSeed1)}
                  className="px-3 py-2 rounded-lg bg-midnight-800 text-midnight-300 hover:bg-midnight-700 transition-colors"
                  title="Randomize"
                >
                  🎲
                </button>
              </div>
            </div>

            {/* Seed 2 */}
            <div>
              <label className="block text-sm font-medium text-midnight-300 mb-2">
                Ending Face (Seed 2)
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed2}
                  onChange={(e) => setSeed2(parseInt(e.target.value) || 0)}
                  className="input-field flex-1"
                  placeholder="Seed for face B"
                />
                <button
                  type="button"
                  onClick={() => randomizeSeed(setSeed2)}
                  className="px-3 py-2 rounded-lg bg-midnight-800 text-midnight-300 hover:bg-midnight-700 transition-colors"
                  title="Randomize"
                >
                  🎲
                </button>
              </div>
            </div>
          </div>

          {/* Number of Steps */}
          <div>
            <label className="block text-sm font-medium text-midnight-300 mb-3">
              Interpolation Steps: <span className="text-cyan-400 font-bold">{numSteps}</span>
            </label>
            <input
              type="range"
              min="2"
              max="20"
              value={numSteps}
              onChange={(e) => setNumSteps(parseInt(e.target.value))}
              className="w-full h-2 bg-midnight-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="flex justify-between text-xs text-midnight-500 mt-1">
              <span>Fewer steps (faster)</span>
              <span>More steps (smoother)</span>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex items-center justify-between pt-4">
            <div className="text-sm text-midnight-400">
              {generationTime && !isLoading && (
                <span>
                  Generated in <span className="text-cyan-400">{generationTime.toFixed(0)}ms</span>
                </span>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary flex items-center gap-3"
              style={{ background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)' }}
            >
              {isLoading ? (
                <>
                  <LoadingSpinner />
                  <span>Interpolating...</span>
                </>
              ) : (
                <>
                  <span>Interpolate</span>
                  <span className="text-xl">↔</span>
                </>
              )}
            </button>
          </div>
        </form>
      </motion.div>

      {/* Interpolation Results */}
      <AnimatePresence>
        {images.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="glass-card rounded-2xl p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-4 text-center">
              Face Morphing Sequence
            </h3>
            
            {/* Image Strip */}
            <div className="overflow-x-auto pb-4">
              <div className="flex gap-3 min-w-max justify-center">
                {images.map((image, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="relative group"
                  >
                    <div className="w-20 h-20 md:w-24 md:h-24 rounded-lg overflow-hidden border-2 border-transparent hover:border-cyan-500/50 transition-colors">
                      <img
                        src={image}
                        alt={`Interpolation step ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    
                    {/* Step indicator */}
                    <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-xs text-midnight-500">
                      {index === 0 ? '🔵' : index === images.length - 1 ? '🟣' : '•'}
                    </div>
                    
                    {/* Download on click */}
                    <button
                      onClick={() => onDownload(image, index)}
                      className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg"
                    >
                      <span className="text-white text-xs">⬇</span>
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Progress bar visualization */}
            <div className="mt-6 relative">
              <div className="h-1 bg-midnight-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" />
              </div>
              <div className="flex justify-between mt-2 text-xs text-midnight-400">
                <span>Seed: {seed1}</span>
                <span className="text-midnight-500">← Latent Space Journey →</span>
                <span>Seed: {seed2}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {!isLoading && images.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <div className="inline-block p-6 rounded-full bg-midnight-900/50 mb-4">
            <span className="text-5xl opacity-50">🔄</span>
          </div>
          <p className="text-midnight-400">
            Enter two seeds and click "Interpolate" to see the morphing sequence
          </p>
        </motion.div>
      )}
    </div>
  );
}

/**
 * Loading Spinner Component
 */
function LoadingSpinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

export default InterpolationPanel;
