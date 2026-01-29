/**
 * Generator Panel Component
 * 
 * Controls for generating anime face images.
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';

function GeneratorPanel({ onGenerate, isLoading, generationTime }) {
  const [numImages, setNumImages] = useState(4);
  const [seed, setSeed] = useState('');
  const [useRandomSeed, setUseRandomSeed] = useState(true);

  const handleSubmit = (e) => {
    e.preventDefault();
    const seedValue = useRandomSeed ? null : parseInt(seed, 10) || null;
    onGenerate(numImages, seedValue);
  };

  const generateRandomSeed = () => {
    const randomSeed = Math.floor(Math.random() * 100000);
    setSeed(randomSeed.toString());
    setUseRandomSeed(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6 md:p-8 mb-8"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sakura-500 to-purple-500 flex items-center justify-center">
          <span className="text-xl">✨</span>
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Generate Faces</h2>
          <p className="text-sm text-midnight-400">Create unique anime characters</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Number of Images Slider */}
        <div>
          <label className="block text-sm font-medium text-midnight-300 mb-3">
            Number of Images: <span className="text-sakura-400 font-bold">{numImages}</span>
          </label>
          <div className="flex items-center gap-4">
            <input
              type="range"
              min="1"
              max="16"
              value={numImages}
              onChange={(e) => setNumImages(parseInt(e.target.value))}
              className="flex-1 h-2 bg-midnight-800 rounded-lg appearance-none cursor-pointer accent-sakura-500"
            />
            <div className="flex gap-2">
              {[1, 4, 9, 16].map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => setNumImages(n)}
                  className={`w-10 h-10 rounded-lg text-sm font-medium transition-all ${
                    numImages === n
                      ? 'bg-sakura-500 text-white shadow-lg shadow-sakura-500/30'
                      : 'bg-midnight-800 text-midnight-300 hover:bg-midnight-700'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Seed Control */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-medium text-midnight-300">
              Random Seed
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useRandomSeed}
                onChange={(e) => setUseRandomSeed(e.target.checked)}
                className="w-4 h-4 rounded bg-midnight-800 border-midnight-600 text-sakura-500 focus:ring-sakura-500/30"
              />
              <span className="text-sm text-midnight-400">Random</span>
            </label>
          </div>
          <div className="flex gap-3">
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Enter seed for reproducibility"
              disabled={useRandomSeed}
              className={`input-field flex-1 ${useRandomSeed ? 'opacity-50' : ''}`}
            />
            <button
              type="button"
              onClick={generateRandomSeed}
              className="px-4 py-3 rounded-lg bg-midnight-800 text-midnight-300 hover:bg-midnight-700 hover:text-white transition-all"
              title="Generate random seed"
            >
              🎲
            </button>
          </div>
          <p className="mt-2 text-xs text-midnight-500">
            Use a seed to generate the same faces again
          </p>
        </div>

        {/* Generate Button */}
        <div className="flex items-center justify-between pt-4">
          <div className="text-sm text-midnight-400">
            {generationTime && !isLoading && (
              <span>
                Generated in <span className="text-sakura-400">{generationTime.toFixed(0)}ms</span>
              </span>
            )}
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary flex items-center gap-3"
          >
            {isLoading ? (
              <>
                <LoadingSpinner />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <span>Generate</span>
                <span className="text-xl">→</span>
              </>
            )}
          </button>
        </div>
      </form>
    </motion.div>
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

export default GeneratorPanel;