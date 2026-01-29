/**
 * Anime Face Generator - Main Application Component
 * 
 * A beautiful, interactive UI for generating anime character faces
 * using a DCGAN model served via FastAPI.
 */

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import GeneratorPanel from './components/GeneratorPanel';
import ImageGallery from './components/ImageGallery';
import InterpolationPanel from './components/InterpolationPanel';
import Footer from './components/Footer';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // State for generated images
  const [images, setImages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generationTime, setGenerationTime] = useState(null);
  
  // State for interpolation
  const [interpolationImages, setInterpolationImages] = useState([]);
  const [isInterpolating, setIsInterpolating] = useState(false);
  
  // Active tab
  const [activeTab, setActiveTab] = useState('generate');

  /**
   * Generate anime face images
   */
  const handleGenerate = useCallback(async (numImages, seed) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_images: numImages,
          seed: seed || null,
          format: 'base64'
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setImages(data.images);
        setGenerationTime(data.generation_time_ms);
      } else {
        throw new Error('Generation failed');
      }
    } catch (err) {
      console.error('Generation error:', err);
      setError(err.message || 'Failed to generate images. Is the API running?');
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Interpolate between two faces
   */
  const handleInterpolate = useCallback(async (seed1, seed2, numSteps) => {
    setIsInterpolating(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/interpolate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          seed1: seed1,
          seed2: seed2,
          num_steps: numSteps
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setInterpolationImages(data.images);
        setGenerationTime(data.generation_time_ms);
      } else {
        throw new Error('Interpolation failed');
      }
    } catch (err) {
      console.error('Interpolation error:', err);
      setError(err.message || 'Failed to interpolate. Is the API running?');
    } finally {
      setIsInterpolating(false);
    }
  }, []);

  /**
   * Download a single image
   */
  const handleDownload = useCallback((imageDataUrl, index) => {
    const link = document.createElement('a');
    link.href = imageDataUrl;
    link.download = `anime_face_${index + 1}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-mesh">
      {/* Decorative Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-sakura-500/10 rounded-full blur-3xl float-animation" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl float-animation-delay" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-cyan-500/5 rounded-full blur-3xl" />
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        <Header />
        
        <main className="container mx-auto px-4 py-8 max-w-7xl">
          {/* Tab Navigation */}
          <div className="flex justify-center mb-8">
            <div className="glass-card rounded-full p-1 flex gap-1">
              <TabButton 
                active={activeTab === 'generate'}
                onClick={() => setActiveTab('generate')}
              >
                🎨 Generate
              </TabButton>
              <TabButton 
                active={activeTab === 'interpolate'}
                onClick={() => setActiveTab('interpolate')}
              >
                🔄 Interpolate
              </TabButton>
            </div>
          </div>

          {/* Error Display */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="mb-6 p-4 rounded-xl bg-red-500/20 border border-red-500/30 text-red-200 text-center"
              >
                <span className="mr-2">⚠️</span>
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Content Panels */}
          <AnimatePresence mode="wait">
            {activeTab === 'generate' ? (
              <motion.div
                key="generate"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <GeneratorPanel 
                  onGenerate={handleGenerate}
                  isLoading={isLoading}
                  generationTime={generationTime}
                />
                
                <ImageGallery 
                  images={images}
                  isLoading={isLoading}
                  onDownload={handleDownload}
                />
              </motion.div>
            ) : (
              <motion.div
                key="interpolate"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <InterpolationPanel
                  onInterpolate={handleInterpolate}
                  isLoading={isInterpolating}
                  images={interpolationImages}
                  generationTime={generationTime}
                  onDownload={handleDownload}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        <Footer />
      </div>
    </div>
  );
}

/**
 * Tab Button Component
 */
function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`
        px-6 py-2 rounded-full font-medium transition-all duration-300
        ${active 
          ? 'bg-gradient-to-r from-sakura-500 to-purple-500 text-white shadow-lg shadow-sakura-500/30' 
          : 'text-midnight-300 hover:text-white hover:bg-white/5'
        }
      `}
    >
      {children}
    </button>
  );
}

export default App;
