/**
 * Image Gallery Component
 * 
 * Displays generated anime face images in a responsive grid.
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function ImageGallery({ images, isLoading, onDownload }) {
  // Calculate grid columns based on number of images
  const getGridCols = (count) => {
    if (count === 1) return 'grid-cols-1 max-w-md';
    if (count <= 4) return 'grid-cols-2 max-w-2xl';
    if (count <= 9) return 'grid-cols-3 max-w-3xl';
    return 'grid-cols-4 max-w-4xl';
  };

  // Empty state
  if (!isLoading && images.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center py-16"
      >
        <div className="inline-block p-6 rounded-full bg-midnight-900/50 mb-4">
          <span className="text-6xl opacity-50">🖼️</span>
        </div>
        <p className="text-midnight-400 text-lg">
          Click "Generate" to create anime faces
        </p>
        <p className="text-midnight-500 text-sm mt-2">
          Your generated images will appear here
        </p>
      </motion.div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className={`grid ${getGridCols(4)} gap-4 mx-auto`}>
        {[...Array(4)].map((_, index) => (
          <div
            key={index}
            className="aspect-square rounded-xl bg-midnight-900/50 loading-shimmer"
          />
        ))}
      </div>
    );
  }

  return (
    <div className={`grid ${getGridCols(images.length)} gap-4 mx-auto`}>
      <AnimatePresence mode="popLayout">
        {images.map((image, index) => (
          <ImageCard
            key={`${image.substring(0, 50)}-${index}`}
            image={image}
            index={index}
            onDownload={onDownload}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}

/**
 * Individual Image Card Component
 */
function ImageCard({ image, index, onDownload }) {
  const [isHovered, setIsHovered] = React.useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ 
        duration: 0.4,
        delay: index * 0.05,
        type: "spring",
        stiffness: 200
      }}
      className="image-card aspect-square group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Image */}
      <img
        src={image}
        alt={`Generated anime face ${index + 1}`}
        className="w-full h-full object-cover"
        loading="lazy"
      />

      {/* Hover Overlay */}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent flex items-end justify-center p-4"
          >
            <div className="flex gap-2">
              {/* Download Button */}
              <motion.button
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 20, opacity: 0 }}
                transition={{ delay: 0.05 }}
                onClick={() => onDownload(image, index)}
                className="px-4 py-2 rounded-lg bg-white/10 backdrop-blur-sm text-white text-sm font-medium hover:bg-white/20 transition-colors flex items-center gap-2"
              >
                <DownloadIcon />
                Download
              </motion.button>

              {/* View Full Button */}
              <motion.a
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 20, opacity: 0 }}
                transition={{ delay: 0.1 }}
                href={image}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-white/10 backdrop-blur-sm text-white hover:bg-white/20 transition-colors"
                title="View full size"
              >
                <ExpandIcon />
              </motion.a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Index Badge */}
      <div className="absolute top-2 left-2 px-2 py-1 rounded-md bg-black/50 backdrop-blur-sm text-xs text-white/70 font-medium">
        #{index + 1}
      </div>
    </motion.div>
  );
}

/**
 * Download Icon
 */
function DownloadIcon() {
  return (
    <svg
      className="w-4 h-4"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
      />
    </svg>
  );
}

/**
 * Expand Icon
 */
function ExpandIcon() {
  return (
    <svg
      className="w-4 h-4"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
      />
    </svg>
  );
}

export default ImageGallery;
