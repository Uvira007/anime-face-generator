/**
 * Footer Component
 * 
 * Application footer with credits and links.
 */

import React from 'react';
import { motion } from 'framer-motion';

function Footer() {
  return (
    <footer className="py-8 px-4 mt-16 border-t border-midnight-800/50">
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center"
        >
          {/* Project Info */}
          <div className="mb-6">
            <h3 className="font-display text-lg font-semibold text-white mb-2">
              Anime Face Generator
            </h3>
            <p className="text-midnight-400 text-sm max-w-lg mx-auto">
              A deep learning project using DCGAN (Deep Convolutional Generative Adversarial Networks) 
              to generate unique anime character faces.
            </p>
          </div>

          {/* Tech Stack */}
          <div className="flex flex-wrap justify-center gap-4 mb-6">
            <TechBadge icon="🔥" label="PyTorch" />
            <TechBadge icon="⚡" label="ONNX Runtime" />
            <TechBadge icon="🚀" label="FastAPI" />
            <TechBadge icon="⚛️" label="React" />
            <TechBadge icon="🎨" label="TailwindCSS" />
          </div>

          {/* Dataset Credit */}
          <div className="mb-6 p-4 rounded-xl bg-midnight-900/30 inline-block">
            <p className="text-sm text-midnight-400">
              <span className="text-midnight-300">Dataset:</span>{' '}
              <a
                href="https://www.kaggle.com/datasets/splcher/animefacedataset"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sakura-400 hover:text-sakura-300 underline underline-offset-2"
              >
                Anime Face Dataset
              </a>
              {' '}(~63K images, CC0 License)
            </p>
          </div>

          {/* Links */}
          <div className="flex justify-center gap-6 mb-6">
            <FooterLink
              href="https://github.com"
              icon={<GitHubIcon />}
              label="Source Code"
            />
            <FooterLink
              href="https://arxiv.org/abs/1511.06434"
              icon={<PaperIcon />}
              label="DCGAN Paper"
            />
          </div>

          {/* Copyright */}
          <p className="text-midnight-500 text-xs">
            Built with 💜 for learning and portfolio demonstration
            <br />
            © {new Date().getFullYear()} - Open Source under MIT License
          </p>
        </motion.div>
      </div>
    </footer>
  );
}

/**
 * Technology Badge Component
 */
function TechBadge({ icon, label }) {
  return (
    <span className="px-3 py-1.5 rounded-lg bg-midnight-900/50 border border-midnight-700/50 text-sm text-midnight-300 flex items-center gap-2">
      <span>{icon}</span>
      <span>{label}</span>
    </span>
  );
}

/**
 * Footer Link Component
 */
function FooterLink({ href, icon, label }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 text-midnight-400 hover:text-white transition-colors"
    >
      {icon}
      <span className="text-sm">{label}</span>
    </a>
  );
}

/**
 * GitHub Icon
 */
function GitHubIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"
      />
    </svg>
  );
}

/**
 * Paper/Document Icon
 */
function PaperIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      />
    </svg>
  );
}

export default Footer;