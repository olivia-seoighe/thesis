import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api/generation': {
        target: process.env.VITE_GENERATION_URL || 'http://localhost:18002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/generation/, ''),
      },
      '/api/retrieval': {
        target: process.env.VITE_RETRIEVAL_URL || 'http://localhost:18000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/retrieval/, ''),
      },
    },
  },
})
