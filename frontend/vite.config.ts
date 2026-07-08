import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-Server proxyt /api auf die FastAPI (uvicorn, Port 8000),
// damit das Frontend ohne CORS-Sonderfälle dieselbe Origin nutzt.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
