import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// A small demo SPA. Vitest config lives here too (pure-helper unit tests only —
// node environment, no jsdom, no three.js, no network).
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
  },
})
