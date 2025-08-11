import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Inject a unique build id into index.html and use relative asset paths
export default defineConfig({
  base: './', // required for GCS/static hosting
  plugins: [
    react(),
    {
      name: 'inject-build-id',
      transformIndexHtml(html) {
        const id = Date.now().toString() // unique per build
        return html.replace(
          '</head>',
          `  <meta name="build-id" content="${id}">\n</head>`
        )
      },
    },
  ],
})
