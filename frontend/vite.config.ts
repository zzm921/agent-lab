import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [
    vue(),
    {
      name: 'content-hot-reload',
      configureServer(server) {
        // 后端 backend/content/ 下的 md 变更时触发整页刷新（内容由 /api/content 接口提供，无需重建）
        const contentDir = resolve(process.cwd(), '../backend/content').replace(/\\/g, '/')
        server.watcher.add(contentDir)
        const onContentChange = (file: string) => {
          const f = String(file).replace(/\\/g, '/')
          if (f.startsWith(contentDir)) {
            server.ws.send({ type: 'full-reload' })
          }
        }
        server.watcher.on('add', onContentChange)
        server.watcher.on('change', onContentChange)
        server.watcher.on('unlink', onContentChange)
      },
    },
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
