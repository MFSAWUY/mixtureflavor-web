import { defineConfig } from 'vite'

export default defineConfig(({ mode }) => ({
  server: { port: 5173, strictPort: true, proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { target: 'es2022', chunkSizeWarningLimit: 900 },
  plugins: [{ name: 'showcase-metadata', transformIndexHtml(html) {
    return mode === 'showcase' ? html.replace('MixtureFlavor · 分子风味研究平台</title>', 'MixtureFlavor · 风味预测示例展示</title>')
      .replace(/(<meta name="description" content=")[^"]+/, '$1MixtureFlavor · 单分子、批量与混合物的预计算风味示例。Precomputed synthetic-benchmark examples; no live inference.') : html
  } }],
}))
