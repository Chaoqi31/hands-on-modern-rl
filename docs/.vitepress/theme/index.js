import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import { defineAsyncComponent } from 'vue'
import './style.css'
import Layout from './Layout.vue'
import NavCard from './components/NavCard.vue'
import NavGrid from './components/NavGrid.vue'
import StepBar from './components/StepBar.vue'

const Mermaid = defineAsyncComponent(
  () => import('vitepress-plugin-mermaid/Mermaid.vue')
)

function loadFonts() {
  if (typeof document === 'undefined') return
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href =
    'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap'
  document.head.appendChild(link)
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('NavCard', NavCard)
    ctx.app.component('NavGrid', NavGrid)
    ctx.app.component('StepBar', StepBar)
    ctx.app.component('Mermaid', Mermaid)
    loadFonts()
  }
}
