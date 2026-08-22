import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { initTheme } from './composables/useTheme'
import 'highlight.js/styles/github-dark.css'
import './styles/index.css'

initTheme()

createApp(App).use(router).mount('#app')
