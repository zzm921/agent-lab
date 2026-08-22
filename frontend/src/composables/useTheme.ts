import { ref } from 'vue'

type Theme = 'dark' | 'light'

const THEME_KEY = 'agent-lab-theme'

const theme = ref<Theme>((localStorage.getItem(THEME_KEY) as Theme) || 'dark')

function applyTheme(t: Theme) {
  document.documentElement.dataset.theme = t
}

function initTheme() {
  applyTheme(theme.value)
}

function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  applyTheme(theme.value)
  localStorage.setItem(THEME_KEY, theme.value)
}

export { theme, initTheme, toggleTheme }
