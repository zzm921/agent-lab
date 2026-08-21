import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ModuleView from '../views/ModuleView.vue'
import CompareView from '../views/CompareView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/module/:id', name: 'module', component: ModuleView },
    { path: '/compare', name: 'compare', component: CompareView },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
