import { createRouter, createWebHistory } from 'vue-router'
import LandingView from '../views/LandingView.vue'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'landing', component: LandingView },
    { path: '/lab', name: 'lab', component: HomeView },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
