<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'
import { Github, Moon, Settings, Sun } from 'lucide-vue-next'

const { isDark, theme } = useData()

const emit = defineEmits(['open-settings'])

function toggleAppearance() {
  isDark.value = !isDark.value
}

const githubUrl = computed(() => {
  const repo = theme.value.editLink?.repo
  if (repo) return `https://github.com/${repo}`
  return 'https://github.com/walkinglabs/hands-on-modern-rl'
})
</script>

<template>
  <div class="ct-sidebar-footer">
    <div class="ct-sidebar-footer-divider" />
    <div class="ct-sidebar-footer-row">
      <div class="ct-sidebar-footer-actions">
        <button
          class="ct-sidebar-footer-btn"
          :title="isDark ? '切换到浅色' : '切换到深色'"
          @click="toggleAppearance"
        >
          <Sun v-if="isDark" :size="16" :stroke-width="2" />
          <Moon v-else :size="16" :stroke-width="2" />
        </button>
        <button
          class="ct-sidebar-footer-btn"
          title="阅读与外观设置"
          @click="emit('open-settings')"
        >
          <Settings :size="16" :stroke-width="2" />
        </button>
      </div>
      <a
        class="ct-sidebar-footer-link"
        :href="githubUrl"
        target="_blank"
        rel="noopener noreferrer"
        title="GitHub"
      >
        <Github :size="16" :stroke-width="2" />
      </a>
    </div>
  </div>
</template>
