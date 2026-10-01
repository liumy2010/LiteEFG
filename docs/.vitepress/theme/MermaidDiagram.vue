<script setup>
import { computed, onMounted, ref, useId, watch } from 'vue'
import { useData } from 'vitepress'

const props = defineProps({ code: { type: String, required: true } })
const source = computed(() => decodeURIComponent(props.code))
const { isDark } = useData()
const id = `mermaid-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`
const svg = ref('')
const error = ref(false)
let revision = 0

async function render() {
  const current = ++revision
  try {
    const { default: mermaid } = await import('mermaid')
    if (current !== revision) return
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      htmlLabels: true,
      theme: 'base',
      themeVariables: isDark.value ? {
        darkMode: true,
        primaryColor: '#1e343b',
        primaryTextColor: '#e1edec',
        primaryBorderColor: '#5d9d8d',
        lineColor: '#92b8ad',
        tertiaryColor: '#1b2229'
      } : {
        primaryColor: '#edf8f5',
        primaryTextColor: '#243835',
        primaryBorderColor: '#74a89a',
        lineColor: '#608578',
        tertiaryColor: '#f6f6f7'
      },
      fontFamily: 'inherit',
      flowchart: { useMaxWidth: true }
    })
    const result = await mermaid.render(`${id}-${current}`, source.value)
    if (current === revision) {
      svg.value = result.svg
      error.value = false
    }
  } catch (cause) {
    if (current === revision) {
      error.value = true
      svg.value = ''
      console.error('Unable to render documentation diagram:', cause)
    }
  }
}

onMounted(() => {
  render()
  watch([isDark, source], render)
})
</script>

<template>
  <figure class="mermaid-diagram">
    <div v-if="svg" class="mermaid-canvas" tabindex="0" role="region" aria-label="Diagram; scroll horizontally if needed" v-html="svg" />
    <figcaption v-if="svg" class="mermaid-hint">Scroll horizontally to read the diagram.</figcaption>
    <pre v-else><code>{{ source }}</code></pre>
    <p v-if="error" role="status">The diagram could not be rendered. Its source is shown above.</p>
    <details v-if="svg">
      <summary>View diagram source</summary>
      <pre><code>{{ source }}</code></pre>
    </details>
  </figure>
</template>
