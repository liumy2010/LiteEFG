<script setup>
import { onMounted } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps({
  target: { type: String, required: true },
  sections: { type: Array, required: true }
})

onMounted(() => {
  const id = window.location.hash.slice(1)
  const section = props.sections.find(section => section.id === id)
  window.location.replace(withBase(section?.target ?? props.target))
})
</script>

<template>
  <p><a :href="withBase(target)">Open the API reference</a>.</p>
  <ul>
    <li v-for="section in sections" :id="section.id" :key="section.id">
      <a :href="withBase(section.target)">{{ section.label }}</a>
    </li>
  </ul>
</template>
