---
description: Links to the graph and operation application programming interface reference.
search: false
outline: false
---

<script setup>
import ApiRedirect from '../.vitepress/theme/ApiRedirect.vue'

const page = '/guide/computation-graph.html'
const sections = [
  { id: 'graph', label: 'Graph', target: `${page}#graph` },
  { id: 'execution-contexts', label: 'Execution contexts', target: `${page}#execution-contexts` },
  { id: 'graphnode-and-persistent-state', label: 'GraphNode and persistent state', target: `${page}#graphnode-and-persistent-state` },
  { id: 'const', label: 'const', target: `${page}#const` },
  { id: 'arithmetic-and-comparisons', label: 'Arithmetic and comparisons', target: `${page}#arithmetic-and-comparisons` },
  { id: 'reductions-and-elementwise-functions', label: 'Reductions and elementwise functions', target: `${page}#reductions-and-elementwise-functions` },
  { id: 'normalize', label: 'normalize', target: `${page}#normalize` },
  { id: 'aggregate', label: 'aggregate', target: `${page}#aggregate` },
  { id: 'project', label: 'project', target: `${page}#project` },
  { id: 'random-nodes-and-seeding', label: 'Random nodes and seeding', target: `${page}#random-nodes-and-seeding` },
  { id: 'numerical-behavior-and-auxiliary-types', label: 'Numerical behavior', target: `${page}#numerical-behavior` }
]
</script>

# Graph application programming interface (API) {#graph-api}

<ApiRedirect :target="`${page}#api-reference`" :sections="sections" />
