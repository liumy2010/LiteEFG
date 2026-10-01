---
description: Links to the environment and strategy application programming interface reference.
search: false
outline: false
---

<script setup>
import ApiRedirect from '../.vitepress/theme/ApiRedirect.vue'

const page = '/guide/environments.html'
const fileEnv = '/guide/environments/file-env.html'
const openSpiel = '/guide/environments/open-spiel.html'
const sections = [
  { id: 'player-numbering', label: 'Player numbering', target: `${page}#player-numbering` },
  { id: 'constructors', label: 'Constructors', target: `${page}#tabular-environments` },
  { id: 'fileenv', label: 'FileEnv', target: `${fileEnv}#fileenv-1` },
  { id: 'openspielenv', label: 'OpenSpielEnv', target: `${openSpiel}#openspielenv` },
  { id: 'graph-lifecycle', label: 'Graph lifecycle', target: `${page}#graph-lifecycle` },
  { id: 'set-graph', label: 'set_graph', target: `${page}#set-graph` },
  { id: 'update', label: 'update', target: `${page}#update` },
  { id: 'update-strategy', label: 'update_strategy', target: `${page}#update-strategy` },
  { id: 'strategy-versions', label: 'Strategy versions', target: `${page}#strategy-versions` },
  { id: 'evaluate-a-profile', label: 'Evaluate a profile', target: `${page}#evaluate-a-profile` },
  { id: 'utility', label: 'utility', target: `${page}#utility` },
  { id: 'exploitability', label: 'exploitability', target: `${page}#exploitability` },
  { id: 'read-and-write-local-values', label: 'Read and write local values', target: `${page}#read-and-write-local-values` },
  { id: 'get-value', label: 'get_value', target: `${page}#get-value` },
  { id: 'set-value', label: 'set_value', target: `${page}#set-value` },
  { id: 'export-strategies', label: 'Export strategies', target: `${openSpiel}#export-and-inspect-policies` },
  { id: 'base-get-strategy', label: 'Base get_strategy', target: `${fileEnv}#get-strategy` },
  { id: 'openspielenv-get-strategy', label: 'OpenSpielEnv.get_strategy', target: `${openSpiel}#openspielenv-get-strategy` },
  { id: 'openspielenv-interact', label: 'OpenSpielEnv.interact', target: `${openSpiel}#openspielenv-interact` }
]
</script>

# Environment application programming interface (API) {#environment-api}

<ApiRedirect :target="`${page}#api-reference`" :sections="sections" />
