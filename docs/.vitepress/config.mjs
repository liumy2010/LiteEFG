import { defineConfig } from 'vitepress'

const repository = 'https://github.com/liumy2010/LiteEFG'

export default defineConfig({
  lang: 'en-US',
  title: 'LiteEFG',
  description: 'Define local update rules in Python. Solve extensive-form games with a C++ computation engine.',
  base: '/LiteEFG/',
  cleanUrls: false,
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/LiteEFG/logo.png' }],
    ['meta', { name: 'theme-color', content: '#087f71' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'LiteEFG' }]
  ],
  sitemap: { hostname: 'https://liumy2010.github.io/LiteEFG/' },
  themeConfig: {
    logo: { src: '/logo.png', alt: '' },
    siteTitle: 'LiteEFG',
    nav: [
      { text: 'Guide', link: '/guide/quick-start', activeMatch: '/guide/(installation|quick-start|concepts|computation-graph|deep-learning(?!/baselines)|environments|cpp-games)' },
      { text: 'Baselines', activeMatch: '/guide/(algorithms|baselines|examples|deep-learning/baselines)', items: [
        { text: 'Tabular baselines', link: '/guide/algorithms' },
        { text: 'Deep learning baselines', link: '/guide/deep-learning/baselines' }
      ] },
      { text: 'Paper and citation', link: '/research' }
    ],
    sidebar: [
      { text: 'Get started', items: [
        { text: 'Installation', link: '/guide/installation' }
      ] },
      { text: 'Tabular solvers', link: '/guide/quick-start', collapsed: false, items: [
        { text: 'Baselines', link: '/guide/algorithms', collapsed: true, items: [
          { text: 'CFR', link: '/guide/baselines/cfr' },
          { text: 'CFR+', link: '/guide/baselines/cfr-plus' },
          { text: 'Discounted CFR', link: '/guide/baselines/dcfr' },
          { text: 'Predictive CFR+', link: '/guide/baselines/pcfr' },
          { text: 'Outcome-sampling MCCFR', link: '/guide/baselines/os-mccfr' },
          { text: 'DOMD', link: '/guide/baselines/domd' },
          { text: 'Clairvoyant mirror descent', link: '/guide/baselines/cmd' },
          { text: 'Regularized DOMD', link: '/guide/baselines/reg-domd' },
          { text: 'Regularized CFR', link: '/guide/baselines/reg-cfr' },
          { text: 'Magnetic mirror descent', link: '/guide/baselines/mmd' },
          { text: 'QFR', link: '/guide/baselines/qfr' },
          { text: 'Implicit exploration OMD', link: '/guide/baselines/ixomd' },
          { text: 'Balanced OMD', link: '/guide/baselines/balanced-omd' },
          { text: 'Balanced FTRL', link: '/guide/baselines/balanced-ftrl' },
          { text: 'FTPL', link: '/guide/baselines/ftpl' }
        ] },
        { text: 'Concepts', link: '/guide/concepts' },
        { text: 'Computation graph', link: '/guide/computation-graph' },
        { text: 'Examples', link: '/guide/examples' },
        { text: 'Environments', link: '/guide/environments', collapsed: true, items: [
          { text: 'FileEnv', link: '/guide/environments/file-env' },
          { text: 'OpenSpiel', link: '/guide/environments/open-spiel' },
          { text: 'CppEnv', link: '/guide/environments/cpp-env' }
        ] }
      ] },
      { text: 'Deep learning', link: '/guide/deep-learning', collapsed: false, items: [
        { text: 'Baselines', link: '/guide/deep-learning/baselines', collapsed: true, items: [
          { text: 'PPO', link: '/guide/deep-learning/baselines/ppo' }
        ] },
        { text: 'Models and optimizers', link: '/guide/deep-learning/models' },
        { text: 'Training', link: '/guide/deep-learning/training' },
        { text: 'Policies and evaluation', link: '/guide/deep-learning/policies' },
        { text: 'Custom objectives', link: '/guide/deep-learning/objectives' },
        { text: 'Parallelism and memory', link: '/guide/deep-learning/scaling' },
        { text: 'Environments', link: '/guide/deep-learning/environments', collapsed: true, items: [
          { text: 'Goofspiel', link: '/guide/deep-learning/goofspiel' },
          { text: 'Dark chess', link: '/guide/deep-learning/dark-chess' }
        ] }
      ] },
      { text: 'Project', items: [
        { text: 'Paper and citation', link: '/research' },
        { text: 'Contributing to the docs', link: '/contributing' }
      ] }
    ],
    outline: { level: [2, 3], label: 'On this page' },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: repository }],
    editLink: { pattern: `${repository}/edit/main/docs/:path`, text: 'Edit this page on GitHub' },
    footer: {
      message: `Released under the <a href="${repository}/blob/main/LICENSE">MIT License</a>.`,
      copyright: 'LiteEFG · Mingyang Liu and contributors'
    }
  },
  markdown: {
    theme: { light: 'github-light', dark: 'github-dark' },
    config(md) {
      const fence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, index, options, env, renderer) => {
        if (tokens[index].info.trim() === 'mermaid') {
          const code = md.utils.escapeHtml(encodeURIComponent(tokens[index].content))
          return `<MermaidDiagram code="${code}" />`
        }
        return fence(tokens, index, options, env, renderer)
      }
    }
  }
})
