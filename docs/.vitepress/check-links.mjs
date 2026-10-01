// Check generated links, anchors, and assets using only Node built-ins.
import { readdir, readFile, stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('./dist/', import.meta.url))
const origin = 'https://liumy2010.github.io'
const base = '/LiteEFG/'
const failures = new Set()
const ids = new Map()
let checked = 0

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const groups = await Promise.all(entries.map(entry => {
    const filename = path.join(directory, entry.name)
    return entry.isDirectory() ? walk(filename) : [filename]
  }))
  return groups.flat()
}

function decode(value) {
  return value.replace(/&amp;/g, '&').replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'").replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
}

async function check(value, from, checkFragment = true) {
  if (!value || /^(?:data:|mailto:|tel:|javascript:)/i.test(value)) return
  const url = new URL(decode(value), origin + base + path.relative(root, from).split(path.sep).join('/'))
  if (url.origin !== origin) return
  const label = path.relative(root, from)
  if (!url.pathname.startsWith(base)) {
    failures.add(`${label}: URL escapes ${base}: ${value}`)
    return
  }
  let target = path.join(root, decodeURIComponent(url.pathname.slice(base.length)))
  try {
    if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html')
    await stat(target)
  } catch {
    failures.add(`${label}: missing target: ${value}`)
    return
  }
  checked++
  if (checkFragment && url.hash && target.endsWith('.html')) {
    const fragment = decodeURIComponent(url.hash.slice(1))
    if (!ids.get(target)?.has(fragment)) failures.add(`${label}: missing anchor: ${value}`)
  }
}

let files
try {
  files = await walk(root)
} catch {
  console.error('Build the site first with npm run docs:build.')
  process.exit(1)
}

const pages = files.filter(file => file.endsWith('.html'))
const html = new Map(await Promise.all(pages.map(async file => [file, await readFile(file, 'utf8')])))
for (const [file, content] of html) {
  ids.set(file, new Set([...content.matchAll(/\bid="([^"]*)"/g)].map(match => decode(match[1]))))
}

for (const [file, content] of html) {
  for (const tag of content.matchAll(/<(?:a|link|script|img|source|video|audio)\b[^>]*>/g)) {
    for (const attribute of tag[0].matchAll(/\b(?:href|src|poster)="([^"]*)"/g)) {
      await check(attribute[1], file)
    }
  }
}

for (const file of files.filter(file => file.endsWith('.css'))) {
  const content = await readFile(file, 'utf8')
  for (const match of content.matchAll(/url\(\s*["']?([^\s)'";]+)["']?\s*\)/g)) {
    if (!match[1].startsWith('#')) await check(match[1], file, false)
  }
}

if (failures.size) {
  console.error([...failures].join('\n'))
  process.exit(1)
}
console.log(`Checked ${pages.length} HTML pages and ${checked} local references: links, anchors, and ${base} assets are valid.`)
