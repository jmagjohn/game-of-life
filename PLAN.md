# LIFE — Game of Life

A fast, pretty cellular-automata playground. Single `index.html`, no dependencies.

## What it is now (rewritten 2026-07)

- **Full-screen canvas** grid sized to the viewport (toroidal / wrap-around)
- **Low-latency engine**: typed-array double buffer, LUT-colored ImageData, one
  `drawImage` per frame — smooth on phones and desktops
- **Age-based color ramps + fading glow trails** (newborn cells flash bright,
  old cells cool off, dead cells leave decaying trails)
- **6 themes**: Cyber Neon, Rainbow Flow (animated hue cycle), Inferno, Matrix,
  Vaporwave, Aurora
- **7 rule sets**: Conway, HighLife, Day & Night, Seeds, Maze, Coral, No Death
- **Tools**: draw, erase, pattern stamp (8 classic patterns w/ rotation), pan
- **Symmetry painting**: mirror and 4-way kaleidoscope
- **Pan/zoom**: pinch on mobile, scroll wheel on desktop, zoom-reset chip
- **Touch-first UI**: glass bottom toolbar, slide-up panels, haptics
- **Keyboard**: space play/pause · S step · D soup · C clear · R rotate stamp ·
  B/E/P tools · T cycle theme · [ ] brush size

## Run

    npm start   # npx serve .

Or just open `index.html` in a browser.
