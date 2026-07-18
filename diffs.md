Diffs summary — edits applied on April 2, 2026

Files changed: index.html

1) Rename app title/header
- Changed <title> from "LIFE-EVOLVE // Conway + Genetic Algorithm" to "Game of Life"
- Changed header H1 from "LIFE-EVOLVE" to "Game of Life"

2) DNA/Spawn behavior (prior change)
- Mobile DNA toolbar now selects draw tool only (no spawn)
- Side-panel spawn button cycles species 1–6 (species 1/green now included)

3) Add `PLAN.md` and `diffs.md`
- `PLAN.md` created with implementation plan.
- `diffs.md` (this file) created to track applied diffs.

4) Psychedelic improvements (brighter, faster)
- Added `let psychedelicEnabled = true` toggle.
- Updated `PSYCH_COOL` and `PSYCH_WARM` palettes to be more saturated and brighter.
- Increased per-cell phase frequency/amplitude in `psychCellColor` for stronger color shifts.
- Increased global phase speed in `drawGrid` (psychT now uses Date.now()/600 and sin multipliers increased).
- Gated per-frame psychedelic work with `psychedelicEnabled` so it can be disabled on mobile.
- Added a UI checkbox `#psy-toggle` to toggle psychedelic animation.

5) Theme cleanup — removed redundant "arctic"
- Removed `arctic` entry from `THEMES` (keeps palette list concise).
- Removed "ARCTIC" option from the `#theme-select` dropdown and from the theme overlay.

6) Mobile performance tweaks
- Reduced mobile grid default from 75 → 60 in `resizeCanvas`.
- Reduced mobile GA population `POP_SIZE` from 10 → 8 in `applyMobileDefaults`.
- Lowered mobile default FPS from 6 → 5.
- Disabled `psychedelicEnabled` by default in `applyMobileDefaults` to improve performance on older phones.

7) UI accents
- Ensured spawn accent for species 1 is green (`#39ff14`) so toolbar glow reflects green correctly.

Notes / rationale
- Psychedelic math is now toggleable to give a bright, bold effect on capable devices while allowing older phones to opt out.
- Arctic was removed because it duplicated the neon/cyan theme space; keep the palette list shorter for clarity.

Next steps performed (todo updates)
- Improved psychedelic theme: DONE
- Removed arctic: DONE
- Mobile tweaks (grid/FPS/psy-disable): DONE

Remaining
- Verification & testing on real devices (I recommend testing on an iPhone 12 and a newer phone)

If you want, I can run a quick static syntax check or adjust the psychedelic intensity further.
