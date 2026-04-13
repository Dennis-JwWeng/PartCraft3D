# VLM Phase-1 Global & Material Edit Diversity — Design Spec

**Date:** 2026-04-13
**File:** `partcraft/pipeline_v2/s1_vlm_core.py`
**Scope:** `USER_PROMPT_TEMPLATE` — global and material edit type sections only
**Status:** Approved
**Related:** `2026-04-13-vlm-edit-quality-design.md` (separate fix, same file)

---

## Problem

Phase-1 VLM outputs show two systematic biases in global (`glb`) and material (`mat`)
edit types, diagnosed from shard05 test output across ~20 objects.

### Bias A: Global style collapse

Global (`target_style`) values cluster on ≤8 distinct tokens across all objects:

| Style         | Count |
|---|---|
| minimalist    | 10    |
| futuristic    | 8     |
| vintage       | 6     |
| cartoon/cartoonish | 6 |
| realistic     | 4     |
| low-poly      | 4     |

Most objects receive nearly the same three global edits regardless of object type.

### Bias B: Material/global semantic overlap

Some global styles are surface-property words rather than artistic aesthetics:
`wireframe`, `clay`, `stone`, `ice`, `plastic` appear in `target_style` fields.
These are semantically indistinguishable from `target_material` values.

### Root Cause

Per-type prompt guidance volume is severely imbalanced:
- `modification`: ~15 lines + examples + explicit FORBIDDEN block → **zero violations**
- `global`: 1 line (`global: {"target_style": "..."}`) → **constant bias**
- `material`: 1 line → no boundary enforcement

The model CAN follow explicit constraints — modification compliance is 100% in
shard05. The global and material sections simply have none.

Constraint: VLM uses `enable_thinking: False`. No CoT. Explicit examples and
prohibition rules are the only effective lever.

---

## Design

**Zero schema changes.** `extract_json_object`, `validate`, `quota_for`,
JSON field names — all unchanged.

---

### Change 1: Add `# GLOBAL STYLE EDITS` block

Modelled on the existing `# MODIFICATION EDITS` block. Insert immediately
before the `edit_params` schema table.

```
# GLOBAL STYLE EDITS — ARTISTIC / RENDERING AESTHETIC ONLY

A global edit transforms the ENTIRE object's artistic or rendering aesthetic.
It must change how the object looks as a *visual artwork* — NOT what material
it is made of.

STRICTLY FORBIDDEN in global target_style:
  • Surface-material words: gold, silver, metal, wood, stone, clay, glass,
    ceramic, rubber, plastic, ice, crystal, fabric, concrete, leather.
    → Those belong in "material" edits.
  • Generic quality descriptors: "realistic", "detailed", "high quality".
  • Near-duplicate styles: "cartoon", "cartoonish", "toon" count as ONE choice.

VALID target_style — choose from DIFFERENT categories each object:

  Rendering / shading:
    cel-shading, flat-shading, wireframe outline with colored faces,
    watercolor wash, oil-painting impasto, impressionist brushstroke,
    pointillist dots, charcoal sketch, ink wash (sumi-e), stained-glass mosaic,
    neon-glow bloom

  Historical / regional art movements:
    Art Nouveau organic lines, Art Deco geometric, ukiyo-e woodblock print,
    Bauhaus functional, brutalist concrete aesthetic, baroque gilded,
    gothic cathedral tracery, ancient terracotta figurine,
    Ming dynasty blue-and-white porcelain

  Genre / subculture:
    cyberpunk neon-and-chrome, steampunk brass-and-gears, solarpunk organic-tech,
    vaporwave pastel grid, retro-1980s pixel-art, lo-fi cassette-tape grain,
    biomechanical flesh-and-machine, origami paper-fold geometry,
    LEGO brick construction, Islamic geometric mosaic tile

DIVERSITY RULE: For this object's {n_global} global edits, each target_style
MUST come from a DIFFERENT category. Near-synonyms count as the same choice.
```

---

### Change 2: Extend material schema entry with boundary clause

Replace the current single-line material entry in the `edit_params` table:

Before:
```
material:     {{"target_material": "..."}}
```

After:
```
material:     {{"target_material": "..."}}
              Target must be a specific surface substance or finish, e.g.:
              "polished walnut wood", "brushed stainless steel",
              "frosted borosilicate glass", "hand-stitched leather",
              "poured concrete", "translucent amber resin".
              FORBIDDEN: style/aesthetic words (cartoon, vintage, futuristic,
              minimalist, steampunk) — those belong in "global" edits.
```

---

## Files Changed

`partcraft/pipeline_v2/s1_vlm_core.py` — `USER_PROMPT_TEMPLATE` string only.
No logic, no schema, no other files.

---

## Success Criteria

Re-run Phase A on 5–10 objects and verify:

1. No single `target_style` token appears in >20% of global edits across the sample.
2. No surface-material word (gold, metal, clay, stone, glass, wood, rubber, plastic)
   appears in any `target_style`.
3. No aesthetic/style word (cartoon, vintage, futuristic, minimalist) appears in any
   `target_material`.
4. Each object's multiple global edits use styles from distinct categories.

Verification command:
```bash
python3 -c "
import json, glob
for p in glob.glob('outputs/partverse/*/objects/*/*/phase1/parsed.json'):
    d = json.load(open(p))
    obj = d['parsed']['object']['full_desc'][:40]
    for e in d['parsed']['edits']:
        et = e['edit_type']
        if et == 'global':
            print(f'GLB | {e[\"edit_params\"].get(\"target_style\",\"\")} | {obj}')
        elif et == 'material':
            print(f'MAT | {e[\"edit_params\"].get(\"target_material\",\"\")} | {obj}')
" | sort
```

---

## Out of Scope

- Splitting global into sub-types
- Per-object style sampling from a fixed vocabulary pool
- Any changes outside `USER_PROMPT_TEMPLATE`
