"""Edit type definitions for PartCraft3D.

Centralizes all edit type constants, metadata, and routing logic.
Each edit type maps to a specific TRELLIS execution strategy.

Taxonomy (from the 3D editing data design doc):
  1.1 Topological Evolution   — deletion, addition, modification (swap)
  1.2 Geometric Deformation   — scale (anisotropic part scaling)
  1.3 Attribute Decoupling     — material (part-level texture), global (whole-object style)
  1.5 Identity & Negatives     — identity (no-op, anti-hallucination)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Edit type constants
# ---------------------------------------------------------------------------

# 1.1 Topological Evolution
DELETION = "deletion"          # Remove part(s) from object — GT mesh removal
ADDITION = "addition"          # Add part(s) back — reverse of deletion
MODIFICATION = "modification"  # Swap part shape — TRELLIS S1+S2 repaint

# 1.2 Geometric Deformation
SCALE = "scale"                # Anisotropic part scaling — TRELLIS S1+S2 repaint

# 1.3 Attribute Decoupling
MATERIAL = "material"          # Part-level material/texture change — S2 only
GLOBAL = "global"              # Whole-object style/theme — S2 only (full mask)

# 1.5 Identity & Negatives
IDENTITY = "identity"          # No-op: input=output, irrelevant instruction

ALL_TYPES = {DELETION, ADDITION, MODIFICATION, SCALE, MATERIAL, GLOBAL, IDENTITY}

# ---------------------------------------------------------------------------
# Edit type → TRELLIS execution strategy
# ---------------------------------------------------------------------------

# Types that use TRELLIS Flow Inversion + S1+S2 repaint (geometry changes)
S1_S2_TYPES = {MODIFICATION, SCALE}

# Types that use TRELLIS S2 only (appearance changes, preserve geometry)
S2_ONLY_TYPES = {MATERIAL, GLOBAL}

# Types that use GT mesh operations (no TRELLIS generation)
MESH_ONLY_TYPES = {DELETION}

# Types that need no generation at all
NO_GEN_TYPES = {IDENTITY, ADDITION}

# Types that need a part mask (not full 64³)
PART_MASK_TYPES = {MODIFICATION, SCALE, MATERIAL, DELETION}

# Types that use full 64³ mask
FULL_MASK_TYPES = {GLOBAL}

# ---------------------------------------------------------------------------
# Edit ID prefixes
# ---------------------------------------------------------------------------

ID_PREFIX = {
    DELETION: "del",
    ADDITION: "add",
    MODIFICATION: "mod",
    SCALE: "scl",
    MATERIAL: "mat",
    GLOBAL: "glb",
    IDENTITY: "idt",
}

# ---------------------------------------------------------------------------
# TRELLIS effective type mapping
# ---------------------------------------------------------------------------

def trellis_effective_type(edit_type: str) -> str:
    """Map PartCraft edit type → interweave_Trellis_TI edit_type string.

    interweave_Trellis_TI understands: Modification, Addition, TextureOnly,
    Deletion, HybridDeletion.  We map our higher-level types to these.
    """
    if edit_type in (MODIFICATION, SCALE):
        return "Modification"
    if edit_type in (MATERIAL, GLOBAL):
        return "TextureOnly"
    if edit_type == DELETION:
        return "Deletion"
    if edit_type == ADDITION:
        return "Addition"
    return "Modification"  # fallback


# ---------------------------------------------------------------------------
# Processing order (for streaming pipeline)
# ---------------------------------------------------------------------------

TYPE_ORDER = {
    DELETION: 0,
    MODIFICATION: 1,
    SCALE: 2,
    MATERIAL: 3,
    GLOBAL: 4,
    IDENTITY: 5,
    # addition handled separately (after all deletions)
}

# ---------------------------------------------------------------------------
# Programmatic edit templates
# ---------------------------------------------------------------------------

# Scale edit templates: (prompt_template, before_template, after_template)
# {part} is replaced with a natural-language part phrase (record ``desc`` or
# humanized ``label``) in plan_edits_for_record.
SCALE_TEMPLATES = [
    ("Make the {part} taller",
     "{part}", "taller {part}"),
    ("Make the {part} shorter",
     "{part}", "shorter compact {part}"),
    ("Make the {part} wider",
     "{part}", "wider {part}"),
    ("Make the {part} thinner",
     "{part}", "thinner slender {part}"),
    ("Make the {part} larger",
     "{part}", "larger {part}"),
    ("Make the {part} smaller and more compact",
     "{part}", "smaller compact {part}"),
    ("Stretch the {part} longer",
     "{part}", "elongated {part}"),
    ("Make the {part} thicker and sturdier",
     "{part}", "thicker sturdier {part}"),
]

# Material edit templates: (prompt_template, after_part_desc_template)
# {part} is replaced with the same natural-language phrase as scale templates.
MATERIAL_TEMPLATES = [
    ("Change the {part} to wooden material",
     "wooden {part}"),
    ("Make the {part} metallic chrome",
     "chrome metallic {part}"),
    ("Change the {part} to glass material",
     "transparent glass {part}"),
    ("Make the {part} look like stone",
     "stone carved {part}"),
    ("Change the {part} to rusty iron",
     "rusty corroded iron {part}"),
    ("Make the {part} golden and polished",
     "golden polished {part}"),
    ("Change the {part} to matte black rubber",
     "matte black rubber {part}"),
    ("Make the {part} ceramic and glossy",
     "glossy ceramic {part}"),
]

# Identity: irrelevant instructions (object should remain unchanged)
IDENTITY_PROMPTS = [
    "Make the sky brighter",
    "Change the background color to blue",
    "Add more sunlight to the scene",
    "Rotate the camera angle slightly",
    "Increase the ambient lighting",
    "Make the shadows softer",
    "Add a subtle glow effect",
    "Adjust the white balance",
]
