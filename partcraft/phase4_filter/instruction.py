"""Phase 4: Edit instruction generation from edit specs."""

from __future__ import annotations

import random
from partcraft.phase1_planning.planner import EditSpec


# Instruction templates per edit type
_DELETION_TEMPLATES = [
    "Remove the {part} from the {obj}",
    "Delete the {part}",
    "Take away the {part} of the {obj}",
    "The {obj} should not have the {part}",
    "Get rid of the {part} on the {obj}",
]

_ADDITION_TEMPLATES = [
    "Add {part} to the {obj}",
    "Attach {part} to the {obj}",
    "Put {part} on the {obj}",
    "Give the {obj} a {part}",
    "The {obj} should have {part}",
]

_MODIFICATION_TEMPLATES = [
    "Replace the {old} with {new}",
    "Change the {old} to {new}",
    "Swap the {old} for {new}",
    "Use {new} instead of the {old}",
    "Substitute the {old} with {new}",
]

_GRAFT_TEMPLATES = [
    "Add {part} from the {donor} to the {obj}",
    "Transplant {part} onto the {obj}",
    "Give the {obj} the {part} of a {donor}",
    "Attach a {part} (like the one on a {donor}) to the {obj}",
]


def _humanize_label(label: str) -> str:
    """Convert 'chair_leg' → 'chair leg'."""
    return label.replace("_", " ").strip()


def generate_instructions(spec: EditSpec, n_variants: int = 3) -> list[str]:
    """Generate n instruction variants for an edit spec.

    For deletion: object_desc describes the 'before' state (complete) — used as-is.
    For addition: object_desc describes the complete object, but 'before' state
                  is missing the part, so we strip the part mention from the desc.
    """
    obj_full = _humanize_label(spec.object_desc) or "object"
    # For addition: before_desc describes the object WITHOUT the part (from VLM)
    obj_before = _humanize_label(spec.before_desc) if spec.before_desc else obj_full
    instructions = []

    if spec.edit_type == "deletion":
        part = _humanize_label(", ".join(spec.remove_labels)) or "part"
        templates = _DELETION_TEMPLATES
        for tmpl in random.sample(templates, min(n_variants, len(templates))):
            instructions.append(tmpl.format(part=part, obj=obj_full))

    elif spec.edit_type == "addition":
        if spec.new_obj_id:
            # Graft from another object
            part = _humanize_label(spec.new_label) or "part"
            donor = spec.new_obj_id[:8]  # truncated ID as placeholder
            templates = _GRAFT_TEMPLATES
            for tmpl in random.sample(templates, min(n_variants, len(templates))):
                instructions.append(tmpl.format(part=part, obj=obj_before, donor=donor))
        else:
            part = _humanize_label(", ".join(spec.add_labels)) or "part"
            templates = _ADDITION_TEMPLATES
            for tmpl in random.sample(templates, min(n_variants, len(templates))):
                instructions.append(tmpl.format(part=part, obj=obj_before))

    elif spec.edit_type == "modification":
        old = _humanize_label(spec.old_label) or "old part"
        new = _humanize_label(spec.new_label) or "new part"
        templates = _MODIFICATION_TEMPLATES
        for tmpl in random.sample(templates, min(n_variants, len(templates))):
            instructions.append(tmpl.format(old=old, new=new, obj=obj_full))

    return instructions
