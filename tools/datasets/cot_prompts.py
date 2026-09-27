"""Annotation prompts for the ECOT schema consumed by MiBot.

The object proposal follows the existing VLM-proposal pipeline. The planning
and rationale prompts use the schema of the released collaborator annotations.
"""

SYSTEM_PROMPT = "You annotate robot demonstrations from visual and recorded state evidence. Return JSON only."

PROPOSAL_PROMPT = """List the visible, localizable objects in this image.
Merge synonyms and repeated mentions into canonical object names. Exclude the
robot arm, gripper, fingers and other robot parts. Do not invent unseen objects
or provide bounding boxes. Return {"objects": ["object name", ...]}.
"""

PLANNING_PROMPT = """Annotate this complete robot demonstration. The context
contains the task instruction, the episode length, sampled EEF poses, gripper
states, movement descriptions and any available object detections. Images are
ordered and labelled with their original, zero-based episode frame indices.

Identify the observed goal and divide the demonstration into meaningful
subtasks. Describe what actually happens, including failed or partial actions.
Use the visual and gripper evidence together: proximity alone is not evidence
of grasping. A nearly constant gripper signal is not evidence of a new grasp.
Keep names and object references consistent throughout the episode. Do not
invent action details that are not supported by the provided observations.

Return this JSON object, with no markdown:
{
  "goal": "A concise instruction describing the observed task",
  "global_plan": "A concise description of the observed sequence",
  "subtasks": [
    {"id": 1, "name": "Approach the target", "start": 0, "end": 12,
     "planning": "A short description of this segment and its purpose"}
  ]
}

The example indices are illustrative. Choose indices for THIS episode.
Subtask ids start at 1. start and end are inclusive integer frame indices.
Subtasks must cover every frame from 0 to num_frames - 1 exactly once, in
order, with no gaps or overlaps. Use English for all annotation text.
"""

REASONING_PROMPT = """Annotate the current subtask of a robot demonstration.
Use its start/middle/end images, task goal, full plan and recorded EEF/gripper
evidence. Write a short task rationale describing the visible situation, the
intended next manipulation and why it contributes to the goal. Ground it in
the supplied observations. Do not infer a grasp from proximity or assert an
unobserved success. This rationale will label the frames in this subtask;
avoid claiming that a segment's final outcome has already happened at its start.
Return only {"subtask_reasoning": "..."}, in English.
"""
