"""VLM proposals and LLMDet grounding, following the existing annotation scripts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cot_prompts import PROPOSAL_PROMPT


def propose_objects(client, frame_paths: list[Path]) -> list[str]:
    path = frame_paths[len(frame_paths) // 2]
    with Image.open(path) as image:
        result = client.complete(PROPOSAL_PROMPT, {}, [(int(path.stem.split("_")[-1]), image)])
    objects = result.get("objects")
    if not isinstance(objects, list) or any(not isinstance(x, str) or not x.strip() for x in objects):
        raise ValueError("VLM proposal must return an objects list")
    return list(dict.fromkeys(name.strip() for name in objects))


class LLMDetGrounding:
    """Use the same Transformers LLMDet interface as the collaborator's writers."""

    def __init__(self, model_path: str, device: str, threshold: float = 0.3):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device).eval()

    def annotate(self, cot: dict, frame_paths: list[Path], objects: list[str]) -> None:
        import torch

        for path in frame_paths:
            t = int(path.stem.split("_")[-1])
            visible = []
            with Image.open(path) as source:
                image = source.convert("RGB")
                # Keep the source pipeline's bounded text batches and 0..1000 boxes.
                for offset in range(0, len(objects), 10):
                    labels = objects[offset:offset + 10]
                    inputs = self.processor(images=image, text=[[f"a {label}" for label in labels]], return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        output = self.model(**inputs)
                    result = self.processor.post_process_grounded_object_detection(
                        output, input_ids=inputs.input_ids, threshold=self.threshold,
                        target_sizes=[(image.height, image.width)],
                    )[0]
                    detected_labels = result.get("text_labels", result.get("labels"))
                    for box, label in zip(result["boxes"], detected_labels):
                        if not isinstance(label, str):
                            label = labels[int(label)]
                        coords = box.tolist()
                        normalized = [round(1000 * value / size) for value, size in zip(coords, [image.width, image.height] * 2)]
                        visible.append({"entity": label.removeprefix("a ").strip(), "bbox_xyxy": [min(1000, max(0, x)) for x in normalized]})
            cot["frames"][t].setdefault("Grounding", {})["visible_objects"] = {"cam0": visible}
        cot["bbox_source"] = "llmdet"
