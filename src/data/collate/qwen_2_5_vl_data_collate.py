import itertools

import torch
from transformers import AutoProcessor


class CustomCollate:
    def __init__(self, max_length=4096):
        """
        Initialize collate function with sequence length limit

        Args:
            max_length (int): Maximum allowed cumulative sequence length
        """
        self.max_length = max_length

    def _select_samples(self, batch):
        """
        Greedily select samples in order until max_length exceeded

        Args:
            batch (List): List of samples with vlm_input_ids field

        Returns:
            List: Subset of samples that fit within max_length
        """
        selected_samples = []
        total_length = 0

        for item in batch:
            item_length = item["vlm_input_ids"].size(1)

            # If adding this sample would exceed max_length, stop
            if total_length + item_length > self.max_length:
                continue

            # Otherwise, add the sample
            selected_samples.append(item)
            total_length += item_length

        return selected_samples

    def process_action_fields(self, batch_selected):
        """
        Process non-VLM fields from action samples

        Args:
            batch_selected (List): Filtered samples that fit within max_length

        Returns:
            Dict: Collated non-VLM tensors
        """
        action_items = [item for item in batch_selected if "action" in item]
        if not action_items:
            return {}

        # Extract non-VLM keys from first action item
        non_vlm_keys = [k for k in action_items[0].keys() if not k.startswith("vlm_")]

        return {key: torch.stack([torch.from_numpy(item[key]) for item in action_items]) for key in non_vlm_keys}

    def process_vlm(self, batch_selected):
        """
        Process VLM fields from selected samples

        Args:
            batch_selected (List): Filtered samples that fit within max_length

        Returns:
            Dict: Collated VLM tensors and metadata
        """
        if not batch_selected:
            return self._empty_vlm_dict()

        # Get sequence lengths of selected samples
        item_lengths = [item["vlm_input_ids"].size(1) for item in batch_selected]

        # Build cumulative lengths
        cumsum_seq_lens = [0] + list(itertools.accumulate(item_lengths))

        # Build action segments
        action_segments = [
            [cumsum_seq_lens[i], cumsum_seq_lens[i + 1]] for i, item in enumerate(batch_selected) if "action" in item
        ]
        action_vlm_condition_segments = [
            [cumsum_seq_lens[i], cumsum_seq_lens[i] + item["vlm_action_condition_length"]] for i, item in enumerate(batch_selected) if "action" in item
        ]

        # Merge tensors
        input_ids = torch.cat([item["vlm_input_ids"] for item in batch_selected], dim=1)
        labels = torch.cat([item["vlm_labels"] for item in batch_selected], dim=1)
        position_ids = torch.cat([item["vlm_position_ids"] for item in batch_selected], dim=2)

        vlm_action_target = self.extract(batch_selected, "vlm_action_target")
        vlm_action_mask = self.extract(batch_selected, "vlm_action_mask")
        vlm_action_actual_length = self.extract(batch_selected, "vlm_action_actual_length")

        # Extract multimedia data
        images = self.extract(batch_selected, "vlm_pixel_values")
        videos = self.extract(batch_selected, "vlm_pixel_values_videos")
        image_grid_thw = self.extract(batch_selected, "vlm_image_grid_thw")
        video_grid_thw = self.extract(batch_selected, "vlm_video_grid_thw")

        # Build result dictionary
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "pixel_values": torch.cat(images) if images else None,
            "pixel_values_videos": torch.cat(videos) if videos else None,
            "image_grid_thw": torch.cat(image_grid_thw) if image_grid_thw else None,
            "video_grid_thw": torch.cat(video_grid_thw) if video_grid_thw else None,
            "cu_seq_lens_q": torch.tensor(cumsum_seq_lens, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(cumsum_seq_lens, dtype=torch.int32),
            "max_length_q": max(item_lengths) if item_lengths else 0,
            "max_length_k": max(item_lengths) if item_lengths else 0,
            "action_segments": torch.tensor(action_segments, dtype=torch.int64)
            if action_segments
            else torch.tensor([[0, 5]], dtype=torch.int64),
            "action_vlm_condition_segments": torch.tensor(action_vlm_condition_segments, dtype=torch.int64)
            if action_vlm_condition_segments
            else torch.tensor([[0, 5]], dtype=torch.int64),
            "vlm_action_target": torch.cat(vlm_action_target) if vlm_action_target else None,
            "vlm_action_mask": torch.cat(vlm_action_mask) if vlm_action_mask else None,
            "vlm_action_actual_length": torch.tensor(vlm_action_actual_length) if vlm_action_actual_length else None,
        }

    def _empty_vlm_dict(self):
        """Return an empty VLM dictionary for edge cases"""
        return {
            "input_ids": torch.tensor([]),
            "labels": torch.tensor([]),
            "position_ids": torch.tensor([]),
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "cu_seq_lens_q": torch.tensor([0], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0], dtype=torch.int32),
            "max_length_q": 0,
            "max_length_k": 0,
            "action_segments": torch.empty(0, 2, dtype=torch.int64),
            "action_vlm_condition_segments": torch.empty(0, 2, dtype=torch.int64),
            "vlm_action_target": None,
            "vlm_action_mask": None,
            "vlm_action_actual_length": None
        }

    @staticmethod
    def extract(batch, key):
        res = []
        for item in batch:
            if key in item:
                res.append(item[key])
        return res

    def __call__(self, batch):
        """
        Process a batch of samples with greedy length filtering

        Args:
            batch (List): Raw batch of samples

        Returns:
            Dict: Fully processed batch dictionary
        """
        # 1. Greedily select samples that fit within max_length
        batch_selected = self._select_samples(batch)

        # 2. Process VLM fields
        processed_vlm = self.process_vlm(batch_selected)

        # 3. Process non-VLM fields
        collated_non_vlm = self.process_action_fields(batch_selected)

        # 4. Combine results
        return {**collated_non_vlm, **processed_vlm}
