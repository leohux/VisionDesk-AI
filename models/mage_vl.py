"""Mage-VL narrator — image understanding for DualBrain deepen (no boxes)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from core.types import DetectionResult

DEFAULT_MODEL = os.environ.get("MAGE_VL_PATH", "microsoft/Mage-VL")
DEFAULT_QUESTION = "Briefly describe what is happening in this scene."


class MageVLNarrator:
    """Duck-typed deep detector: predict(roi) → DetectionResult with summary only."""

    backend_name = "mage-vl"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        max_side: int = 960,
        max_new_tokens: int = 256,
        question: str = DEFAULT_QUESTION,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = int(max_side)
        self.max_new_tokens = int(max_new_tokens)
        self.question = question or DEFAULT_QUESTION
        self._context_yolo = ""
        self._context_reason = ""

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading Mage-VL from {self.model_path} on {self.device}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Prefer SDPA so flash-attn is not required.
        kwargs: dict = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "attn_implementation": "sdpa",
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto", **kwargs
            ).eval()
        except TypeError:
            # Older transformers may not accept attn_implementation
            kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto", **kwargs
            ).eval()
        print("Mage-VL ready.", flush=True)

    def set_context(self, yolo_summary: str = "", reason: str = "") -> None:
        """Optional DualBrain context injected before predict()."""
        self._context_yolo = (yolo_summary or "").strip()
        self._context_reason = (reason or "").strip()

    def _build_question(self) -> str:
        parts = [self.question]
        if self._context_yolo:
            parts.append(f"YOLO saw: {self._context_yolo}")
        if self._context_reason:
            parts.append(f"Trigger: {self._context_reason}")
        return " ".join(parts)

    def _prepare(self, frame_bgr: np.ndarray) -> Image.Image:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> DetectionResult:
        pil = self._prepare(frame_bgr)
        question = self._build_question()
        t0 = time.perf_counter()
        answer = self._generate(pil, question)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        summary = (answer or "").strip() or "(empty Mage-VL response)"
        return DetectionResult(
            boxes=[],
            summary=summary,
            backend="mage-vl",
            infer_ms=infer_ms,
            extras={"raw_preview": summary[:400], "question": question},
        )

    def _generate(self, image: Image.Image, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")], return_tensors="pt"
        )
        # Move tensors onto the model device / dtype
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = torch.device(self.device)

        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = v.to(model_device)
            else:
                moved[k] = v
        if "pixel_values" in moved and hasattr(moved["pixel_values"], "to"):
            dtype = getattr(self.model, "dtype", None)
            if dtype is not None:
                moved["pixel_values"] = moved["pixel_values"].to(dtype)

        output = self.model.generate(
            **moved, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        prompt_len = moved["input_ids"].shape[1]
        decoded = self.processor.tokenizer.decode(
            output[0, prompt_len:], skip_special_tokens=True
        )
        return decoded.strip()
