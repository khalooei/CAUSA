"""Wrapper around Qwen2-Audio-7B-Instruct for black-box grounding tests.

Exposes two query modes on the same underlying model:
  - `score_yes_no`: a forced-choice log-probability score for "Is there a
    sound of X?" style questions, obtained from a single forward pass
    (no sampling). This is what the causal necessity/sufficiency tests
    and the AAD-style baseline both consume, so all methods are compared
    on identical, deterministic evidence.
  - `generate`: free-form text generation, kept for sanity-checking model
    outputs against the forced-choice score.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

QUESTION_TEMPLATE = "Is there a sound of {object} in the audio?"
SAMPLE_RATE = 16000

_YES_VARIANTS = ["Yes", " Yes", "yes", " yes"]
_NO_VARIANTS = ["No", " No", "no", " no"]


@dataclasses.dataclass
class YesNoScore:
    p_yes: float
    p_no: float
    logit_yes: float
    logit_no: float

    @property
    def claim(self) -> str:
        return "yes" if self.p_yes >= self.p_no else "no"


class Qwen2AudioLALM:
    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.model.eval()
        self.device = device
        self._yes_ids = self._first_token_ids(_YES_VARIANTS)
        self._no_ids = self._first_token_ids(_NO_VARIANTS)

    def _first_token_ids(self, variants: list[str]) -> list[int]:
        tok = self.processor.tokenizer
        ids = set()
        for v in variants:
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _build_inputs(self, audios: list[np.ndarray | None], questions: list[str]):
        """Builds one batch from N (audio-or-None, question) pairs. A batch
        must be homogeneous - either every item has audio or none do -
        since the processor aligns audio arrays to text positionally and
        can't interleave the two. Tokenizer padding is left-sided, so the
        last real token of every sequence lands at position -1 regardless
        of length, which is what makes batched next-token scoring simple.
        """
        has_audio = [a is not None for a in audios]
        assert all(has_audio) or not any(has_audio), \
            "batch must be all-audio or all-text-only, can't mix"
        texts = []
        for audio, q in zip(audios, questions):
            content = [{"type": "text", "text": q}]
            if audio is not None:
                content.insert(0, {"type": "audio", "audio_url": "placeholder"})
            conversation = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            ))
        kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if has_audio[0]:
            kwargs["audio"] = list(audios)
            kwargs["sampling_rate"] = SAMPLE_RATE
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _scores_from_logits(self, logits: torch.Tensor) -> list[YesNoScore]:
        """logits: [batch, vocab] (already sliced to the last real token
        of each sequence)."""
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        logit_yes = torch.logsumexp(log_probs[:, self._yes_ids], dim=-1)
        logit_no = torch.logsumexp(log_probs[:, self._no_ids], dim=-1)
        m = torch.maximum(logit_yes, logit_no)
        z = torch.exp(logit_yes - m) + torch.exp(logit_no - m)
        p_yes = torch.exp(logit_yes - m) / z
        p_no = torch.exp(logit_no - m) / z
        return [
            YesNoScore(p_yes=float(p_yes[i]), p_no=float(p_no[i]),
                       logit_yes=float(logit_yes[i]), logit_no=float(logit_no[i]))
            for i in range(logits.shape[0])
        ]

    @torch.inference_mode()
    def score_yes_no(self, audio: np.ndarray | None, object_name: str,
                      question: str | None = None) -> YesNoScore:
        """Forced-choice yes/no log-prob at the first generated token.

        `audio=None` skips the audio content block entirely (a genuine
        text-only prompt, not a zero-waveform), giving the model's
        language-only prior for this object/question with no audio
        branch invoked at all - a purer control than the all-zero
        "blank audio" condition, which still runs the encoder.
        """
        return self.score_yes_no_batch([audio], [object_name], [question])[0]

    @torch.inference_mode()
    def score_yes_no_batch(self, audios: list[np.ndarray | None], object_names: list[str],
                            questions: list[str | None] | None = None) -> list[YesNoScore]:
        """Batched form of `score_yes_no`: one forward pass for N items
        instead of N sequential ones. Items must be homogeneous (all-audio
        or all-text-only) - see `_build_inputs`.

        CAUTION - not bit-exact with sequential scoring: testing found up
        to ~0.03 absolute difference in `p_yes` for a minority of items
        when compared against `score_yes_no` called one at a time on the
        same inputs, most likely bf16 batched-matmul numerical
        instability (different batch sizes take different CUDA kernel/
        tiling paths under reduced precision). Batch size 1 is unaffected
        (nothing to batch against) - only genuinely multi-item batches
        show it. Safe for quick exploratory checks; every result that
        gets reported in this project's outputs is computed sequentially
        instead, deliberately, to avoid that risk.
        """
        qs = questions or [None] * len(audios)
        qs = [q or QUESTION_TEMPLATE.format(object=o) for q, o in zip(qs, object_names)]
        inputs = self._build_inputs(audios, qs)
        out = self.model(**inputs)
        last_logits = out.logits[:, -1, :]
        return self._scores_from_logits(last_logits)

    @torch.inference_mode()
    def generate(self, audio: np.ndarray, question: str, max_new_tokens: int = 64) -> str:
        inputs = self._build_inputs([audio], [question])
        input_len = inputs["input_ids"].shape[1]
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = gen_ids[:, input_len:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
