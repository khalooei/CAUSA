"""Wrapper around Qwen2.5-Omni-7B for the same black-box grounding tests
as Qwen2AudioLALM / AudioFlamingo3LALM (see lalm.py, af3_lalm.py).

A third, newer LALM (2025 release, successor to Qwen2-Audio, adds a
"talker" speech-generation head we don't need and explicitly disable) -
added to test whether the cross-model heterogeneity this project's
central finding rests on (localized ablation barely beats random for
Qwen2-Audio, clearly beats it for AF3) is a two-point coincidence or
holds up against a third, architecturally related but materially newer
model. Exposes the identical `score_yes_no` interface so it drops into
`pipeline.GroundingPipeline` unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from causa.lalm import YesNoScore

SAMPLE_RATE = 16000
QUESTION_TEMPLATE = "Is there a sound of {object} in the audio?"

_YES_VARIANTS = ["Yes", " Yes", "yes", " yes"]
_NO_VARIANTS = ["No", " No", "no", " no"]


class Qwen25OmniLALM:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Omni-7B", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.model.disable_talker()  # text-only use, saves ~2GB, no speech synthesis needed
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

    def _build_inputs(self, audio: np.ndarray | None, question: str):
        content = [{"type": "text", "text": question}]
        if audio is not None:
            content.insert(0, {"type": "audio", "audio": "placeholder"})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        kwargs = dict(text=text, return_tensors="pt")
        if audio is not None:
            kwargs["audio"] = [audio]
        inputs = self.processor(**kwargs)
        out = {}
        for k, v in inputs.items():
            v = v.to(self.device)
            if torch.is_floating_point(v):
                v = v.to(self.model.dtype)
            out[k] = v
        return out

    @torch.inference_mode()
    def score_yes_no(self, audio: np.ndarray | None, object_name: str,
                      question: str | None = None) -> YesNoScore:
        """Identical semantics to Qwen2AudioLALM.score_yes_no (see lalm.py):
        forced-choice yes/no log-prob at the first generated token, single
        deterministic forward pass, `audio=None` for a genuine text-only
        (not blank-waveform) query."""
        q = question or QUESTION_TEMPLATE.format(object=object_name)
        inputs = self._build_inputs(audio, q)
        # the top-level Qwen2_5OmniForConditionalGeneration has no forward of
        # its own (it's a thinker+talker container) - the thinker submodule
        # is the actual causal LM and is what has .logits
        out = self.model.thinker(**inputs)
        last_logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)
        logit_yes = torch.logsumexp(log_probs[self._yes_ids], dim=0).item()
        logit_no = torch.logsumexp(log_probs[self._no_ids], dim=0).item()
        m = max(logit_yes, logit_no)
        z = np.exp(logit_yes - m) + np.exp(logit_no - m)
        p_yes = float(np.exp(logit_yes - m) / z)
        p_no = float(np.exp(logit_no - m) / z)
        return YesNoScore(p_yes=p_yes, p_no=p_no, logit_yes=logit_yes, logit_no=logit_no)

    @torch.inference_mode()
    def generate(self, audio: np.ndarray, question: str, max_new_tokens: int = 64) -> str:
        inputs = self._build_inputs(audio, question)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False, return_audio=False)
        if isinstance(gen_ids, tuple):
            gen_ids = gen_ids[0]
        gen_ids = gen_ids[:, input_len:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
