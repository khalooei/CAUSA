"""Wrapper around NVIDIA Audio Flamingo 3 (nvidia/audio-flamingo-3-hf) for
the same black-box grounding tests as Qwen2AudioLALM (see lalm.py).

A second, architecturally distinct LALM (Whisper-style encoder + Qwen2.5-7B
decoder, vs. Qwen2-Audio's own encoder/decoder pairing, different training
data/recipe) used to check whether this project's central finding -
localized ablation barely beats a random-window control, dominated by
removal quantity not location - is specific to Qwen2-Audio or holds more
generally. Exposes the identical `score_yes_no` interface so it drops
into `pipeline.GroundingPipeline` unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from causa.lalm import YesNoScore

SAMPLE_RATE = 16000

# AF3-specific: unlike Qwen2-Audio, AF3 defaults toward hedging/"No" under
# forced first-token extraction on the bare "Is there a sound of X in the
# audio?" phrasing (confirmed empirically: 0/40 "yes" claims on a random
# sample including unambiguous positives). An explicit answer-format
# instruction fixes this cleanly without changing the underlying question.
QUESTION_TEMPLATE = "Is there a sound of {object} in the audio? Answer with only the word Yes or No."

_YES_VARIANTS = ["Yes", " Yes", "yes", " yes"]
_NO_VARIANTS = ["No", " No", "no", " no"]


class AudioFlamingo3LALM:
    def __init__(self, model_id: str = "nvidia/audio-flamingo-3-hf", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
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

    def _build_inputs(self, audio: np.ndarray | None, question: str):
        # AF3's own documented examples order content as [text, audio] -
        # matching that (not [audio, text], as Qwen2-Audio's convention is)
        # matters for this model's behavior, confirmed empirically.
        content = [{"type": "text", "text": question}]
        if audio is not None:
            content.append({"type": "audio", "path": "placeholder"})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        kwargs = dict(text=text, return_tensors="pt")
        if audio is not None:
            kwargs["audio"] = [audio]
        inputs = self.processor(**kwargs)
        # unlike Qwen2Audio's processor, AF3's audio conv front-end does not
        # auto-cast float inputs to the model's dtype - input_features must
        # be cast explicitly or the conv1d bias/input dtypes mismatch.
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
        out = self.model(**inputs)
        last_logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)
        logit_yes = torch.logsumexp(log_probs[self._yes_ids], dim=0).item()
        logit_no = torch.logsumexp(log_probs[self._no_ids], dim=0).item()
        m = max(logit_yes, logit_no)
        z = np.exp(logit_yes - m) + np.exp(logit_no - m)
        p_yes = float(np.exp(logit_yes - m) / z)
        p_no = float(np.exp(logit_no - m) / z)
        return YesNoScore(p_yes=p_yes, p_no=p_no, logit_yes=logit_yes, logit_no=logit_no)
