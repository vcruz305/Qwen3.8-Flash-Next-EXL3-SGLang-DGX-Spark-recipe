#!/usr/bin/env python3
"""Share target embed_tokens + EXL3 lm_head into the EAGLE draft.

EXL3 ParallelLMHead has trellis/suh/svh, not .weight. The pin's
init_lm_head calls get_embed_and_head() and crashes. The pack also has no
mtp.embed_* tensors — skip sharing embeddings and accept rate stays 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = '''    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        target_lm_head = getattr(self.target_worker.model_runner.model, "lm_head", None)
'''

NEW = '''    def init_lm_head(self):
        target_model = self.target_worker.model_runner.model
        target_lm_head = getattr(target_model, "lm_head", None)
        draft = self.draft_runner.model
        embed = None
        try:
            embed = target_model.model.embed_tokens.weight
        except Exception:
            try:
                embed = target_model.embed_tokens.weight
            except Exception:
                embed = None
        if embed is not None:
            if hasattr(draft, "set_embed"):
                draft.set_embed(embed)
            elif hasattr(draft, "model") and hasattr(draft.model, "embed_tokens"):
                try:
                    del draft.model.embed_tokens.weight
                except Exception:
                    pass
                draft.model.embed_tokens.weight = embed
        # EXL3 ParallelLMHead has trellis/suh/svh, not .weight. Share the module.
        if target_lm_head is not None and not hasattr(target_lm_head, "weight"):
            if hasattr(draft, "set_lm_head_from_target"):
                draft.set_lm_head_from_target(target_lm_head)
                return
        embed, head = target_model.get_embed_and_head()
'''


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text()
    if "set_lm_head_from_target(target_lm_head)" in text and "embed_tokens.weight" in text:
        print(f"already patched {path}")
        return
    if OLD not in text:
        raise SystemExit(f"init_lm_head pattern missing in {path}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
