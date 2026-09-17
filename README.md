# Qwen3.8-Flash-Next EXL3 on one NVIDIA DGX Spark -- SGLang

Serves [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
on a single NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64) through
**SGLang** and the [sglang-exl3](https://github.com/vcruz305/sglang-exl3)
plugin.

This is the **SGLang** sibling of
[Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe)
(vLLM). Same model, same pack, different engine -- pick this one if you are
integrating with SGLang upstream or need SGLang-specific features (radix
cache, its scheduler, its speculative-decoding stack).

## Status

Measured on one DGX Spark GB10 (2026-09-16): EAGLE/MTP 3/1/4 + ReplaySSM +
fused `exl3_moe` at **~35 tok/s** decode (`n=256`, ctx 4096). One known
idle-crash bug is still open (see below) -- read it before relying on this
for unattended production traffic.

## Hardware

```text
GPU:     NVIDIA GB10
Memory:  128 GB unified (aarch64)
Engine:  SGLang, --quantization exl3, --load-format exl3_mmap
```

One Spark. Single in-flight request (`--max-running-requests 1`) in the
measured configuration.

## Model

Qwen3.8-Flash-Next is a `Qwen4ExpForConditionalGeneration` model: 48 layers
(36 linear attention, 12 full attention with QSA sparse attention), 512
experts with top-3 routing, hidden size 2560, one MTP (multi-token
prediction / NEXTN) layer, a per-layer n-gram (PLE) embedding table
(320,001,536 rows x 160, K=5 packed, ~95 GiB file-offloaded), and a vision
tower.

## Prerequisites

- One DGX Spark (GB10, aarch64), NVMe with room for the ~87 GB pack plus the
  ~95 GiB PLE table (file-offloaded, not fully resident).
- A SGLang build with `Qwen4ExpForConditionalGeneration` and
  `qwen4_exp_mtp.py`. The 2026-09-16 numbers used pin commit **`4ccff141d`**
  (`lmsysorg/sglang:dev-qwen38-next-local` linux/arm64) + Torch **2.13.0+cu130**.
  Overlay that pin with `scripts/apply_sglang_pin_overlays.sh` (see below).
- The [`sglang-exl3`](https://github.com/vcruz305/sglang-exl3) plugin at
  **PR #3** (`fix/gb10-torch214-sglang-cli`, includes `b1d9eab`) or later.
  Need: `SGLANG_EXL3_MTP_EXL3`, mixed-K GDN/`lm_head` EXL3, **do not**
  EXL3-quantize `embed_tokens`, and the 35-arg `exl3_moe` call. Launch with
  `python -m sglang_exl3.launch_server` (bare `sglang.launch_server` never
  registers EXL3).
- `exllamav3` built against the same Torch (GB10: `TORCH_CUDA_ARCH_LIST=12.1a`).
  Current `exl3_moe` is 35 args (`num_active`, scratch, tables, count_lo/hi,
  `m_tile`). A 29-arg call TypeErrors and falls back to the python expert
  loop (~10 tok/s).
- Do **not** set `EXL3_FUSED_MOE=0` or `SGLANG_EXL3_ALL_LINEARS=1`.
  Plugin `native` MoE expects 4096x2048; this pack is 2560x640 -- keep
  `SGLANG_EXL3_MOE_KERNEL=exllamav3`.

## Quick start

### 1. Install the runtime

Install SGLang (Qwen4-Exp pin), `exllamav3` from source against that Torch,
and the `sglang-exl3` plugin per Prerequisites. Overlay the pin, then confirm:

```bash
SGLANG_PIN=/path/to/sglang/python bash scripts/apply_sglang_pin_overlays.sh
python -c "import sglang; print(sglang.__version__)"
python -c "import sglang_exl3; print('sglang_exl3 OK')"
python -c "import exllamav3_ext; print(exllamav3_ext.exl3_moe.__doc__.split(chr(10))[0])"
```

### 2. Download the pack

```bash
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw
```

About 87 GB: safetensors shards, `ngram_embedding.safetensors`, MTP layer
weights, `config.json`, `model.safetensors.index.json`, tokenizer files.
SGLang's `exl3_mmap` load format reads the pack in its native layout --
unlike the vLLM recipe, **no pack-rewrite step is needed here.**

### 3. Serve

No draft (baseline):

```bash
MODEL_DIR=~/models/Qwen3.8-Flash-Next-exl3-3.05bpw \
  bash scripts/serve_one_spark_qwen_sglang.sh
```

With EAGLE/MTP draft (recommended -- faster, see Headline):

```bash
MODEL_DIR=~/models/Qwen3.8-Flash-Next-exl3-3.05bpw \
  bash scripts/serve_one_spark_qwen_sglang_mtp.sh
```

Both bind `127.0.0.1:30000` by default (override `PORT`). Load takes about
5-6 minutes for the target plus ~30-50s for the MTP draft. The MTP script
defaults to ReplaySSM on, `EXL3_FUSED_MOE=1`, `sglang_exl3.launch_server`.
See each script header for overrides (`CTX_LEN`, `MEM_FRAC`, `MAX_REQS`,
`PLE_OFFLOAD_DIR`, `SPEC_STEPS`, `SPEC_TOPK`, `SPEC_DRAFT_TOKENS`,
`ENABLE_REPLAYSSM`).

Optionally run `scripts/memory-guard.sh` alongside as a watchdog -- GB10 is
unified memory, so there is no separate VRAM pool to protect against OOM;
see the script's own header for tuning `FLOOR_GIB`/`GUARD_GIB`.

### 4. Benchmark

```bash
python scripts/bench_decode.py --base-url http://127.0.0.1:30000 \
  --model Qwen3.8-Flash-Next --max-tokens 256
```

`n=256` is `max_tokens` (generated tokens), not context. Measured ctx is
`--context-length 4096`. Wall tok/s includes TTFT; the scheduler
`gen throughput` line is decode-only.

## Headline (measured 2026-09-16, one DGX Spark GB10, ctx 4096)

Prompt `The capital of France is`. CUDA graphs off (PLE file offload).
Fused `exl3_moe` (35-arg ABI). EAGLE topk=1.

| Config | Accept rate | Accept len | tok/s | Notes |
|---|---|---|---|---|
| No draft | n/a | n/a | ~10 | python MoE loop, or fused off |
| MTP 3/1/4, accept 0 | 0.00 | 1.00 | ~5 | draft embeddings/fc not bound -- **slower than no-draft** |
| MTP 3/1/4, no ReplaySSM | ~0.62 | ~2.9 | ~13 | fused MoE |
| MTP 2/1/3 + ReplaySSM | ~0.75 | ~2.5 | ~13 | extra draft step did not pay |
| **MTP 3/1/4 + ReplaySSM + fused exl3_moe** | **~0.69** | **~3.1** | **~35 wall / ~36 gen (`n=256`)** | **recommended** |

Older 2026-09-10 row (~17 no-draft / 22-31 MTP) was the same 3/1/4 flags
before this pin's 35-arg `exl3_moe` and the embed-share overlay; treat the
2026-09-16 table as the one to reproduce.

## Pin overlays (required for the 35 tok/s path)

SGLang `4ccff141d` still has `nn.Linear` for `mtp.fc_embedding`/`fc_hidden`
and `eagle_worker_v2.init_lm_head` reading `.weight`. Apply once per pin:

```bash
SGLANG_PIN=/path/to/sglang/python bash scripts/apply_sglang_pin_overlays.sh
```

That copies `patches/qwen4_exp_mtp.py` (`ReplicatedLinear` + quant_config
prefix) and patches `init_lm_head` to (1) assign
`draft.model.embed_tokens.weight = target.model.embed_tokens.weight` and
(2) `set_lm_head_from_target` when the head has no `.weight`. Skipping (1)
leaves accept rate at 0.

## MTP fix: draft must load native EXL3, not dense weights

**Root cause (fixed in `sglang-exl3` PR #2, merged to master):** the
plugin's `get_quant_method` treated every MTP/`nextn`-prefixed layer as
FP8-or-BF16-only. Qwen3.8-Flash-Next's checkpoint embeds the MTP draft in
EXL3 too (`mtp_bits` in `quantization_config.json`, full trellis/mul1 packs
under `mtp.*`), so the draft's `fc_embedding`, `fc_hidden`, `qkv_proj`,
`o_proj`, shared-expert, and indexer weights all silently fell back to
random-initialized dense `nn.Linear`/`ParallelLMHead` tensors that were
never loaded from the checkpoint. The draft ran, but on garbage weights --
**accept rate was permanently 0.00**, and the extra draft-forward cost made
serving *slower* than no MTP at all (~11 tok/s vs ~17 tok/s baseline).

**Fix:** `SGLANG_EXL3_MTP_EXL3=1` (default after the fix) makes
`get_quant_method` return `EXL3LinearMethod`/`EXL3MoEMethod` for
MTP/`nextn`-prefixed layers, reading `mtp_bits` from the quant config. The
serve tree's `qwen4_exp_mtp.py` also needed `fc_embedding`/`fc_hidden`
switched from plain `nn.Linear` to `ReplicatedLinear(quant_config=...)`
with `prefix="mtp.fc_embedding"` / `"mtp.fc_hidden"` so those two layers'
EXL3 packs actually match a `params_dict` key.

**Also required (PR #3):** never EXL3-quantize `embed_tokens` /
`VocabParallelEmbedding`; share the target embedding tensor into the
draft; share the EXL3 `lm_head` module (no `.weight`). Call `exl3_moe`
with the 35-arg ABI. `EXL3_FUSED_MOE=0` or a 29-arg call drops to the
python expert loop and caps decode around 10–16 tok/s even with MTP.

## Known limitation: `speculative_eagle_topk` is hard-capped at 1

Qwen4-Exp's QSA (query-sparse-attention) MTP path raises

```text
NotImplementedError: Qwen4-Exp QSA MTP currently supports speculative_eagle_topk=1
```

for any `--speculative-eagle-topk` above 1. This is a real architectural
constraint in the current SGLang QSA-MTP implementation, not a tuning knob
-- do not attempt `topk>1` configs on this model.

## Known limitation: raising `speculative_num_steps` / `speculative_num_draft_tokens` crashes

The verified-working draft config is `speculative_num_steps=3
speculative_eagle_topk=1 speculative_num_draft_tokens=4`. Raising either
`speculative_num_steps` (tried 4, 5, 6) or `speculative_num_draft_tokens`
(tried 5, 6, 8) above that pair reliably crashes the scheduler with:

```text
torch.AcceleratorError: CUDA error: device-side assert triggered
.../IndexKernelUtils.cu:19: vectorized_gather_kernel:
  Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
```

surfacing inside the QSA indexer's `apply_rope` /
`_ensure_cos_sin_cache_length` path. Working hypothesis: the shared MTP
sparse-index buffer (`QSAMTPSharedSparseIndices` in
`sglang/srt/layers/attention/qwen_sparse_attn_backend.py`, sized with
`tail_width = speculative_num_steps + 1` in
`sglang/srt/speculative/eagle_worker_v2.py::_configure_qsa_mtp_index_share`)
is allocated once against the launch-time `speculative_num_steps` and does
not get re-sized if that value changes -- not confirmed with a synchronous
(`CUDA_LAUNCH_BLOCKING=1`) trace yet. **Do not raise these flags** without
first fixing that buffer's sizing.

## Open issue: idle-crash after ~60-70 minutes

**Reproduced twice, deterministically.** A server that has loaded cleanly
and served requests successfully then sits fully idle (zero traffic) for
roughly 60-70 minutes crashes on the very first request after that idle
window -- including a bare `/health` GET in one repro -- with the same CUDA
device-side assert as above (`vectorized_gather_kernel` index out of
bounds), but surfacing at a **different** line each time
(`update_key_state_and_compress`'s `source_keys[group_locs]` gather in one
trace, `apply_rope`'s `_ensure_cos_sin_cache_length` in another). The
scheduler process dies and the server becomes fully unreachable
(`curl /health` returns connection-refused) -- this is not a slow request,
it is a hard crash requiring a full relaunch.

This is independent of the `speculative_num_steps`/`topk` tuning issue
above -- both repros ran the stock, verified-working `steps=3 topk=1
draft_tokens=4` config. Not yet root-caused; PyTorch's own async-CUDA-error
warning ("the stacktrace below might be incorrect") means the two different
surfaced lines are likely both downstream of one earlier real fault, not
two separate bugs. Candidates not yet ruled out: the PLE table's
background resident-set eviction thread (runs every 30s, evicts down to an
8 GiB cap -- see `PLE table: resident set capped` in the server log) racing
with the QSA shared-index state, or a time/position-based counter overflow
in the shared MTP index buffer.

**Mitigation until root-caused:** don't leave an MTP-enabled server idle
for extended periods in production; add an external keep-alive/health-check
loop with a short enough interval, or restart on a schedule.

## CUDA graph: blocked by PLE file offload

`--disable-cuda-graph` is required in both serve scripts. CUDA graph
capture fails when `--ple-offload-embedding --ple-offload-backend file` is
active:

```text
Capture cuda graph failed: Cannot copy between CPU and CUDA tensors during
CUDA graph capture ...
```

The PLE table's ~95 GiB size makes fully-resident (non-offloaded) serving
impractical on a single Spark's 128 GiB unified memory alongside the ~87 GB
model weights -- so this recipe runs without CUDA graphs. Revisit if either
a pinned-memory PLE offload path is added, or the model is served with PLE
fully resident on a host with more headroom.

## Where the time goes

Not yet profiled on SGLang (see the [vLLM recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe)'s
torch-profiler breakdown for the same architecture on the same hardware --
EXL3 dense GEMV/GEMM and fused MoE dominate decode time there, and the
mechanism (trellis dequant bound, not bandwidth bound) should transfer).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| MTP accept rate stuck at 0.00, tok/s worse than no-draft | `sglang-exl3` missing PR #2/#3, pin overlays not applied, or `embed_tokens` still EXL3 -- update plugin, run `apply_sglang_pin_overlays.sh` |
| `exl3_moe signature mismatch` / decode stuck ~10–16 tok/s with MTP accepting | 29-arg call vs 35-arg ext, or `EXL3_FUSED_MOE=0` -- need plugin `b1d9eab+` and `EXL3_FUSED_MOE=1` |
| `Parameter fc_embedding.trellis not found in params_dict, skip loading` (repeated for `qkv_proj`, `o_proj`, shared-expert, indexer) | same as above -- draft is loading dense weights, not EXL3 |
| `NotImplementedError: Qwen4-Exp QSA MTP currently supports speculative_eagle_topk=1` | you set `--speculative-eagle-topk` above 1 -- not supported, revert to 1 |
| `CUDA error: device-side assert triggered` / `vectorized_gather_kernel: index out of bounds` right after relaunch with a non-default `speculative_num_steps`/`speculative_num_draft_tokens` | you raised those above the verified `steps=3 draft_tokens=4` pair -- revert |
| same assert, but on a server that had been idle for ~an hour | the open idle-crash issue above -- relaunch, and see Mitigation |
| `Capture cuda graph failed: Cannot copy between CPU and CUDA tensors during CUDA graph capture` | expected with `--ple-offload-embedding` active -- keep `--disable-cuda-graph` |
| memory watchdog kills the process, or `MemAvailable` runs low | lower `--mem-fraction-static`, or check `--ple-offload-dir`'s resident-set cap |

## Related repositories

| Repo | Role |
|---|---|
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | the pack this recipe serves |
| [sglang-exl3](https://github.com/vcruz305/sglang-exl3) | the SGLang EXL3 plugin: routed-expert `mul1` codebook fix (PR #1) and native-EXL3-draft MTP fix (PR #2) this recipe depends on |
| [Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) | sibling recipe for the same model/pack on **vLLM** instead of SGLang |
| [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | sibling recipe this one is modeled on (vLLM) |
| [GLM-5.3-Flash-EXL3-K2-SGLang-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-SGLang-DGX-Spark-recipe) | sibling recipe this one is modeled on (SGLang) |

## Credits and upstream work

**ExLlamaV3 by Turboderp ([@turboderp](https://github.com/turboderp-org/exllamav3)).**
The EXL3 trellis format, the MCG/mul1 codebooks, the quantization method,
and the `Qwen3.8-Flash-Next-exl3` pack itself are theirs. MIT, Copyright
(c) 2025 Turboderp.

**[SGLang](https://github.com/sgl-project/sglang)** is the serving engine
this recipe runs on.

## License

MIT for the scripts and notes in this repo (see `LICENSE`). Weights are
**not** redistributed here -- pull them from Hugging Face and respect
turboderp's pack license. SGLang, ExLlamaV3/EXL3, and `sglang-exl3` have
their own licenses.
