# ComfyUI-MiniMax-H3-PDD-Acc

Native ComfyUI support for the **official MiniMax-H3 8-step PDD acceleration LoRAs**
([alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)) —
full audio+video generation in **8 (or 4) sampler steps**, no CFG.

These files are *not* ordinary LoRAs: alongside a rank-64 trunk LoRA they carry a
**Parallel Decoding Distillation head bank** — 32 per-interval copies of the final-layer
video/audio projections that get fused into one mean-block-velocity head per sampler step
([PDD, Shaul et al. 2026](https://arxiv.org/abs/2607.26004)). A plain LoRA loader can't read
them, and dropping the head bank silently loses the distill. This pack loads the whole thing.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
```

Put the PDD file(s) in `ComfyUI/models/pdd_acc/` (the folder is created on first launch).
Either release works — the loader auto-detects the format:

| Source | Files |
|---|---|
| Original (alibaba-pai) | [MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs): `MiniMax-H3-FL2VA-Acc-8Step.safetensors`, `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` |
| Pre-converted ComfyUI keys | [aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI): `minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors`, `minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors` |

Pair **FL2VA** with an fl2va UNET and **Ref2VA** with a ref2va UNET (bf16 originals or int8
convrot builds both work — LoRA application goes through ComfyUI's quant-aware patch path).

## BSAI 优化版说明

本版本在原版基础上做了以下优化，提升独立性和用户体验：

### 1. 选项显示名优化（避免误判为报错）
- `on_off_grid`：选项从 `["error", "clamp"]` 改为 `["raise", "clamp"]`，默认值 `raise`
- `partition_check`：选项从 `["error", "warn", "off"]` 改为 `["strict", "warn", "off"]`，默认值 `strict`
- **完全向后兼容**：旧工作流中保存的 `error` 值会自动转换为 `raise`/`strict`，无需手动修改

### 2. 版本兼容层（_compat.py）
- 新增 `_compat.py` 模块，集中管理所有 ComfyUI 核心版本依赖
- 所有对 `comfy.nested_tensor.NestedTensor`、`comfy.patcher_extension.WrappersMP` 的直接引用已替换为兼容层函数
- 版本不兼容时给出清晰的错误提示，而不是深层 AttributeError
- 最低要求：ComfyUI >= 0.33.0（MiniMax-H3 carried-audio rework）

### 3. 依赖声明
- 新增 `requirements.txt`，明确 Python 依赖（torch、safetensors、numpy）
- 所有依赖均为 ComfyUI 便携版自带，非便携环境可参考安装

### 4. 无外部自定义插件依赖
- 本插件**不依赖任何其他 ComfyUI 自定义插件**，仅依赖 ComfyUI 核心 + PyTorch + safetensors
- `pdd_acc_core.py` 为纯 torch 实现，可独立于 ComfyUI 使用


**ComfyUI version:** v0.33.0 or newer (the MiniMax-H3 carried-audio mechanics,
comfyanonymous/ComfyUI#15243 — the node fails closed with an update message on older cores).
Both pre- and post-#15375 cores work; the final-layer patch delegates to your core's own
forward rather than replicating its internals.

## Nodes

### MiniMax H3 PDD Acc LoRA (Apply) — `MiniMaxH3PDDAccApply`
`MODEL → MODEL + SIGMAS + info`. One node does everything: applies the trunk LoRA
(converting diffusers keys to ComfyUI naming in memory when given an original-format file)
and installs the PDD head bank on `final_layer`, armed per step **by sigma** — so looping /
chunked samplers, resumes and split schedules can't desync it.

- **nfe** — model evaluations. `8` = trained block size (default). `4` regroups two blocks
  per step (officially sanctioned — the release demos both). `6` uses the non-uniform default
  partition `8,8,4,4,4,4` (the two merged size-8 blocks sit at high sigma where the block
  boundaries span almost no sigma, and the late heavyweight blocks stay at trained size —
  every knot stays on the trained fine grid).
- **partition** (optional) — custom block sizes in fine steps, comma-separated, summing to 32
  (e.g. `8,4,4,4,4,4,4` for 7 steps). Overrides `nfe`; the sigmas output follows.
- **Only block sizes 4 and 8 are legal** — the training envelope. PDD heads are conditioned on
  trunk features from block *starts* on the L_min=4 grid with blocks of 4 or 8 fine steps;
  evaluating the trunk anywhere else feeds the heads features they never trained on and renders
  as heavy noise (community-reported at 32 steps on FL2VA, reproduced locally on plain SDPA —
  it is not an attention-backend issue). The node therefore rejects off-envelope step counts
  and partitions instead of letting them degrade. More steps than 8 is not "closer to the
  teacher" here: the per-interval heads are only ever decoded from envelope block starts.
- **lora_strength / head_strength** — trained at 1.0 / 1.0.
- **on_off_grid** — `error` (default): refuse evaluation at sigmas that are not trained block
  boundaries, with a message telling you what to fix. `clamp`: nearest block, degraded output.
- **enabled** (optional, default true) — `false` = full bypass: the input model and
  `bypass_sigmas` pass through untouched (nothing is loaded or patched). Wire a boolean node
  here to A/B the distill or drive a subgraph toggle.
- **bypass_sigmas** (optional SIGMAS) — returned as the sigmas output when `enabled=false`
  (wire the schedule for the un-distilled model, e.g. a BasicScheduler). The node errors if
  you disable it without wiring this — the PDD block boundaries would be a wrong schedule for
  an unpatched model. Remember the rest of the un-distilled recipe (CFG, sampler, steps)
  differs too.

### MiniMax H3 PDD Acc Scheduler — `MiniMaxH3PDDAccScheduler`
Standalone SIGMAS emitter for partial-denoise / split-sigma workflows. At `denoise 1.0` it
equals the Apply node's sigmas output.

## Required recipe

| Setting | Value | Why |
|---|---|---|
| Sampler | **euler** (KSamplerSelect) | each step consumes one mean block velocity; multi-stage samplers (er_sde, dpmpp, res_*) evaluate off-grid |
| Sigmas | the Apply node's **sigmas output** → SamplerCustomAdvanced | trained boundaries `12t/(1+11t)`, `t = linspace(1,0,nfe+1)` |
| Guidance | **CFG 1.0** (BasicGuider) | guidance is distilled in; single forward per step |
| SigmaShift | **12.0 / 3.0** exactly | the training grid; the node fails closed otherwise |

**Remove** other distill LoRAs (lightx2v turbo etc.) — distills don't stack. Character LoRAs
stack normally. **Do not stack** step-caching packs (blockcache / EasyCache — the final-layer
patch fails closed, and an 8-step distill has nothing to cache anyway).

## Trunk pairing guard (partition fingerprint)

The FL2VA and Ref2VA trunks ship **identical tensor key sets**, so pairing an FL2VA distill
with a ref2va UNET (or vice versa) applies cleanly and renders **silently wrong**. The Apply
node now identifies the loaded model's trunk from its `final_layer.video_out.weight` — that
tensor is fp32-unquantized in every published build, bit-identical across the
int8_convrot/pruned/rebased variants of one trunk, and the two trunks sit 0.0503 apart in
relative Frobenius distance (fingerprints shipped fp16 in `partition_fingerprints/`,
tolerance 0.015 ≫ cast/storage noise ~2e-3). A confident mismatch **errors**; set the
optional `partition_check` input to `warn` for deliberate cross-trunk experiments, or to
`off` to disable the model-type analysis entirely — the fingerprint check never runs and
mispairings apply unchecked (the info output notes it). A
finetune or full-merge that matches neither fingerprint just logs "inconclusive" and
proceeds — the guard never blocks checkpoints it has no fingerprint for. New trunks:
`python3 bake_partition_fingerprint.py <checkpoint> <name>`.

Guard design after [fblissjr/ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
which shipped a partition fingerprint first.

## Pruned checkpoints

Pruned H3 UNETs (Comfy-Org `*_pruned_*`, the GGUF/w4a8/nvfp4 re-quants of them) replace the
dense adaln with a shared 8-dim curve table — a dense adaln LoRA can't patch them, which is
why plain loaders spam ~50 `ERROR lora ... adaln_proj` lines and silently drop that part of
the distill. This pack handles it: on a pruned model the 50 adaln LoRA modules are
**rebased onto the model's curve basis** (weight diff `B(AV)` + the mandatory DC bias diff
`B(Ac)`, from the affine fit `silu(t_emb(t)) ≈ c + V·table(t)` solved in float64 against a
matching full checkpoint — fit residual ~1.4e-5, effectively exact). The node matches the
model's adaln table against the two shipped bases in `adaln_basis/` (one per trunk)
automatically and warns on trunk mismatches.

A repacked or requantized pruned build may carry a table that is **not byte-identical** to
the Comfy-Org ones but still describes the same trunk's curve. The node handles that too:
when no exact match is found it **auto-refits** each shipped basis onto the model's table
(rows of every table sample the same fixed timestep grid, so this is a float64 least-squares
fit) and accepts the best fit when the residual is same-trunk small (~1e-5; a genuinely
different finetune lands around 1e-1 and is refused). If your pruned checkpoint is refused,
it is not a repack of a known trunk — bake a basis with `bake_adaln_basis.py` (see its
docstring) or open an issue naming the exact checkpoint file/source so a basis can be
shipped.

**Hybrid trunks (fl2va+ref2va block merges):** a hybrid carries ONE adaln table — its BASE
trunk's — so e.g. an fl2va-based `b15-49` hybrid matches the fl2va basis and pairing it with
the Ref2VA PDD file logs a trunk-mismatch warning. If the hybrid pairing is what you intend,
the warning is informational: PDD fully applies (the node fails closed if any patch key
misses, and sampling would error — not silently skip PDD — if the heads were not armed;
check the `info` output for the applied module count). But hybrids are **off-label for
PDD**: the trunk LoRA and head bank were trained on the pure trunks, so quality on a merge
is untested. If output looks weak or wrong, A/B against the matching plain trunk before
blaming settings.

## Example workflows

- [`example_workflows/pdd_acc_t2v_basic.json`](example_workflows/pdd_acc_t2v_basic.json) —
  prompt-to-video+audio in 8 steps (Ref2VA trunk, zero references; wire images into
  `ref_image_0…` for identity-locked r2v). Drag into ComfyUI.
- [`example_workflows/pdd_acc_t2v_warmup_split.json`](example_workflows/pdd_acc_t2v_warmup_split.json) —
  two-phase warmup for better reference likeness: the Warmup Scheduler's sigmas are split at
  its `phase2_start_step` with core `SplitSigmas`; pass 1 samples the un-distilled BASE model
  over the warmup segment, pass 2 chains its `output` latent (via `DisableNoise`) into the
  PDD-patched model for the trained tail.
- [`example_workflows/pdd_acc_t2v_latent_upscale.json`](example_workflows/pdd_acc_t2v_latent_upscale.json) —
  two-pass latent upscale (hi-res fix): full 8-step PDD render at 896x512, then
  `MiniMax H3 AV Latent Upscale By` x1.5 (lands exactly on the model-native 1344x768) into a
  partial-denoise PDD pass — the `PDD Acc Scheduler` with `denoise 0.25` re-runs only the
  LAST 2 trained blocks (resume sigma 0.8), so the refine stays on the trained grid
  (0.125 = 1 block/subtle, 0.375 = 3 blocks/strong). Audio is decoded from pass 1 and is
  untouched by the refine. The upscale node exists because core `LatentUpscale` cannot
  handle H3's nested video+audio latent; it resizes the video half per frame (audio passes
  through) and snaps to the model's 2x2 patch grid.

- [`example_workflows/pdd_video_upscale_long.json`](example_workflows/pdd_video_upscale_long.json) —
  **upscale an EXISTING video of any length** (video-to-video hi-res fix): load by path,
  VAE-encode, neural latent upscale to 1344x768, then a PDD partial-denoise refine
  (`denoise 0.25` = last 2 trained blocks) sampled in 73-frame windows with 22-frame
  overlap and anchor frames — so a 70s clip refines in ~25 cheap windows instead of one
  quadratic-cost 500k-token pass. The source audio is muxed straight through untouched.
  The neural upscaler and the windowed refine sampler (`MinimaxH3LatentUpscaler3D`,
  `MMH3SplitUpscale`, `MMH3TemporalSplitParams`) are by **LBH-123-AI** — install their
  [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  node pack and download the
  [upscaler model](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) into
  `models/latent_upscale_models/`. Also needs
  [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for the video load.
  Using the mamad-converted PDD file instead: set nfe 4 on BOTH the Apply node and the Scheduler.

## Converter (optional)

`convert_pdd_acc.py` produces the pre-converted redistribution format from an original file —
standalone, no ComfyUI required:

```bash
python3 convert_pdd_acc.py MiniMax-H3-FL2VA-Acc-8Step.safetensors \
    minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors
```

Output is bit-identical to what the loader computes in memory (tested), with the trunk LoRA
in standard `diffusion_model.*.lora_A/B.weight` + `.alpha` keys and full provenance metadata.

## Baking the trunk (optional — for cards where the model doesn't fully fit)

ComfyUI merges LoRA patches into weights **once at load** — but only for modules that fit
in VRAM. Offloaded modules get a per-forward patch instead: the LoRA (plus a dequantize) is
re-applied on **every step**. On a card at the VRAM edge that fixed per-step cost is large
at low resolutions (issue #4 measured ~2× s/it at 864×480 on a 32GB RTX 5090) and vanishes
into attention time at high ones. **If your model fully loads, baking buys you nothing** —
the runtime path already costs zero per step there.

Measured 2026-08-30 (H200, ref2va int8_convrot, 832×480×124f, 8 steps, SDPA; offload forced
with `--reserve-vram 120` so the whole 32.4GB trunk streams — the log's `lowvram patches`
count is the mechanism made visible):

| regime | runtime patches | baked trunk |
|---|---|---|
| fully offloaded | 2.44 s/it (`lowvram patches: 258`) | **2.06 s/it** (`lowvram patches: 0`) |
| fully loaded | 1.82 s/it | 1.75 s/it (same within noise) |

The bake removes the per-forward patch term entirely; what remains in the offloaded row is
pure weight streaming, which both arms pay equally. The patch term is hardware-dependent —
~0.4 s/step on an H200, ~2.5 s/step in the issue-#4 5090 report — so the win grows as the
card gets smaller, which is exactly who this is for.

`bake_pdd_trunk.py` merges the trunk LoRA (and the adaln update — curve-rebased first on
pruned bases) into the quantized checkpoint offline, using the same `comfy-kitchen` kernels
ComfyUI dequantizes with. Only the head bank stays runtime — it swaps `final_layer` per
fine-interval and cannot be baked. The write is streaming (peak RAM is one module, not one
checkpoint) and every tensor keeps its exact dtype, shape and byte length:

```bash
# audit first (no write): requant error per sampled module
python3 bake_pdd_trunk.py --check \
    --base minimax_h3_ref2va_int8_convrot.safetensors \
    --pdd  MiniMax-H3-Ref2VA-Acc-8Step.safetensors

python3 bake_pdd_trunk.py \
    --base minimax_h3_ref2va_int8_convrot.safetensors \
    --pdd  MiniMax-H3-Ref2VA-Acc-8Step.safetensors \
    --out  minimax_h3_ref2va_pddbaked_int8_convrot.safetensors
```

Load the baked file with a normal UNETLoader and set the Apply node's **`lora_strength`
to `0.0`** (baked-trunk mode: trunk patching skipped, heads/sigmas/guards unchanged; the
info output says so). Caveats: the strength is frozen into the file (re-bake to change
it); on an **unbaked** trunk, strength 0.0 renders the un-distilled model with PDD heads —
nothing can detect that, so check your file's `pdd_acc_baked` metadata if unsure. GGUF and
non-convrot formats are refused, not guessed. Requantizing the merged weight costs the
same class of error the runtime merge pays (it also requantizes); the `--check` audit
prints the measured number for your files.

## How it works (short version)

- **LoRA conversion** (verified against both codebases' sources): `to_q/to_k/to_v` fuse into
  ComfyUI's `attn.qkv_proj` (concatenated `lora_A`, block-diagonal `lora_B`, alpha ×3);
  `ff.net.0.proj → mlp.fc1` with the SwiGLU `[value;gate] → [gate;value]` half-swap;
  `to_out.0 → attn.out_proj`, `ff.net.2 → mlp.fc2`, `adaln_proj.linear` 1:1 (layouts are
  bit-identical); `token_refiner.refiner_blocks → token_refiner.blocks`.
- **Head bank**: per-block plans (fine step sizes normalized per modality on the shift-12
  video / shift-3 audio grids) fuse the 32 heads into `nfe` fused fp32 heads at load —
  identical math to the reference `minimax_h3_pdd.py` einsum. A `DIFFUSION_MODEL` wrapper
  stashes the current sigma; an object patch on `final_layer.forward` selects the block and
  runs the fused projections (everything else in the layer is untouched).
- **Audio needs no extra conversion** on current ComfyUI core: the model's carried-audio
  mapping integrates a *mean* block velocity exactly over finite Euler steps
  (`s·Δσ_v/(c_i·c_j) = Δσ_a` is an algebraic identity since `c` is linear in σ — unit-tested).

## Tests

```bash
python3 tests/test_pdd_acc.py          # torch-only, no ComfyUI needed
PDD_ACC_SLOW=1 python3 tests/test_pdd_acc.py   # + full-tensor checks on the real files
```

13 tests: grid/plan/fusion vs the verbatim reference implementation shipped in the official
repo, boundary sigmas vs diffusers `set_timesteps`, qkv block-diag + SwiGLU swap numerics,
the carried-audio exactness identity, dual-format round-trip, and structural checks against
the real safetensors headers.

## Credits & license

- Acceleration LoRAs: [alibaba-pai](https://huggingface.co/alibaba-pai) (Apache-2.0);
  `tests/reference_minimax_h3_pdd.py` is their reference loader, kept verbatim as test oracle.
- Method: [Parallel Decoding Distillation](https://arxiv.org/abs/2607.26004), Shaul et al.
- Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

This pack: Apache-2.0.


---

## 中文完整说明 / Chinese Full Guide

### 插件介绍 / What This Is

为 **MiniMax-H3 官方 8 步 PDD 加速 LoRA**（[alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)）提供 ComfyUI 原生支持——**8（或 4）步采样即可生成完整 音视频，无需 CFG**。

这些文件不是普通 LoRA：除了 rank-64 主干 LoRA，还带一个**并行解码蒸馏（PDD）头组**——32 份每间隔一份的最终层视频/音频投影副本，在每个采样步融合为一个平均块速度头（[PDD, Shaul et al. 2026](https://arxiv.org/abs/2607.26004)）。普通 LoRA 加载器读不了它们，丢掉头组会静默丢失蒸馏效果；本插件完整加载。

### 安装 / Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-MiniMax-H3-PDD-Acc
```

把 PDD 文件放入 `ComfyUI/models/pdd_acc/`（首次启动自动创建）。两种版本均可，加载器自动识别格式：

| 来源 | 文件 |
|---|---|
| 原版 (alibaba-pai) | [MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)：`MiniMax-H3-FL2VA-Acc-8Step.safetensors`、`MiniMax-H3-Ref2VA-Acc-8Step.safetensors` |
| ComfyUI 预转换 | [aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI)：`minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors`、`minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors` |

**FL2VA 配 fl2va UNET，Ref2VA 配 ref2va UNET**（bf16 原版或 int8 convrot 均可——LoRA 应用走 ComfyUI 量化感知补丁路径）。

依赖：仅 ComfyUI 核心 + PyTorch + safetensors（`pip install torch safetensors numpy` 或便携版自带），**不依赖任何其他自定义插件**。最低 ComfyUI >= 0.33.0（MiniMax-H3 carried-audio rework）。

### 节点 / Nodes

#### MiniMax H3 PDD Acc LoRA (Apply) — `MiniMaxH3PDDAccApply`
`MODEL → MODEL + SIGMAS + info`。一个节点完成全部：应用主干 LoRA（原版文件自动在内存中把 diffusers key 转成 ComfyUI 命名）+ 在 `final_layer` 上安装 PDD 头组，**按 sigma 分步武装**——循环/分块采样、resume、拆分 schedule 都不会失同步。

- **nfe**：模型评估次数。`8` = 训练块大小（默认）。`4` = 每步合并两块（官方认可）。`6` = 非均匀默认分区 `8,8,4,4,4,4`（高 sigma 处两块 8 步合并，后期重块保持训练尺寸）。
- **partition**（可选）：自定义块大小（细步），逗号分隔、合计 32（如 `8,4,4,4,4,4,4` 为 7 步）。覆盖 nfe，sigmas 输出随之变化。
- **只允许块大小 4 和 8**（训练包络）：头组按 L_min=4 网格上的块起点条件化，包络外的步数会渲染成重度噪点，节点会拒绝而非劣化。
- **lora_strength / head_strength**：训练于 1.0 / 1.0。
- **on_off_grid**：`raise`（默认）拒绝非训练块边界的 sigma 并提示；`clamp` 就近取块（画质降级）。
- **enabled**（可选，默认 true）：`false` = 完全旁路（不加载不补丁），可接线做 A/B 对比。

#### MiniMax H3 PDD Acc Scheduler — `MiniMaxH3PDDAccScheduler`
独立的 SIGMAS 发射器，用于 partial-denoise / split-sigma 工作流。`denoise 1.0` 时等于 Apply 节点的 sigmas 输出。

### 必需配方 / Required Recipe

| 设置 | 值 | 原因 |
|---|---|---|
| Sampler | **euler**（KSamplerSelect） | 每步消耗一个平均块速度；多阶段采样器（er_sde/dpmpp/res_*）会离格评估 |
| Sigmas | Apply 节点 **sigmas 输出** → SamplerCustomAdvanced | 训练边界 `12t/(1+11t)`，`t = linspace(1,0,nfe+1)` |
| Guidance | **CFG 1.0**（BasicGuider） | 引导已蒸馏进模型；每步单次前向 |
| SigmaShift | **12.0 / 3.0** 精确 | 训练网格；否则节点 fail-closed |

**移除**其它蒸馏 LoRA（lightx2v turbo 等）——蒸馏不叠加。角色 LoRA 正常叠加。**不要叠加**步数缓存包（blockcache / EasyCache——final_layer 补丁 fail-closed，8 步蒸馏也没有可缓存的东西）。

### 剪枝模型 / Pruned Checkpoints

剪枝 H3 UNET（Comfy-Org `*_pruned_*` 及其 GGUF/w4a8/nvfp4 重量化）把密集 adaln 换成共享 8 维曲线表——普通加载器会刷 ~50 条 `adaln_proj` 报错并静默丢掉这部分蒸馏。本插件在剪枝模型上把 50 个 adaln LoRA 模块**重基到模型曲线基**（`adaln_basis/` 内置两主干基，自动匹配并警告主干不匹配）；无法精确匹配时**自动重拟合**（float64 最小二乘），残差同主干级 ~1e-5 才接受，异主干 ~1e-1 拒绝。被拒即非已知主干的重打包——用 `bake_adaln_basis.py` 烘焙或开 issue。

**混合主干（fl2va+ref2va 块合并）**：只带 BASE 主干的单条 adaln 表；配对会警告主干不匹配（PDD 仍完整应用），但混合对 PDD 属 off-label，画质未验证，建议先用纯主干 A/B。

### 示例工作流 / Example Workflows

- `example_workflows/pdd_acc_t2v_basic.json` — 8 步文生视频+音频（Ref2VA 主干、零参考图；把图接到 `ref_image_0…` 即锁身份 r2v）。拖入 ComfyUI 即用。
- `example_workflows/pdd_acc_t2v_warmup_split.json` — 两阶段 warmup 提升参考相似度：Warmup Scheduler 的 sigmas 在 `phase2_start_step` 用 `SplitSigmas` 拆分；第一遍用未蒸馏 BASE 模型采样 warmup 段，第二遍把输出 latent（经 `DisableNoise`）链入 PDD 补丁模型跑训练尾段。
- `example_workflows/pdd_acc_t2v_latent_upscale.json` — 两遍潜空间放大（hi-res fix）：896×512 全 8 步 PDD 渲染 → `MiniMax H3 AV Latent Upscale By` ×1.5（正好落在模型原生 1344×768）→ `PDD Acc Scheduler` 以 `denoise 0.25` 只重跑最后 2 个训练块（resume sigma 0.8）。音频来自第一遍解码、精修不动。
- `example_workflows/pdd_video_upscale_long.json` — **任意长度已有视频放大**（video-to-video hi-res fix）：按路径加载 → VAE 编码 → 神经潜放大到 1344×768 → PDD partial-denoise 精修（`denoise 0.25`），73 帧窗口 + 22 帧重叠 + anchor 帧采样，70s 素材约 25 个廉价窗口完成；源音频直通。需要 [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)（LBH-123-AI）与 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)。用 mamad 转换版 PDD 时：Apply 与 Scheduler 都设 nfe 4。

### 转换器（可选）/ Converter

```bash
python3 convert_pdd_acc.py MiniMax-H3-FL2VA-Acc-8Step.safetensors \
    minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors
```
输出与加载器内存计算逐位一致，含完整溯源元数据。

### 烘焙主干（可选，显存吃紧时）/ Baking the Trunk

ComfyUI 只在加载时把 LoRA 合并进权重——**装不下的模块每步都重新应用补丁**（低分辨率下固定成本大，实测 32GB RTX 5090 在 864×480 约 2× s/it）。**模型能完全加载时烘焙无收益**。`bake_pdd_trunk.py` 把主干 LoRA（剪枝基上先曲线重基）+ adaln 更新离线合并进量化权重，头组仍运行时（不可烘焙）。烘焙后用普通 UNETLoader 加载，Apply 节点 `lora_strength` 设 **0.0**（烘焙主干模式）。GGUF 与非 convrot 格式拒绝（不猜）。先跑 `--check` 审计。

### 原理（简版）/ How It Works

- **LoRA 转换**：`to_q/to_k/to_v` 融合进 `attn.qkv_proj`（拼接 `lora_A`、块对角 `lora_B`、alpha ×3）；`ff.net.0.proj → mlp.fc1` 带 SwiGLU `[value;gate] → [gate;value]` 半交换；`to_out.0 → attn.out_proj`、`ff.net.2 → mlp.fc2`、`adaln_proj.linear` 1:1；`token_refiner.refiner_blocks → token_refiner.blocks`。
- **头组**：按块计划（细步大小按 shift-12 视频 / shift-3 音频网格逐模态归一）在加载时把 32 头融合成 `nfe` 个 fp32 头——与官方 `minimax_h3_pdd.py` einsum 数学一致。
- **音频**：当前 ComfyUI 核心无需额外转换——carried-audio 映射在有限 Euler 步上精确积分平均块速度（代数恒等，单测覆盖）。

### 测试 / Tests

```bash
python3 tests/test_pdd_acc.py          # 仅 torch，无需 ComfyUI
PDD_ACC_SLOW=1 python3 tests/test_pdd_acc.py   # + 真实文件全张量检查
```
13 项测试：网格/计划/融合 vs 官方仓库逐字参考实现、边界 sigmas vs diffusers `set_timesteps`、qkv 块对角 + SwiGLU 交换数值、carried-audio 精确恒等、双格式往返、真实 safetensors 头结构检查。

### 优化版说明（BSAI 分支）

- `on_off_grid` 选项 `["error","clamp"]` → `["raise","clamp"]`（默认 `raise`）；`partition_check` → `["strict","warn","off"]`（默认 `strict`）；旧工作流保存的 `error` 值自动转换，**完全向后兼容**。
- 新增 `_compat.py` 版本兼容层（ComfyUI 核心版本依赖集中管理），版本不兼容给出清晰提示而非深层 AttributeError。
- 新增 `requirements.txt` 依赖声明；`pdd_acc_core.py` 纯 torch 实现可独立于 ComfyUI 使用。
- 主干配对守卫（partition fingerprint）：FL2VA / Ref2VA 张量 key 完全相同，配对错误会"静默画错"；Apply 节点从 `final_layer.video_out.weight` 识别主干（两主干相对 Frobenius 距离 0.0503，容差 0.015），明确不匹配即报错；`partition_check=warn` 可放行跨主干实验。

### 许可证 / License

- 加速 LoRA：[alibaba-pai](https://huggingface.co/alibaba-pai)（Apache-2.0）；`tests/reference_minimax_h3_pdd.py` 为其参考加载器逐字保留作测试 oracle。
- 方法：[Parallel Decoding Distillation](https://arxiv.org/abs/2607.26004)，Shaul et al.
- 基座：[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)。本插件 Apache-2.0。
