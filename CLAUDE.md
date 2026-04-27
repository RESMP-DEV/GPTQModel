# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GPTQModel is a quantization library for large language models supporting GPTQ, AWQ, ParoQuant, EoRA, QQQ, GGUF, FP8, and EXL3 methods. It provides 94+ model architecture definitions and 35+ inference kernel backends (Marlin, Machete, ExLLaMA v2/v3, BitBLAS, Triton, pure PyTorch, etc.). CUDA kernels are JIT-compiled at runtime, not bundled in wheels.

## Build and Development

```bash
pip install -e .                    # basic install
pip install -e ".[test,quality]"    # with test + lint deps
```

Force kernel rebuild: `GPTQMODEL_KERNEL_REBUILD=1`

## Testing

```bash
pytest tests/                              # all tests
pytest tests/test_foo.py                   # single file
pytest -m "not slow" tests/               # skip slow tests
pytest -m cuda tests/                     # CUDA-only
CUDA_VISIBLE_DEVICES=0 pytest tests/      # specific GPU
```

Markers: `ci`, `cuda`, `cpu`, `gpu`, `inference`, `model`, `mps`, `slow`. Config in `tests/pytest.ini` (addopts: `-s -v`, log_cli on).

## Linting

```bash
bash format/format.sh
```

Runs `ruff==0.14.2` with `--fix --unsafe-fixes` over `gptqmodel/{models,nn_modules,quantization,utils}`, `gptqmodel/__init__.py`, `docs/eora`, `tests`, `setup.py`.

## Architecture

**Entry point:** `GPTQModel` in `gptqmodel/models/auto.py` — static `load()` / `quantize()` / `save()` methods dispatch to the appropriate model class.

**Base class:** `BaseQModel` in `gptqmodel/models/base.py` (~3000 lines) — wraps a HuggingFace `PreTrainedModel`, handles device management, module patching, and quantization orchestration.

**Model definitions:** `gptqmodel/models/definitions/` — each file defines one architecture as a `BaseQModel` subclass. The key abstraction is `module_tree`, a list/dict structure that declares the layer hierarchy and which linear modules to quantize. Notation: `:0` = quantize target, `:1` = skip, `:!` = norm layer. `#` is a repeat marker for numbered layer stacks. `LlamaQModel` is imported first since many architectures inherit from it.

**Quantization pipeline:** `gptqmodel/looper/` — `ModuleLooper` iterates through model layers. Algorithm-specific processors (`gptq_processor.py`, `awq_processor.py`, `paroquant_processor.py`) handle the actual weight quantization. `NamedModule` provides thread-safe module wrapping with scratch state for multi-GPU parallelism via `DeviceThreadPool`.

**Inference kernels:** `gptqmodel/nn_modules/qlinear/` — 35+ `QuantLinear` implementations. Kernel auto-selection is based on device type, compute capability, quantization config, and available extensions. Pure PyTorch fallback in `torch.py`.

**Native extensions:** `gptqmodel_ext/` — C++/CUDA source for Marlin, Machete, ExLLaMA, AWQ, ParoQuant kernels. JIT-compiled via `gptqmodel/extension.py`; compiled artifacts are cached globally.

**Quantization configs:** `gptqmodel/quantization/` — `QuantizeConfig` base plus method-specific subclasses (`GPTQConfig`, `AWQConfig`, `ParoConfig`, `FOEMConfig`, etc.).

## Adding a New Model Architecture

1. Create `gptqmodel/models/definitions/<arch>.py` inheriting from `BaseQModel` (or `LlamaQModel` if the architecture is Llama-like).
2. Define `module_tree` matching the model's `nn.Module` hierarchy.
3. Set `pre_lm_head_norm_module` to the final norm layer path.
4. Register it in `gptqmodel/models/definitions/__init__.py`.
5. Add the mapping in `gptqmodel/models/auto.py`.

## Key Environment Variables

- `GPTQMODEL_KERNEL_REBUILD=1` — force recompile of JIT kernels
- `CUDA_VISIBLE_DEVICES` — GPU selection
- `HF_TOKEN` — HuggingFace Hub authentication
- `TORCH_CUDA_ARCH_LIST` — target CUDA architectures (CI uses `8.0;8.6;8.9;9.0;12.0`)

## File Conventions

All source files carry SPDX license headers (Apache-2.0). Python >=3.10 (supports through 3.14 including free-threading).
