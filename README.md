<h1 align="center">Strata</h1>

<p align="center">
  <b>High-Performance Inference Engine for 125B MoE Models on Consumer Hardware</b><br>
  Run Qwen3.8-Flash-Next on ordinary gaming PCs (NVIDIA or AMD graphics card + system RAM) · Windows & Linux · Free & Open Source
</p>

<p align="center">
  <a href="https://github.com/0xra0/Strata/releases"><img src="https://img.shields.io/github/v/release/0xra0/Strata?style=flat-square&color=blue" alt="Latest Release"></a>
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square" alt="C++20">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-12%2B-brightgreen.svg?style=flat-square" alt="CUDA">
  <img src="https://img.shields.io/badge/ROCm%2FHIP-gfx1030%20%7C%20gfx1100%2B-brightgreen.svg?style=flat-square" alt="ROCm/HIP">
</p>

---

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (and its Coder, Swift 1.5, and Unsloth variants) directly on consumer gaming hardware. By orchestrating a hybrid memory architecture across your GPU VRAM, host system RAM, and storage tiers, Strata enables a 125-billion parameter mixture-of-experts (MoE) model to run entirely locally without requiring enterprise server clusters.

This repository includes **advanced memory lifecycle management, dynamic working-set replacement, kernel-level swap prefetching, cache-avoidant prompt streaming, and reasoning recovery enhancements**.

---

## Table of Contents

- [Key Enhancements in this Release](#key-enhancements-in-this-release)
- [Performance & Benchmarks](#performance--benchmarks)
- [Hardware Requirements](#hardware-requirements)
- [Model Selection Guide](#model-selection-guide)
- [Quick Start & Installation](#quick-start--installation)
  - [Automated Setup](#automated-setup)
  - [Building from Source (Linux / ROCm / CUDA)](#building-from-source-linux--rocm--cuda)
  - [Applying Advanced Rebuild Scripts](#applying-advanced-rebuild-scripts)
- [Configuration & Runtime Flags](#configuration--runtime-flags)
- [API & Agent Integrations](#api--agent-integrations)
  - [OpenAI Compatible API](#openai-compatible-api)
  - [Anthropic Messages API (Claude Code)](#anthropic-messages-api-claude-code)
  - [OpenAI Responses API (Codex CLI)](#openai-responses-api-codex-cli)
  - [Zed Editor Integration](#zed-editor-integration)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
- [Testing & Validation](#testing--validation)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License & Credits](#license--credits)

---

## Key Enhancements in this Release

This version incorporates a suite of memory, I/O, and inference stability improvements engineered specifically for high-load sessions, massive context windows, and multi-modal workloads:

### 1. Idle GPU Shadows Reclaim (`STRATA_MEMORY_LIFECYCLE=1`)
- **Zero-Waste Idle Footprint:** The resident expert complement retains pageable host copies of GPU-cached experts to allow prefill scratch borrowing. While idle, duplicate host pages are safely discarded without re-reading model files.
- **Live Ownership v2:** Validates against the live dynamic `host_res` routing table rather than stale startup profiles, preventing wrong-expert restoration and repetition loops across continuous chats.
- **Fail-Closed Safety:** If memory cannot be safely synchronized or verified, inference fails closed rather than producing corrupted tokens.

### 2. CPU Vision Compute Workspace Release (`vision.release_workspace: true`)
- **Immediate Scheduler Reclamation:** After image encoding, the GGML graph scheduler compute workspace buffers are immediately released (`strata-vision --release-workspace`) and glibc heap pages are trimmed before engine prefill begins.
- **Weight Retention:** Vision model and projector weights remain resident at constant memory addresses; only transient execution buffers are freed.

### 3. Eviction-First GDN Checkpoint Reuse
- **Double-Allocation Elimination:** The LRU cache captures GDN state vectors by reusing the state memory of evicted victim checkpoints directly, avoiding memory spikes during long chat histories.

### 4. Long-Context Dynamic RAM Working Set (`STRATA_RAM_WORKING_SET=1`)
- **Bounded In-Place Replacement:** In long-context requests, frequently routed file-tier experts are dynamically promoted into colder resident RAM slots within the existing complement allocation (bounded at 4 experts / 8 MiB per cycle).
- **Reduced Disk Traffic:** Achieves up to an **86.7% reduction** in storage I/O on repetitive multi-turn contexts without expanding system RAM allocations.

### 5. Decoupled Thinking Allowance (`reasoning_auto_budget_tokens`)
- **Context-Independent Reasoning:** Decouples default thinking allowances from expanding context windows. While standard models could consume 50,000+ thinking tokens on 64k+ contexts, the engine sets an intelligent default (e.g. 4,096 tokens for medium/high effort, 2,048 for low, 8,192 for xhigh).
- **Explicit Override Support:** Requests specifying explicit `reasoning_budget_tokens` take immediate priority.

### 6. Resident Swap Read-Ahead (`STRATA_RESIDENT_SWAP_PREFETCH=1`)
- **Kernel-Level Lookahead:** Inspects page residency via `mincore` and issues asynchronous `MADV_WILLNEED` advice for predicted resident experts that Linux has paged into swap.
- **Safety Floors:** Enforces a 2,560 MiB free system RAM floor and strict rate caps (128 MiB per 100 ms) to avoid thrashing or memory exhaustion.

### 7. Cache-Avoidant Prompt Stream Reader (`STRATA_PROMPT_STREAM=1`)
- **Direct & Uncached I/O:** Reads prompt expert spans via `preadv2` with `RWF_DONTCACHE` or aligned `O_DIRECT`.
- **Page-Cache Protection:** Prevents multi-gigabyte prompt prefill streams from evicting hot resident experts from the Linux buffer cache.

### 8. Reasoning-Only Completion Recovery (`reasoning_close_retry v2`)
- **Guaranteed Answer Generation:** When reasoning models hit natural EOS stops after thinking without outputting an answer or tool call, the server performs a bounded single continuation in the same HTTP stream to ensure an answer is delivered.
- Eliminates "empty reply" issues observed in agentic workflows such as the Zed editor.

---

## Performance & Benchmarks

Measured on consumer hardware running Qwen3.8-Flash-Next (1 token ≈ ¾ of an English word):

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Quantization | Answer Generation | Prompt Ingestion (32K) |
| :--- | :---: | :---: |
| **Q2_0** | **94 tok/s** | 2,650 tok/s |
| **IQ2_XS** | **79 tok/s** | 2,090 tok/s |
| **IQ3_XXS** | **62 tok/s** | 1,750 tok/s |
| **IQ3_S** | **53 tok/s** | 1,620 tok/s |
| **Coder** | **55 tok/s** | 2,180 tok/s |

</td><td>

| Quantization | Answer Generation | Prompt Ingestion (32K) |
| :--- | :---: | :---: |
| **Q2_0** | **60 tok/s** | 1,160 tok/s |
| **IQ2_XS** | **52 tok/s** | 1,110 tok/s |
| **Coder** | **44 tok/s** | 1,420 tok/s |

</td></tr>
</table>

- GPUs with 24 GB VRAM (e.g., RTX 3090 / 4090) achieve **100–140+ tokens/s**.
- Multi-GPU setups (2–3 cards) can pool VRAM across PCIe buses.

---

## Hardware Requirements

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **GPU** | **NVIDIA:** RTX 20/30/40/50 series with 12 GB+ VRAM<br>**AMD:** RX 6800/6900, 7700/7800/7900, 9060/9070 series with 12 GB+ VRAM | 16 GB – 24 GB VRAM (RTX 4080/4090, RX 7900 XTX) |
| **System RAM** | 32 GB RAM (for Coder variant) | 48 GB – 64 GB+ RAM |
| **Storage** | 80 GB free disk space | Fast NVMe PCIe 4.0 SSD |
| **OS** | Linux (Ubuntu 22.04+, Arch, Fedora) or Windows 10/11 | Linux with ROCm 6.x or CUDA 12.x |

---

## Model Selection Guide

Your system RAM determines the ideal model size:

| System RAM | Recommended Model | Notes |
| :--- | :--- | :--- |
| **32 GB** | **Coder (IQ1_M)** | Coding variant with non-critical experts pruned. 91% SWE-bench Verified. |
| **48 GB** | **IQ2_XS** or **Q2_0** | Full model capabilities; Q2_0 provides the fastest generation speeds. |
| **64 GB** | **IQ2_XS** (Recommended) or **IQ3_S** | Every size fits comfortably; IQ3_S provides top-tier reasoning. |
| **96 GB+** | **IQ3_S** or **Unsloth UD-IQ4_XS** | Maximum precision and complete expert complement in RAM. |

---

## Quick Start & Installation

### Automated Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0xra0/Strata.git
   cd Strata
   ```

2. **Run the setup script:**
   - **Linux:**
     ```bash
     chmod +x setup.sh START-HERE.sh
     ./setup.sh
     ```
   - **Windows:** Double-click `START-HERE.bat` or run:
     ```cmd
     setup.bat
     ```

3. **Follow the interactive wizard:**
   - The installer inspects your GPU, VRAM, and RAM, suggests the best quantization, sets up Python virtual environments, builds the engine if needed, and downloads the model shards.
   - Access the web interface at `http://127.0.0.1:8080`.

---

### Building from Source (Linux / ROCm / CUDA)

#### Prerequisites
- CMake 3.22+, GCC/Clang with C++20 support
- Python 3.10+
- **For AMD:** ROCm / HIP SDK (6.0+)
- **For NVIDIA:** CUDA Toolkit 12.x+

#### Building the Engine
```bash
# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure and compile with HIP (AMD gfx1030 example)
cmake -B build-hip -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1030" \
  -DSTRATA_NATIVE_EXPERTS=ON

cmake --build build-hip --target strata -j$(nproc)

# Or compile for NVIDIA CUDA:
cmake -B build-cuda -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON

cmake --build build-cuda --target strata -j$(nproc)
```

#### Building the Vision Module
```bash
cmake -B build-vision -S tools/vision -DCMAKE_BUILD_TYPE=Release
cmake --build build-vision --target strata-vision -j$(nproc)
```

---

### Applying Advanced Rebuild Scripts

To enable memory lifecycle, swap prefetch, and prompt streaming optimizations on an existing build:

```bash
# 1. Build and configure memory lifecycle & CPU vision workspace release
bash tools/rebuild-memory-fix.sh

# 2. Add long-context dynamic working set & thinking auto-budget
bash tools/rebuild-long-context-fix.sh

# 3. Add kernel swap prefetch lookahead
bash tools/rebuild-swap-prefetch-fix.sh

# 4. Add cache-avoidant prompt file stream reader
bash tools/rebuild-prompt-stream-fix.sh

# 5. Enable reasoning-only completion recovery in server configuration
python3 tools/enable_reasoning_answer_fix.py
```

---

## Configuration & Runtime Flags

Model profiles are defined in `strata-<model>.json`. Key configuration parameters include:

```json
{
  "model_name": "qwen3.8-flash-next",
  "port": 8080,
  "args": [
    "--pack", "/path/to/packs",
    "--native", "/path/to/model-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/model-00002-of-00002.gguf",
    "--expert-profile", "/path/to/expert-profile.bin",
    "--resident-experts"
  ],
  "env": {
    "STRATA_MEMORY_LIFECYCLE": "1",
    "STRATA_RAM_WORKING_SET": "1",
    "STRATA_RESIDENT_SWAP_PREFETCH": "1",
    "STRATA_PROMPT_STREAM": "1"
  },
  "vision": {
    "model": "/path/to/model.gguf",
    "mmproj": "/path/to/mmproj.gguf",
    "release_workspace": true
  },
  "reasoning_auto_budget_tokens": {
    "low": 2048,
    "medium": 4096,
    "high": 4096,
    "xhigh": 8192
  },
  "reasoning_close_retry": true,
  "runtime_memory_log": true
}
```

### Environment Variables
| Variable | Description |
| :--- | :--- |
| `STRATA_MEMORY_LIFECYCLE=1` | Discards duplicate host expert pages during idle; verifies via Live Ownership v2. |
| `STRATA_RAM_WORKING_SET=1` | Bounded dynamic promotion of file experts into cooler resident RAM slots during long contexts. |
| `STRATA_RESIDENT_SWAP_PREFETCH=1` | Uses `mincore` and `MADV_WILLNEED` to prefetch swapped resident experts during generation. |
| `STRATA_PROMPT_STREAM=1` | Activates `preadv2(RWF_DONTCACHE)` and `O_DIRECT` for uncached prompt expert reading. |
| `STRATA_RESIDENT_HEADROOM_GIB=N` | Leaves $N$ GiB of RAM unreserved for system and vision tasks (default: 4 GiB). |

---

## API & Agent Integrations

Strata exposes a production-ready HTTP server compatible with OpenAI and Anthropic specifications.

### OpenAI Compatible API

Point your client or library to `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are an expert AI assistant."},
        {"role": "user", "content": "Explain kernel prefetch in operating systems."}
    ],
    temperature=0.6,
    extra_body={"reasoning_effort": "high"}
)

print(response.choices[0].message.content)
```

### Anthropic Messages API (Claude Code)

Strata supports the Anthropic Messages specification natively at `/v1/messages`:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="local"
claude
```

### OpenAI Responses API (Codex CLI)

Supports `/v1/responses` for coding agents and modern OpenAI protocol implementations.

### Zed Editor Integration

Add Strata as an OpenAI-compatible provider in `~/.config/zed/settings.json`:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://127.0.0.1:8080/v1",
      "available_models": [
        {
          "name": "qwen3.8-flash-next",
          "display_name": "Strata Qwen3.8 Flash Next",
          "max_tokens": 131072
        }
      ]
    }
  }
}
```

*Note: The included `reasoning_close_retry v2` patch guarantees completed answers when using reasoning in Zed.*

### Model Context Protocol (MCP)

Strata functions as an MCP client and server. AI coding tools can query status, execute commands, or control inference through documented MCP tool definitions (see [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md)).

---

## Testing & Validation

The codebase includes comprehensive unit and integration test fixtures covering server operations, memory lifecycles, and low-level kernel I/O:

```bash
# Run server test suite
python3 -m unittest discover -s serve -p "test_*.py"

# Run memory lifecycle and reasoning tests
python3 -m unittest serve/test_reasoning_completion.py \
                     serve/test_reasoning_policy.py \
                     serve/test_resident_recovery.py \
                     serve/test_vision_memory.py

# Run tool configuration test suite
python3 -m unittest tools/test_configure_memory_lifecycle.py \
                     tools/test_long_context_fix.py \
                     tools/test_prompt_stream_fix.py \
                     tools/test_reasoning_answer_fix.py \
                     tools/test_rebuild_memory_fix.py \
                     tools/test_rebuild_ownership_fix.py \
                     tools/test_swap_prefetch_fix.py
```

---

## How It Works

Models like Qwen3.8-Flash-Next distribute computation across 24,576 individual experts. In standard server environments, all parameters reside in high-bandwidth GPU memory (HBM). Strata partitions these responsibilities across hardware tiers:

1. **GPU Cache Tier (VRAM):** Keeps the hottest, most repeatedly activated experts in fast graphics memory.
2. **Resident Memory Tier (RAM):** Retains the broader complement of active experts in host memory, processed in parallel by CPU kernels.
3. **Storage Tier (NVMe SSD):** Hosts cold experts and lookup indices, accessed on-demand using cache-avoidant asynchronous I/O.
4. **Speculative Decoding:** A lightweight helper draft model predicts upcoming tokens in batches; the primary MoE model validates them in a single parallel pass, delivering 1.6×–1.8× faster generation.

For architectural deep-dives, consult [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md), [`docs/DETAILS.md`](docs/DETAILS.md), and the research paper in [`docs/paper/Strata-Paper.pdf`](docs/paper/Strata-Paper.pdf).

---

## Troubleshooting

- **Server fails with "out of memory" or is killed by earlyoom:**
  Ensure you are using a quantization that matches your system RAM (e.g., Coder for 32 GB, IQ2_XS for 48–64 GB). Enable `STRATA_MEMORY_LIFECYCLE=1` and set `vision.release_workspace: true`.
- **Repetitive generation after long conversations:**
  Ensure the engine was rebuilt with Live Ownership v2 (`idle GPU shadows: live ownership v2` in logs).
- **Thinking concludes but no answer is displayed:**
  Enable `reasoning_close_retry: true` in your configuration (`python3 tools/enable_reasoning_answer_fix.py`).
- **Additional Diagnostics:**
  Consult [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) or submit an issue with your log attached.

---

## License & Credits

- **Strata Engine:** Released under the [MIT License](LICENSE).
- **Base Model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Alibaba Qwen Team.
- **Quantization & Derivatives:** ISTA-DASLab, UkisAI (Swift 1.5), Unsloth, and OrcaRouter.
- **Compute Kernels:** Utilizes components from [llama.cpp](https://github.com/ggml-org/llama.cpp) and GGML under their respective open-source licenses.
