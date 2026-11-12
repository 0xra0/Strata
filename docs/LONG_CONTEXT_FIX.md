# Long-context RAM working set and thinking allowance

The subsequent resident-page read-ahead revision is documented in `RESIDENT_SWAP_PREFETCH.md`. It addresses
pageable host experts being skipped by routing prefetch after Linux swaps them out, without changing this
revision's RAM allocation or thinking policy.

This revision follows the `live ownership v2` correction. The latest supplied run completed without OOM and
without the earlier wrong-expert restore, but its 62,133-token prompt and 16,332 generated tokens took about
26 minutes. Decode averaged 12.1 tok/s. The engine kept 12.31 of the 15.34 GiB complement in RAM and reported
67.0 GB of storage reads during that request. The PID snapshot also showed roughly 0.95 GiB of the expert
arena swapped out. These observations identify file-tier pressure; they do not measure how much of the
20-to-11 tok/s difference is due to attention, paging, drafting or expert reads.

## Code changes

### Bounded replacement within the existing RAM allocation

The current resident complement is initially chosen by the saved profile. GPU adaptation exchanges weights
between RAM and VRAM, but experts outside both tiers can continue to use the file tier throughout a long
request. The existing optional `STRATA_RAM_ADAPT` policy replaces only victims whose decayed routing count is
below 0.5. That policy is off in the supplied config and can also stall when every RAM expert is still active.

The new optional `STRATA_RAM_WORKING_SET=1` policy considers replacements only after file misses, every fourth
decode window, after pending GPU exchanges land. It selects a repeatedly routed file-backed expert and a
colder CPU-only expert in the same layer. Incoming heat must exceed 1.5 times the victim's heat plus 0.5,
and must be at least 2. Up to four experts and at most 8 MiB are promoted per selection pass. A colder active
expert is eligible; it need not become completely unused first.

The existing `FileExpertSource::replace_resident` operation reads into one reusable staging blob before
overwriting the victim's RAM slot and transferring ownership. A failed read leaves the victim intact.
The complement's allocation, byte capacity and GPU ownership table are unchanged. There is no new SSD-backed
arena. Model quantization, routing, CPU/GPU kernels, KV format, context, prefill and speculation are retained.
Selection metadata is small; the existing replacement buffer can grow to one maximum-size expert.

This policy is limited to the supported single-GPU resident serve path, with the ownership correction active.
Batch, pipelines, layer splits, asynchronous/overlapped adaptation, rotated complements, Foresight and remote
expert execution retain the existing behavior. An explicit `STRATA_RAM_ADAPT` environment setting takes
precedence over the new policy, including an explicit zero.

The per-request log reports promotions, failures, promotion bytes and elapsed policy time. The installer also
enables the existing decode timing report so target runs can separate CPU/expert work, GPU-reach waits,
drafting and adaptation. No target speedup is assumed.

### Thinking no longer inherits a context-sized default allowance

Previously the answer-room rule could allow thinking for 75% of an automatically expanded output limit.
Your run therefore allowed 51,699 thinking tokens. The web app also sends `reasoning_effort: high` with its
untouched defaults.

The new optional `reasoning_auto_budget_tokens` setting supplies a default allowance independent of context
length. This profile sets 4,096 tokens for ordinary/medium/high effort, 2,048 for low and 8,192 for xhigh.
The existing safe-boundary wrap-up/continuation path closes thinking and lets the model answer. The full
output limit is preserved; the input and retained context are not shortened. Smaller output caps still
reserve room for an answer using the existing rule.

An explicit `reasoning_budget_tokens` in the request/shared settings or a positive configured budget wins.
Explicit zero skips the new default and retains the previous answer-room behavior. Setting
`reasoning_auto_budget_tokens` to zero disables this new policy. Thinking disabled by the caller stays disabled.

This is a deliberate change to the default amount of reasoning. Shorter reasoning cannot be guaranteed to
preserve answer quality on every difficult task. Such tasks can request a larger explicit budget; weights,
sampling parameters and answer output capacity are unchanged.

## Install over the corrected memory build

Stop Strata. Extract `Strata-long-context-fix.zip` into `/mnt/ssd`, overwriting files under `Strata/`. Then:

```bash
cd /mnt/ssd/Strata
source .venv/bin/activate
bash tools/rebuild-hip.sh
python3 tools/configure_long_context.py
python serve/server.py --engine strata --config strata-swift-abl-iq2_xs.json --port 8099
```

The script reuses `build-memory-fix-hip` and builds only the engine, preserving the gfx1030/native/MMQ build
choices. It checks both ownership and working-set capability markers before replacing the installed engine,
and keeps the old engine as `engine/strata.prev`. `configure_long_context.py` then backs up the config and
enables the new policies. The vision encoder is not rebuilt.

The config's engine arguments, model paths, VRAM reserve, RAM headroom and image settings are preserved.
Existing explicit reasoning settings are also preserved and can override the new default. Source-archive
configs do not enable this revision before the matching engine is built.

On restart, the log should retain `idle GPU shadows: live ownership v2`. For a thinking request without its
own budget, it should show `default thinking allowance: 4096 tokens` (or the effort-specific value). After
generation, look for `RAM working set v1: active`, promotion counts and `strata decode timing`.

The source changes are in commit 7306db6a; see `git show 7306db6a`.

## Validation and limits

`LONG_CONTEXT_TEST_RESULTS.txt` records the checks. The new production planner is exercised with real immutable
file bytes, fixed host buffers, a changed hot set and exact weight comparisons. The fixture reduces logical
file reads from 4,096 to 512 routed reads plus 32 promotions (86.7%). That is a replay fixture, not a measurement
of the model, target SSD traffic, or RX 6800 tokens/sec. Address/undefined-behavior sanitizers pass.

Server tests cover automatic thinking closure and answering, explicit budget precedence, streaming, Responses
and Anthropic requests, and existing API behavior with the policy off. Installer tests cover failed builds,
old-engine rejection, backup retention and preservation of model/context/vision settings.

No AMD GPU, ROCm compiler or target model is available in this workspace. The modified host code passes C++20
syntax checks with CUDA headers, but a full HIP engine build and target inference have not been run here.
The code removes a source of repeated file reads and bounds default thinking; it does not establish that
long-context decode returns to 20 tok/s, that all paging disappears, or that every answer retains its quality.
Compare the same prompt/context length and explicit output/reasoning limits when measuring engine speed.
