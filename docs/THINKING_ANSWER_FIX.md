# Zed finishes with thinking but no answer

This is a Python server update over the prompt-stream revision. The existing
engine binary and its memory/performance fixes are retained. No engine rebuild
is needed.

## Evidence

The attached Zed export has an assistant turn containing only a thinking block.
It ends with a plan to look at the output directory, then the user asks
`you done job ?`. The following assistant turn contains the completed summary.
The native log's corresponding final requests generated 1,713 and 1,695 tokens.
It does not retain the server's finish reason for the thinking-only turn. The
screenshot confirms a normal stop for the later summary, but does not prove the
earlier turn's trigger. Neither the raw SSE nor the model's end delimiter was
attached. The stop variants below are independently reproduced by HTTP tests.

The translation report now confirms both HTTP requests returned 200 and the
correct translation. Its repeat generated eight tokens at 20.5 tok/s.
The latest engine run (PID 3043452, started 2026-10-26 19:54:10) completed 29
requests, generating 30,244 tokens at a weighted 19.40 tok/s. The final request
generated 1,695 tokens at 21.2 tok/s with a 64,797-token prompt. The prompt reader
used RWF_DONTCACHE without fallback/errors. Swap was still nonzero/transient
(peak sample 4.35 GiB); these observations are not a matched speed benchmark or
a guarantee against OOM.

## Cause and correction

Strata previously accepted a model EOS during reasoning as a successful `stop`,
even with no answer or tool call. An optional `reasoning_close_retry` existed,
but was disabled by default and also excluded a closing tag held by the parser.
Consequently merely enabling that option did not cover a stop immediately after
the thinking ended. A local HTTP regression reproduced that exact empty-success
case before this change.

The corrected `reasoning_close_retry v2`:

- Handles reasoning-only natural stops before the end tag, on a held end tag,
  or after closing thinking with only whitespace in the answer.
- Closes thinking only if it is still open. A model-written closing tag is
  retained, and existing separator newlines are reused.
- Makes at most one continuation, inside the same API request, using the
  generated token prefix and the remaining original output allowance.
- Keeps the HTTP stream open until that continuation supplies the answer or
  a tool call. There is one final stream completion, not an intermediate stop.
- Preserves completed/pending tool calls and forced tool selection. It does not
  retry cancelled requests, explicit stop-string matches, token-limit endings,
  valid answers held as a partial stop-string match, or calls already in flight.
- Returns an explicit server error if the model naturally stops empty again.
  Buffered requests use HTTP 502; an already-started stream uses its normal
  error event. This does not unload or restart the healthy engine.

Each continuation/error is recorded in the current engine log without prompt
text. Request history includes `reasoning_answer_retries`. Original input/cache
accounting and the aggregate completion cap cover both native generations.

The feature remains off by default. The supplied activation helper sets only
`reasoning_close_retry: true` in the selected profile. Context, model, cache,
RAM allocation, GPU reserve, thinking budget and sampling are preserved. A valid
ordinary reply performs one generation; a formerly empty completion incurs one
additional bounded generation. Its real-model latency and answer quality still
need checking on the user's hardware.

## Install over the current Strata directory

Stop Strata. Extract Strata-thinking-answer-fix.zip into /mnt/ssd, merging its
Strata directory into /mnt/ssd/Strata. Then:

```bash
cd /mnt/ssd/Strata
python3 tools/enable_reasoning_answer_fix.py
bash run-coder-uncen-iq1_m.sh
```

The helper verifies that the matching server code is present, backs up
strata-coder-uncen-iq1_m.json, and atomically enables the recovery flag. It uses
only Python's standard library. Another profile can be selected with --config.

Startup prints:

```text
[strata] reasoning-only completion recovery enabled (reasoning_close_retry v2)
```

When the model encounters the problematic boundary, the engine log records:

```text
[strata] reasoning-only completion: continuing once to an answer or tool call (reasoning_close_retry v2)
```

To disable this recovery, set reasoning_close_retry to false in the profile and
restart. See THINKING_ANSWER_TEST_RESULTS.txt for the validation and its limits.
