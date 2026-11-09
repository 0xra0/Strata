"""Opt-in default reasoning allowance, independent of available context length."""


def auto_reasoning_budget(default, request, configured_budget=0, effort=None):
    if isinstance(default, bool) or not isinstance(default, int) or default < 0:
        raise ValueError("reasoning_auto_budget_tokens: expected a non-negative whole number")
    request = request or {}
    if not default or configured_budget > 0 or request.get("reasoning_budget_tokens") is not None:
        return None
    nested = request.get("reasoning")
    level = request.get("reasoning_effort") or (nested.get("effort") if isinstance(nested, dict) else None) or effort
    if level == "none":
        return None
    # The web app sends high even with untouched defaults. Effort alone must not
    # restore a context-sized allowance; an explicit token budget can override it.
    if level in ("xhigh", "max"):
        return default * 2
    return max(1, default // 2) if level == "low" else default
