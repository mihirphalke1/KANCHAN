"""
Tamper-evident hash chain for the case-history ledger (data/case_history.json).

Every saved case stores two fields:

  prev_hash   - the entry_hash of the case appended just before it (genesis is
                64 zeros), so the entries form a singly-linked chain by content.
  entry_hash  - sha256(prev_hash + canonical(case)) where canonical() is the
                case's JSON with volatile/non-evidentiary fields removed and
                keys sorted, so the same case always hashes identically.

Why this shape:
  * Silent tampering is detectable — editing any historical case, or removing
    one from the middle, breaks every entry_hash after it, and verify_chain()
    reports the exact index where the chain first diverges.
  * Legacy entries written before the chain existed simply have no entry_hash;
    verification skips them and begins at the first chained entry, so turning
    this on does not invalidate an existing ledger.
  * Deletion is a soft, chain-preserving tombstone (see routers/history.py):
    the `deleted` marker is EXCLUDED from the hashed payload, so marking a case
    deleted never breaks the chain, and the original evidence stays on disk and
    verifiable rather than being silently rewritten out of existence.
"""
import hashlib
import json
from typing import Optional

GENESIS_HASH = "0" * 64

# Fields never folded into a case's content hash: entry_hash is the output
# itself; `deleted` is an after-the-fact tombstone applied to an already-hashed
# entry. prev_hash IS hashed — that is what chains one entry to the next.
_UNHASHED_FIELDS = ("entry_hash", "deleted")


def _canonical(case: dict) -> str:
    payload = {k: v for k, v in case.items() if k not in _UNHASHED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_entry_hash(case: dict, prev_hash: str) -> str:
    """Content hash for a case chained onto prev_hash. Deterministic: depends
    only on prev_hash and the case's non-volatile content."""
    return hashlib.sha256((prev_hash + _canonical(case)).encode("utf-8")).hexdigest()


def _last_chained_hash(history: list) -> str:
    """entry_hash of the most recent chained entry, or genesis if none."""
    for case in reversed(history):
        h = case.get("entry_hash")
        if h:
            return h
    return GENESIS_HASH


def append_with_chain(history: list, case: dict) -> dict:
    """Stamp `case` with prev_hash + entry_hash so it links onto the tail of
    `history`, and return it. Does not mutate `history`; the caller appends."""
    prev = _last_chained_hash(history)
    case["prev_hash"] = prev
    case["entry_hash"] = compute_entry_hash(case, prev)
    return case


def restamp_from(history: list, index: int) -> None:
    """Recompute prev_hash + entry_hash for every chained entry from `index`
    onward, in place. Used when a legitimate, authenticated amendment edits a
    case's hashed content (e.g. adding customer contact details captured after
    intake) — the amendment is recorded in the case's own `amendments` log, and
    re-stamping keeps the chain internally consistent from that point forward.
    Legacy pre-chain entries (no entry_hash) are left untouched."""
    if index <= 0:
        prev = GENESIS_HASH
    else:
        prev = _last_chained_hash(history[:index])
    for i in range(max(index, 0), len(history)):
        case = history[i]
        if not case.get("entry_hash"):
            continue
        case["prev_hash"] = prev
        case["entry_hash"] = compute_entry_hash(case, prev)
        prev = case["entry_hash"]


def verify_chain(history: list) -> dict:
    """Recompute the chain and report the first divergence, if any.

    Returns a summary dict:
      ok            - True if every chained entry verifies in order
      chained       - number of entries that carry chain fields
      unchained     - number of legacy entries skipped (no entry_hash)
      broken_at     - index of the first bad entry, or None
      broken_case   - case_id of that entry, or None
      reason        - human-readable outcome
    """
    prev = GENESIS_HASH
    chained = 0
    unchained = 0
    started = False

    for idx, case in enumerate(history):
        stored = case.get("entry_hash")
        if not stored:
            # Legacy pre-chain entry — allowed only before the chain begins.
            if started:
                return {
                    "ok": False, "chained": chained, "unchained": unchained,
                    "broken_at": idx, "broken_case": case.get("case_id"),
                    "reason": f"Entry {idx} ({case.get('case_id')}) is missing its hash inside a chained ledger — possible tampering or removal",
                }
            unchained += 1
            continue

        started = True
        if case.get("prev_hash") != prev:
            return {
                "ok": False, "chained": chained, "unchained": unchained,
                "broken_at": idx, "broken_case": case.get("case_id"),
                "reason": f"Entry {idx} ({case.get('case_id')}) prev_hash does not match the previous entry — chain link broken",
            }
        recomputed = compute_entry_hash(case, prev)
        if recomputed != stored:
            return {
                "ok": False, "chained": chained, "unchained": unchained,
                "broken_at": idx, "broken_case": case.get("case_id"),
                "reason": f"Entry {idx} ({case.get('case_id')}) content does not match its hash — the record was altered after it was written",
            }
        prev = stored
        chained += 1

    return {
        "ok": True, "chained": chained, "unchained": unchained,
        "broken_at": None, "broken_case": None,
        "reason": (
            f"Chain intact: {chained} chained ent"
            f"{'ry' if chained == 1 else 'ries'} verified"
            + (f", {unchained} legacy pre-chain entries" if unchained else "")
        ),
    }


def tip_hash(history: list) -> Optional[str]:
    """The current head hash of the ledger (entry_hash of the last chained
    entry), or None if the ledger has no chained entries yet."""
    h = _last_chained_hash(history)
    return None if h == GENESIS_HASH else h
