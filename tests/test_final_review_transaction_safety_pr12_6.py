"""Issue #169 gate: CWA transaction-safe submit, reconciliation & recovery.

Deterministic 11-row matrix (LOCKED FINAL CONTRACT §9) over the v3 journal
state machine - no network, no live write:

 1. click dispatched but no POST          -> NO_WRITE_PROVEN before SAFE_TO_RETRY
 2. POST committed, bridge response lost  -> WRITE_FOUND with ZERO resend
 3. crash after side effect, before ACK   -> restart reconciles FIRST
 4. restart sees existing matching turn   -> WRITE_FOUND, never re-delegates
 5. duplicate prevention                  -> ACTIVE_SUBMISSION_EXISTS / replay
 6. ambiguous finality                    -> AMBIGUOUS terminal fail-closed
 7. concurrent user activity              -> no false-positive WRITE_CONFIRMED
 8. stale debugger / leak marker          -> UNKNOWN lane, never NONRETRYABLE
 9. exact fingerprint/binding             -> transformed match, different no
10. safe retry only after negative proof  -> UNKNOWN->RETRY structurally absent
11. happy path                           -> no regression, immutable identity
"""

import json

import pytest

from chatgpt_web_adapter import final_review_transport as frt
from chatgpt_web_adapter.final_review_binding_contract import (
    NO_PREASSIGNED_CONVERSATION,
    bind_final_review_request,
    canonical_request_identity,
    prompt_identity_hash,
)
from chatgpt_web_adapter.final_review_transport import (
    CwaFinalReviewTransport,
    FinalReviewTransportError,
    STATE_AMBIGUOUS,
    STATE_NO_WRITE_PROVEN,
    STATE_PREPARED,
    STATE_RECONCILING,
    STATE_RESPONSE_CONFIRMED,
    STATE_RESPONSE_WAIT,
    STATE_SUBMIT_DELEGATED,
    STATE_WRITE_CONFIRMED,
    STATE_WRITE_FINALITY_UNKNOWN,
    STATE_WRITE_FOUND,
)

REPO = "fixture/repo"
ISSUE = 12
PR = 34
HEAD = "a" * 40
DIGEST = "b" * 64
AUTHORITY = "REQUEST_BOUND_SSE_CONVERSATION_ID_CONSENSUS"
PROMPT = f"Review HEAD {HEAD} packet-marker DIGEST-9f3c end"
CONV = "6aa2a05d-dad4-83ec-b25d-803ba37fe79e"


@pytest.fixture(autouse=True)
def _no_read_waits(monkeypatch):
    monkeypatch.setattr(frt, "_READ_WAIT_S", (0.0, 0.0, 0.0))


def _message(role, text, message_id, *, complete=True):
    return {
        "role": role,
        "text": text,
        "message_id": message_id,
        "finish_reason": "stop" if (role == "assistant" and complete) else None,
        "metadata_preview": (
            {"is_complete": True, "model_slug": "gpt-sol"}
            if (role == "assistant" and complete)
            else {}
        ),
    }


class FakeAck:
    def __init__(self, conversation_id):
        self._conversation_id = conversation_id

    def to_dict(self):
        return {
            "submission_id": "sub-1",
            "transport": "browser-owned",
            "conversation_id": self._conversation_id,
            "turn_lifecycle_id": "tl-1",
            "accepted_at_ms": 1,
        }


class FakeClient:
    def __init__(self, conversations):
        self._conversations = conversations

    def _list_recent_conversations(self, *, limit=10):
        return [{"id": cid} for cid in self._conversations[: max(1, limit)]]


class TransactionRuntime:
    """Scriptable browser-owned runtime.

    mode:
      strong         - write completes with the typed identity authority
      weak_only      - only turn_started transitions (no write identity)
      mismatched_ack - write_completed carries ANOTHER submission's identity
    submit_error:     exception raised after (optional) delegation frame
    conversations:    {conversation_id: [message dicts]} for history probes
    """

    def __init__(
        self,
        *,
        mode="strong",
        submit_error=None,
        emit_delegation=True,
        emit_write_observed=False,
        conversations=None,
        recent=None,
        read_complete=True,
        reply_text=None,
    ):
        self.mode = mode
        self.submit_error = submit_error
        self.emit_delegation = emit_delegation
        self.emit_write_observed = emit_write_observed
        self.conversations = conversations or {}
        self.recent = recent if recent is not None else list(self.conversations)
        self.read_complete = read_complete
        self.reply_text = reply_text
        self.submit_calls = 0
        self.submitted_texts = []
        self.client = FakeClient(self.recent)
        self.read_calls = 0

    def submit(self, text, *, timeout, poll_interval, on_event):
        self.submit_calls += 1
        self.submitted_texts.append(text)
        if self.emit_delegation:
            on_event(
                {
                    "type": "browser_native_delegation_accepted",
                    "submission_id": "sub-1",
                    "acceptedAtMs": 123,
                }
            )
        if self.emit_write_observed:
            on_event(
                {
                    "type": "browser_native_write_observed",
                    "submission_id": "sub-1",
                    "atMs": 456,
                }
            )
        if self.submit_error is not None:
            raise self.submit_error
        if self.mode == "weak_only":
            on_event({"type": "browser_native_turn_started", "submission_id": "sub-1"})
            return FakeAck(CONV)
        if self.mode == "mismatched_ack":
            on_event(
                {
                    "type": "browser_native_write_completed",
                    "submission_id": "SOME-OTHER-SUBMISSION",
                    "conversation_id": CONV,
                    "sse_conversation_identity_authority": AUTHORITY,
                }
            )
            return FakeAck(CONV)
        on_event(
            {
                "type": "browser_native_write_completed",
                "submission_id": "sub-1",
                "conversation_id": CONV,
                "sse_conversation_identity_authority": AUTHORITY,
                "sse_conversation_identity_record_count": 7,
                "sse_conversation_identity_distinct_count": 1,
            }
        )
        return FakeAck(CONV)

    def _items(self, conversation_id):
        messages = list(self.conversations.get(conversation_id, []))
        if conversation_id == CONV and messages == [] and self.submit_calls:
            # default conversation mirrors the last submitted prompt
            messages.append(_message("user", self.submitted_texts[-1], "u-1"))
        return messages

    def get_status(self, conversation):
        self.read_calls += 1
        return {"status": "completed" if self.read_complete else "generating"}

    def get_messages(self, conversation, **kwargs):
        cid = getattr(conversation, "conversation_id", None) or str(conversation)
        items = self._items(cid)
        if (
            cid == CONV
            and any(item.get("role") == "user" for item in items)
            and all(item.get("role") != "assistant" for item in items)
        ):
            items = items + [
                _message(
                    "assistant",
                    self.reply_text or f"verdict for {HEAD}",
                    "a-1",
                    complete=self.read_complete,
                )
            ]
        return [dict(item) for item in items]


def _transport(runtime, tmp_path):
    return CwaFinalReviewTransport(
        expected_repository=REPO,
        expected_issue_number=ISSUE,
        expected_pull_request_number=PR,
        runtime_factory=lambda: runtime,
        durable_store=tmp_path / "store",
    )


def _submit(runtime, tmp_path, **overrides):
    transport = _transport(runtime, tmp_path)
    kwargs = dict(
        prompt=PROMPT,
        repository=REPO,
        issue_number=ISSUE,
        pull_request_number=PR,
        head_sha=HEAD,
        evidence_digest=DIGEST,
        current_head_sha=HEAD,
    )
    kwargs.update(overrides)
    return transport.submit_final_review_request(**kwargs)


def _journal(tmp_path, request_id):
    path = tmp_path / "store" / f"{request_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _code(error):
    return str(error).rsplit(":", 1)[-1]


def _lost_error():
    return Exception(
        "BROWSER_NATIVE_BRIDGE_RESPONSE_LOST_AFTER_DELEGATION: connection reset"
    )


# ---------------------------------------------------------------------------
# Row 1 - click observed but no POST: negative proof BEFORE any retry grant.
# ---------------------------------------------------------------------------
def test_row1_click_without_post_yields_no_write_proven(tmp_path):
    runtime = TransactionRuntime(submit_error=_lost_error(), conversations={})
    with pytest.raises(FinalReviewTransportError) as first:
        _submit(runtime, tmp_path)
    assert _code(first.value) == "WRITE_FINALITY_UNKNOWN"
    assert first.value.reconcile_required is True
    assert first.value.safe_to_retry is False

    with pytest.raises(FinalReviewTransportError) as second:
        _transport(runtime, tmp_path).reconcile_final_review(_journal_id(tmp_path))
    assert _code(second.value) == "NO_WRITE_PROVEN"
    assert second.value.safe_to_retry is True
    assert runtime.submit_calls == 1  # reconcile performs no new write


def _journal_id(tmp_path):
    files = list((tmp_path / "store").glob("*.json"))
    assert len(files) == 1
    return files[0].stem


# ---------------------------------------------------------------------------
# Row 2 - POST happened, bridge response lost: WRITE_FOUND, zero resend.
# ---------------------------------------------------------------------------
def test_row2_response_lost_after_write_is_write_found_zero_resend(tmp_path):
    committed = TransactionRuntime(
        submit_error=_lost_error(),
        conversations={CONV: [_message("user", PROMPT, "u-1")]},
        recent=[CONV],
    )
    with pytest.raises(FinalReviewTransportError) as lost:
        _submit(committed, tmp_path)
    assert _code(lost.value) == "WRITE_FINALITY_UNKNOWN"
    request_id = _journal_id(tmp_path)
    assert _journal(tmp_path, request_id)["state"] == STATE_WRITE_FINALITY_UNKNOWN

    # A FRESH process reconciles: the matching turn exists in history.
    recovery = TransactionRuntime(
        conversations={CONV: [_message("user", PROMPT, "u-1")]},
        recent=[CONV],
    )
    result = _transport(recovery, tmp_path).reconcile_final_review(request_id)
    assert recovery.submit_calls == 0  # NEVER re-delegate a committed write
    assert result.conversation_id == CONV
    journal = _journal(tmp_path, request_id)
    assert journal["state"] == STATE_RESPONSE_CONFIRMED
    # "WRITE_FOUND" appears in the durable reconcile trail (not identity).
    assert journal["canonicalRequestId"] == canonical_request_identity(
        journal["binding"]["canonicalPayload"]
    )
    assert (
        journal["binding"]["canonicalPayload"]["conversationId"]
        == NO_PREASSIGNED_CONVERSATION
    )
    assert journal["conversationId"] == CONV  # correlated EVIDENCE key


# ---------------------------------------------------------------------------
# Row 3 - crash AFTER the side effect, BEFORE the ACK: restart reconciles
# FIRST (journal frozen in SUBMIT_DELEGATED by the WAL).
# ---------------------------------------------------------------------------
def test_row3_crash_after_side_effect_reconcile_first(tmp_path):
    runtime = TransactionRuntime(submit_error=_lost_error(), conversations={})
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    request_id = _journal_id(tmp_path)
    # Simulate process death between WAL persist and the UNKNOWN transition.
    journal_path = tmp_path / "store" / f"{request_id}.json"
    raw = json.loads(journal_path.read_text(encoding="utf-8"))
    raw["state"] = STATE_SUBMIT_DELEGATED
    journal_path.write_text(json.dumps(raw), encoding="utf-8")

    recovered = TransactionRuntime(
        conversations={CONV: [_message("user", PROMPT, "u-1")]},
        recent=[CONV],
    )
    # A restart that calls submit again must reconcile before ANY resend:
    result = _submit(recovered, tmp_path)
    assert recovered.submit_calls == 0
    assert result.conversation_id == CONV
    assert _journal(tmp_path, request_id)["state"] == STATE_RESPONSE_CONFIRMED


# ---------------------------------------------------------------------------
# Row 4 - restart sees existing matching turn: WRITE_FOUND without delegate.
# ---------------------------------------------------------------------------
def test_row4_restart_existing_turn_write_found(tmp_path):
    runtime = TransactionRuntime(
        submit_error=_lost_error(),
        conversations={CONV: [_message("user", PROMPT, "u-1")]},
        recent=[CONV],
    )
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    request_id = _journal_id(tmp_path)
    second = TransactionRuntime(
        conversations={CONV: [_message("user", PROMPT, "u-1")]},
        recent=[CONV],
    )
    result = _transport(second, tmp_path).reconcile_final_review(request_id)
    assert second.submit_calls == 0
    assert result.finality == "FINAL"
    journal = _journal(tmp_path, request_id)
    assert journal["state"] == STATE_RESPONSE_CONFIRMED


# ---------------------------------------------------------------------------
# Row 5 - duplicate prevention: active UNKNOWN lane blocks other requests,
# and same-identity replay performs zero writes.
# ---------------------------------------------------------------------------
def test_row5_duplicate_prevention(tmp_path):
    runtime = TransactionRuntime(submit_error=_lost_error(), conversations={})
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    assert runtime.submit_calls == 1
    with pytest.raises(FinalReviewTransportError) as blocked:
        _submit(
            runtime,
            tmp_path,
            evidence_digest="d" * 64,
        )
    assert _code(blocked.value) == "ACTIVE_SUBMISSION_EXISTS"
    assert runtime.submit_calls == 1
    # Same identity replay: reconcile FIRST - it performs ZERO new writes and
    # surfaces the durable negative proof as safeToRetry (never a silent
    # resend inside the same call).
    with pytest.raises(FinalReviewTransportError) as replay:
        _submit(runtime, tmp_path)
    assert _code(replay.value) == "NO_WRITE_PROVEN"
    assert replay.value.safe_to_retry is True
    assert runtime.submit_calls == 1  # replay NEVER rewrote


# ---------------------------------------------------------------------------
# Row 6 - ambiguous finality: fail-closed forever (submit + reconcile).
# ---------------------------------------------------------------------------
def test_row6_ambiguous_finality_fail_closed(tmp_path):
    # POST was observed by the bridge (durable network evidence) but history
    # cannot locate the turn: contradiction -> AMBIGUOUS, never retry.
    runtime = TransactionRuntime(
        submit_error=_lost_error(),
        emit_write_observed=True,
        conversations={},
        recent=[],
    )
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    request_id = _journal_id(tmp_path)
    assert _journal(tmp_path, request_id)["networkEvidence"]
    with pytest.raises(FinalReviewTransportError) as ambiguous:
        _transport(runtime, tmp_path).reconcile_final_review(request_id)
    assert _code(ambiguous.value) == "AMBIGUOUS_FINALITY"
    assert ambiguous.value.safe_to_retry is False
    assert _journal(tmp_path, request_id)["state"] == STATE_AMBIGUOUS
    assert runtime.submit_calls == 1
    # Later submits on the same identity stay fail-closed.
    with pytest.raises(FinalReviewTransportError) as closed:
        _submit(runtime, tmp_path)
    assert _code(closed.value) == "AMBIGUOUS_NOT_RECONCILABLE"
    assert runtime.submit_calls == 1


# ---------------------------------------------------------------------------
# Row 7 - concurrent user activity: weak/mismatched signals NEVER terminalize
# WRITE_CONFIRMED; someone else's turn is not our write.
# ---------------------------------------------------------------------------
def test_row7_weak_signal_cannot_terminalize_write_confirmed(tmp_path):
    someone_elses = {
        "conv-other": [
            _message("user", "totally unrelated daily question", "u-x"),
            _message("assistant", "answer", "a-x"),
        ]
    }
    runtime = TransactionRuntime(
        mode="weak_only",
        emit_delegation=False,
        conversations=someone_elses,
        recent=list(someone_elses),
    )
    with pytest.raises(FinalReviewTransportError) as raised:
        _submit(runtime, tmp_path)
    assert _code(raised.value) == "WRITE_FINALITY_UNKNOWN"
    request_id = _journal_id(tmp_path)
    journal = _journal(tmp_path, request_id)
    assert journal["state"] == STATE_WRITE_FINALITY_UNKNOWN
    assert journal["writeAckTier"] == "weak"
    # Reconcile: no matching turn anywhere -> provably not ours, safe to retry.
    with pytest.raises(FinalReviewTransportError) as proven:
        _transport(runtime, tmp_path).reconcile_final_review(request_id)
    assert _code(proven.value) == "NO_WRITE_PROVEN"
    assert proven.value.safe_to_retry is True
    assert runtime.submit_calls == 1  # no false-positive confirmation


def test_row7b_write_identity_must_correlate_to_this_request(tmp_path):
    runtime = TransactionRuntime(
        mode="mismatched_ack",
        conversations={},
        recent=[],
    )
    with pytest.raises(FinalReviewTransportError) as raised:
        _submit(runtime, tmp_path)
    assert _code(raised.value) == "WRITE_FINALITY_UNKNOWN"
    assert _journal(tmp_path, _journal_id(tmp_path))["writeAckTier"] == "correlated"


# ---------------------------------------------------------------------------
# Row 8 - stale debugger attachment leak: UNKNOWN lane (reconcile), NOT
# NONRETRYABLE; after negative proof the request is safely retryable.
# ---------------------------------------------------------------------------
def test_row8_debugger_leak_is_unknown_not_nonretryable(tmp_path):
    runtime = TransactionRuntime(
        submit_error=Exception("CHATGPT_DEBUGGER_ATTACHMENT_LEAK"),
        conversations={},
    )
    with pytest.raises(FinalReviewTransportError) as lost:
        _submit(runtime, tmp_path)
    assert _code(lost.value) == "WRITE_FINALITY_UNKNOWN"
    request_id = _journal_id(tmp_path)
    journal = _journal(tmp_path, request_id)
    assert journal["state"] == STATE_WRITE_FINALITY_UNKNOWN
    assert journal["submitFailureLane"] == "POST_DELEGATION"
    with pytest.raises(FinalReviewTransportError) as proven:
        _transport(runtime, tmp_path).reconcile_final_review(request_id)
    assert _code(proven.value) == "NO_WRITE_PROVEN"
    assert proven.value.safe_to_retry is True


# ---------------------------------------------------------------------------
# Row 9 - exact fingerprint/binding: composer-transformed text matches;
# different content never matches (no false WRITE_FOUND).
# ---------------------------------------------------------------------------
def test_row9_transformed_text_matches_and_different_does_not(tmp_path):
    transformed = PROMPT.replace("_", "\\_").replace(" ", "\u00a0")
    runtime = TransactionRuntime(
        submit_error=_lost_error(),
        conversations={CONV: [_message("user", transformed, "u-1")]},
        recent=[CONV],
    )
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    request_id = _journal_id(tmp_path)
    result = _transport(
        TransactionRuntime(
            conversations={CONV: [_message("user", transformed, "u-1")]},
            recent=[CONV],
        ),
        tmp_path,
    ).reconcile_final_review(request_id)
    assert result.conversation_id == CONV

    other = TransactionRuntime(
        submit_error=_lost_error(),
        conversations={CONV: [_message("user", "completely different ask", "u-9")]},
        recent=[CONV],
    )
    with pytest.raises(FinalReviewTransportError):
        _submit(other, tmp_path, evidence_digest="e" * 64)
    # same store now has two journals; ours is the one not already resolved
    second_id = [
        path.stem
        for path in (tmp_path / "store").glob("*.json")
        if path.stem != request_id
    ][0]
    with pytest.raises(FinalReviewTransportError) as proven:
        _transport(other, tmp_path).reconcile_final_review(second_id)
    assert _code(proven.value) == "NO_WRITE_PROVEN"


# ---------------------------------------------------------------------------
# Row 10 - structural retry gate: UNKNOWN->RETRY impossible; the ONLY edge
# into the delegation lane is the NO_WRITE_PROVEN retry, attempt-bounded.
# ---------------------------------------------------------------------------
def test_row10_unknown_retry_structurally_impossible():
    assert STATE_SUBMIT_DELEGATED not in frt._STATE_TRANSITIONS[
        STATE_WRITE_FINALITY_UNKNOWN
    ]
    assert frt._STATE_TRANSITIONS[STATE_WRITE_FINALITY_UNKNOWN] == frozenset(
        {STATE_RECONCILING}
    )
    assert frt._STATE_TRANSITIONS[STATE_NO_WRITE_PROVEN] == frozenset(
        {STATE_SUBMIT_DELEGATED}
    )
    assert frt._STATE_TRANSITIONS[STATE_AMBIGUOUS] == frozenset()
    assert frt._STATE_TRANSITIONS[STATE_RESPONSE_CONFIRMED] == frozenset()


def test_row10b_retry_budget_exhausts_fail_closed(tmp_path):
    # UNKNOWN -> reconcile -> NO_WRITE_PROVEN -> (explicit next submit) one
    # bounded re-delegation -> UNKNOWN again -> NO_WRITE_PROVEN again -> the
    # SECOND retry is structurally refused. Never an unbounded resend loop.
    runtime = TransactionRuntime(submit_error=_lost_error(), conversations={})
    codes = []
    for _ in range(5):
        with pytest.raises(FinalReviewTransportError) as raised:
            _submit(runtime, tmp_path)
        codes.append(_code(raised.value))
    assert codes == [
        "WRITE_FINALITY_UNKNOWN",
        "NO_WRITE_PROVEN",
        "WRITE_FINALITY_UNKNOWN",
        "NO_WRITE_PROVEN",
        "RETRY_BUDGET_EXHAUSTED",
    ]
    assert runtime.submit_calls == 2  # exactly the bounded attempt count


def test_row10c_mutation_of_immutable_journal_fields_fails_closed(tmp_path):
    runtime = TransactionRuntime(submit_error=_lost_error(), conversations={})
    with pytest.raises(FinalReviewTransportError):
        _submit(runtime, tmp_path)
    request_id = _journal_id(tmp_path)
    transport = _transport(runtime, tmp_path)
    journal = transport._load_journal(request_id)
    with pytest.raises(FinalReviewTransportError) as frozen:
        transport._update_journal(journal, payloadText="tampered")
    assert _code(frozen.value) == "IDENTITY_MUTATION_FORBIDDEN"
    with pytest.raises(FinalReviewTransportError) as illegal:
        transport._update_journal(
            journal, to_state=STATE_SUBMIT_DELEGATED
        )  # UNKNOWN -> delegated is the banned edge
    assert _code(illegal.value) == "STATE_MACHINE_VIOLATION"
    # Direct dict tampering is still caught at the storage chokepoint:
    # an identity mutation breaks the requestId self-consistency check.
    journal["binding"]["canonicalPayload"]["headSha"] = "f" * 40
    with pytest.raises(FinalReviewTransportError) as choke:
        transport._persist_journal(journal)
    assert _code(choke.value) == "IDENTITY_MUTATION_FORBIDDEN"
    journal["binding"]["canonicalPayload"]["headSha"] = HEAD
    journal["state"] = "NOT_A_STATE"
    with pytest.raises(FinalReviewTransportError) as bogus:
        transport._persist_journal(journal)
    assert _code(bogus.value) == "STATE_MACHINE_VIOLATION"


# ---------------------------------------------------------------------------
# Row 11 - happy path: unchanged semantics, contract-visible extras only.
# ---------------------------------------------------------------------------
def test_row11_happy_path_no_regression(tmp_path):
    runtime = TransactionRuntime(conversations={}, recent=[])
    result = _submit(runtime, tmp_path)
    assert runtime.submit_calls == 1
    assert result.finality == "FINAL"
    envelope = result.to_dict()
    assert envelope["state"] == STATE_RESPONSE_CONFIRMED
    assert envelope["safeToRetry"] is False
    assert envelope["reconcileRequired"] is False
    journal = _journal(tmp_path, result.canonical_request_id)
    assert journal["state"] == STATE_RESPONSE_CONFIRMED
    assert journal["liveWriteCount"] == 1
    assert journal["delegationAttempts"] == 1
    assert journal["bridgeDelegatedAtMs"] == 123  # host phase ACK persisted
    # Identity v3 7-tuple present and self-consistent.
    payload = journal["binding"]["canonicalPayload"]
    for key in (
        "repository",
        "issueNumber",
        "headSha",
        "reviewAttemptId",
        "conversationId",
        "promptHash",
        "evidenceDigest",
    ):
        assert key in payload
    assert payload["conversationId"] == NO_PREASSIGNED_CONVERSATION
    assert payload["promptHash"] == prompt_identity_hash(PROMPT)
    assert (
        canonical_request_identity(payload) == journal["canonicalRequestId"]
    )


def test_identity_carries_review_attempt_and_expected_conversation(tmp_path):
    runtime = TransactionRuntime(conversations={}, recent=[])
    result = _submit(
        runtime,
        tmp_path,
        review_attempt_id="7",
        expected_conversation_id=CONV,
    )
    payload = _journal(tmp_path, result.canonical_request_id)["binding"][
        "canonicalPayload"
    ]
    assert payload["reviewAttemptId"] == "7"
    assert payload["conversationId"] == CONV


# ---------------------------------------------------------------------------
# Legacy journal migration (pre-#169 durable state -> v3 lane).
# ---------------------------------------------------------------------------
def test_legacy_bound_journal_migrates_to_unknown_not_prepared(tmp_path):
    bound = bind_final_review_request(
        REPO,
        ISSUE,
        PR,
        HEAD,
        DIGEST,
        expected_repository=REPO,
        expected_issue_number=ISSUE,
        expected_pull_request_number=PR,
        current_head_sha=HEAD,
        known_request_ids=set(),
        prompt_sha256=prompt_identity_hash(PROMPT),
    )
    store = tmp_path / "store"
    store.mkdir(parents=True)
    legacy = {
        "schema": 2,
        "probe": "FINAL_REVIEW_TRANSPORT",
        "mode": "REQUEST",
        "state": "BOUND",
        "canonicalRequestId": bound["canonicalRequestId"],
        "binding": bound,
        "payloadText": PROMPT,
        "liveWriteCount": 0,
    }
    (store / f"{bound['canonicalRequestId']}.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    runtime = TransactionRuntime(conversations={}, recent=[])
    transport = _transport(runtime, tmp_path)
    journal = transport._load_journal(bound["canonicalRequestId"])
    assert journal["state"] == STATE_WRITE_FINALITY_UNKNOWN
    assert journal["legacyState"] == "BOUND"
    assert journal["schema"] == 2  # identity/provenance keys untouched
    with pytest.raises(FinalReviewTransportError) as proven:
        transport.reconcile_final_review(bound["canonicalRequestId"])
    assert _code(proven.value) == "NO_WRITE_PROVEN"
    assert proven.value.safe_to_retry is True
