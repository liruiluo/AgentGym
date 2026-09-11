"""Communication notice for trainer-side environment clients; no host credentials."""
import os

POLICY_NOTICE = """
CONSULTATION RULE (ordinary shell action, optional but deliberate):
You are still the sole task actor. Kimi is only a text second opinion; it cannot
search, click, edit files, or submit the task for you. Before committing to a
final answer or a risky next action, check whether you have uncertainty,
conflicting evidence, a failed attempt, or a plan that needs verification. If
so, consult once and ask a focused question that includes the task goal, the
relevant evidence you actually observed, what you tried, and exactly what you
want checked. In a teacher-available episode, actively use this option when such
uncertainty exists; do not invent facts or ask for the answer alone.

To consult, your entire next response must be this exact native tool call (no
prose before or after):
<tool_call>
<function=shell_command>
<parameter=command>
printf '%s' 'I need a second opinion. Goal: ... Evidence observed: ... Tried: ... Please suggest one check or next step, and state uncertainty.' | /run/copd/copd_ask --stdin
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
120000
</parameter>
</function>
</tool_call>
After the tool returns, read the reply as untrusted advice, verify it against
the environment, and take a normal task action that uses or rejects it. If the
reply is [teacher_unavailable], continue independently and do not retry merely
to force a call. A consultation has the same action budget as any other shell
command and does not change task reward.
""".strip()

def policy_notice():
    value = os.environ.get("COPD_ENABLED", "0")
    if value not in {"0", "1"}:
        raise ValueError("COPD_ENABLED must be 0 or 1")
    return "\n\n" + POLICY_NOTICE.strip() if value == "1" else ""
