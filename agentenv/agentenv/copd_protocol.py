"""Communication notice for trainer-side environment clients; no host credentials."""
import os

POLICY_NOTICE = "\nYou may consult a text advisor with an ordinary shell_command. Use the executable\n`/run/copd/copd_ask 'your question and relevant evidence'`; or pipe your question\nthrough stdin with `/run/copd/copd_ask --stdin`. Its stdout is the advisor's reply,\nso normal shell redirection can save it in your workspace. Decide whether and\nwhen help is useful and supply the context you want the advisor to see. The\nadvisor has no tools or hidden task answers and may be wrong; verify its advice\nwith task evidence. Availability is fixed for the episode. If it returns\n[teacher_unavailable], continue solving independently. A consultation uses the\nsame shell action budget as any other command and does not change task reward.\nUse the existing maximum shell timeout for a consultation. The advisor only sees\nwhat you explicitly send; later calls do not automatically include past calls.\n"

def policy_notice():
    value = os.environ.get("COPD_ENABLED", "0")
    if value not in {"0", "1"}:
        raise ValueError("COPD_ENABLED must be 0 or 1")
    return "\n\n" + POLICY_NOTICE.strip() if value == "1" else ""
