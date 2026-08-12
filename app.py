import json
import logging
import os
import sys
import time
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, "src")

import streamlit as st  # noqa: E402

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("KEEP_CALM_MODELS_DIR", str(MODELS_DIR))

st.set_page_config(page_title="Keep Calm", page_icon="🧘", layout="centered")


@st.cache_resource
def load_analyzer():
    from huggingface_hub import snapshot_download
    # snapshot_download is incremental: it only fetches files that changed.
    snapshot_download(
        "elchief84/keep-calm-models",
        local_dir=str(MODELS_DIR),
    )
    from keep_calm.analyzer import KeepCalmAnalyzer
    return KeepCalmAnalyzer()


analyzer = load_analyzer()

st.title("🧘 Keep Calm")
st.caption("*Think twice. Send once.*")

st.markdown(
    "Paste a message you're about to send. Keep Calm analyzes tone, intent, "
    "and risk — then **you** decide. Nothing is stored or logged."
)

msg = st.text_area(
    "Your message",
    placeholder="Type or paste your message here ...",
    height=120,
    max_chars=2000,
)

if st.button("Analyze", type="primary", use_container_width=True):
    if not msg.strip():
        st.warning("Please enter a message.")
    elif analyzer is None:
        st.error(
            "Models not found. Make sure models are on Hugging Face Hub "
            "(`elchief84/keep-calm-models`) or in `data/models/`."
        )
    else:
        result = analyzer.analyze(msg)

        risk = result.communication_risk

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{risk:.0%}")
        col2.metric("Risk Level", result.risk_level.value.capitalize())
        col3.metric("Needs Attention", "Yes ⚠️" if result.needs_attention else "No ✅")

        st.progress(risk, text=f"Risk: {risk:.0%}")
        st.info(f"**{result.explanation}**")

        with st.expander("Details"):
            tones_str = ", ".join(
                f"**{t.label.value}** ({t.confidence:.0%})" for t in result.tones
            )
            st.markdown(f"**Tones:** {tones_str}")
            st.markdown(
                f"**Intent:** {result.intent.value} ({result.intent_confidence:.0%})"
            )

        st.divider()
        st.caption("Was this helpful? *(anonymous — nothing is saved except your vote)*")
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        with fb_col1:
            st.button(
                "👍 Helpful", key="fb_yes",
                on_click=lambda: _log(msg, result.risk_level.value, "yes"),
            )
        with fb_col2:
            st.button(
                "🤔 Partially", key="fb_maybe",
                on_click=lambda: _log(msg, result.risk_level.value, "maybe"),
            )
        with fb_col3:
            st.button(
                "👎 Not helpful", key="fb_no",
                on_click=lambda: _log(msg, result.risk_level.value, "no"),
            )

st.divider()
st.markdown(
    "**How it works** — Risk score estimates perception by a typical reader. "
    "Tones detect emotional signals. Intent classifies the goal. "
    "You always have the final word. · "
    "[GitHub](https://github.com/keep-calm/keep-calm) · Apache 2.0"
)


def _log(message: str, risk: str, helpful: str) -> None:
    entry = {
        "timestamp": time.time(),
        "risk_level": risk,
        "helpful": helpful,
        "message_length": len(message),
    }
    try:
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
