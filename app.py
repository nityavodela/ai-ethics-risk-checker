import os
import streamlit as st
from huggingface_hub import InferenceClient

# ==========================
# 1. Config & setup
# ==========================

st.set_page_config(
    page_title="AI Ethics & Risk Checker",
    page_icon="🧭",
    layout="wide",
)

st.title("AI Ethics & Risk Checker")
st.write("Paste any text and get an ethics, privacy, bias, hallucination, and safety review.")

# Get HF token from environment or Streamlit secrets
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HF_TOKEN in Streamlit secrets or as an environment variable.")
    st.stop()


@st.cache_resource
def get_client():
    """Create and cache the Hugging Face Inference client."""
    return InferenceClient(
        "meta-llama/Llama-3.2-3B-Instruct",
        token=HF_TOKEN,
    )


client = get_client()

# ==========================
# 2. Core LLM helper + prompts
# ==========================

ETHICS_SYSTEM_PROMPT = """
You are an AI Ethics, Safety, and Compliance Checker.

Your job is to carefully analyze a piece of text and identify:
1. Ethical risks
2. Privacy issues
3. Bias / unfair or discriminatory language
4. Hallucination or factual overconfidence (especially if it sounds like AI output)
5. Harmful, unsafe, or illegal content (self-harm, violence, crime, etc.)

Be strict but fair. Assume the user wants to improve and make the text safer.

IMPORTANT:
Always respond in this exact structure:

Overall risk: <Low / Medium / High>

Ethics issues:
- <point 1 or "None found">
- <point 2...>

Privacy issues:
- <point 1 or "None found">

Bias issues:
- <point 1 or "None found">

Hallucination or factual risk:
- <point 1 or "None found">

Safety issues:
- <point 1 or "None found">

Safe rewrite:
<Provide a safer, clearer version of the original text.
If the text is already safe, you can say "The original text is already safe, here is a slightly improved version:" and then rewrite it.>

Short report:
<3–5 sentences summarizing the main risks and how they were fixed in the rewrite.>
"""


def run_llm(system_prompt: str, user_message: str, max_tokens: int = 700, temperature: float = 0.3) -> str:
    """Call the chat model with a system + user message and return plain text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = response.choices[0].message["content"]
    return answer


def parse_analysis_output(raw_text: str) -> dict:
    """Same parser we used in Colab, adapted for the app."""
    result = {
        "overall_risk": None,
        "risk_score": None,
        "ethics_issues": [],
        "privacy_issues": [],
        "bias_issues": [],
        "hallucination_issues": [],
        "safety_issues": [],
        "safe_rewrite": "",
        "short_report": "",
        "raw_output": raw_text,
    }

    current_section = None
    safe_rewrite_lines = []
    short_report_lines = []

    lines = raw_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        if lower.startswith("overall risk:"):
            value = stripped.split(":", 1)[1].strip()
            result["overall_risk"] = value

        elif lower.startswith("ethics issues:"):
            current_section = "ethics_issues"

        elif lower.startswith("privacy issues:"):
            current_section = "privacy_issues"

        elif lower.startswith("bias issues:"):
            current_section = "bias_issues"

        elif lower.startswith("hallucination or factual risk:"):
            current_section = "hallucination_issues"

        elif lower.startswith("safety issues:"):
            current_section = "safety_issues"

        elif lower.startswith("safe rewrite:"):
            current_section = "safe_rewrite"

        elif lower.startswith("short report:"):
            current_section = "short_report"

        else:
            if current_section in [
                "ethics_issues",
                "privacy_issues",
                "bias_issues",
                "hallucination_issues",
                "safety_issues",
            ]:
                if stripped.startswith("-"):
                    item = stripped.lstrip("-").strip()
                    if item and item.lower() != "none found":
                        result[current_section].append(item)

            elif current_section == "safe_rewrite":
                safe_rewrite_lines.append(stripped)

            elif current_section == "short_report":
                short_report_lines.append(stripped)

    result["safe_rewrite"] = "\n".join(safe_rewrite_lines).strip()
    result["short_report"] = " ".join(short_report_lines).strip()

    if result["overall_risk"]:
        risk_level = result["overall_risk"].lower()
        mapping = {
            "low": 20,
            "medium": 60,
            "high": 90,
        }
        for level, score in mapping.items():
            if level in risk_level:
                result["risk_score"] = score
                break

    return result


def analyze_text_structured(text: str) -> dict:
    """Full pipeline for the app."""
    user_message = f"""
Here is the text you need to analyze:

\"\"\"{text}\"\"\"

Please analyze this text according to your instructions and respond in the exact structure you were given.
"""
    raw_output = run_llm(ETHICS_SYSTEM_PROMPT, user_message)
    parsed = parse_analysis_output(raw_output)
    return parsed


# ==========================
# 3. Streamlit UI
# ==========================

st.subheader("Input text")

default_text = """Hi team,

We collected everyone's full names, phone numbers, and home addresses to share with our new marketing partner.
They will use this list to run targeted ads and may also pass the data to other companies if needed.

Don't worry about consent, this is just standard practice.
Also, ignore complaints from older employees; they don't understand how modern data works.

Thanks,
Manager
"""

text = st.text_area(
    "Paste the text you want to check:",
    value=default_text,
    height=250,
)

col1, col2 = st.columns([1, 3])

with col1:
    analyze_button = st.button("Run Ethics & Risk Check")

with col2:
    st.info("The model analyzes ethics, privacy, bias, hallucination and safety. No data is stored.")

if analyze_button:
    if not text.strip():
        st.warning("Please paste some text to analyze.")
    else:
        with st.spinner("Analyzing text with the AI Ethics Checker..."):
            result = analyze_text_structured(text)

        # ===== Overall risk =====
        st.subheader("Overall Risk")
        risk_level = result.get("overall_risk") or "Unknown"
        risk_score = result.get("risk_score") or 0

        st.metric("Risk Level", risk_level, None)
        st.progress(min(max(risk_score, 0), 100) / 100)

        # ===== Detailed issues =====
        st.subheader("Detailed Issues")

        def show_issue_section(title, items):
            with st.expander(title, expanded=True):
                if items:
                    for i in items:
                        st.write(f"- {i}")
                else:
                    st.write("None found.")

        show_issue_section("Ethics issues", result["ethics_issues"])
        show_issue_section("Privacy issues", result["privacy_issues"])
        show_issue_section("Bias issues", result["bias_issues"])
        show_issue_section("Hallucination / factual risk", result["hallucination_issues"])
        show_issue_section("Safety issues", result["safety_issues"])

        # ===== Safe rewrite =====
        st.subheader("Safe Rewritten Version")
        if result["safe_rewrite"]:
            st.code(result["safe_rewrite"], language="markdown")
        else:
            st.write("No rewrite returned by the model.")

        # ===== Short report =====
        st.subheader("Short Report")
        if result["short_report"]:
            st.write(result["short_report"])
        else:
            st.write("No report returned by the model.")

        # (Optional) raw output for debugging
        with st.expander("Raw model output (debug)", expanded=False):
            st.text(result["raw_output"])