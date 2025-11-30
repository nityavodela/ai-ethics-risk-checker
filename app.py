import os
import streamlit as st
from huggingface_hub import InferenceClient
from typing import Optional


# ==========================
# 1. Config & setup
# ==========================

st.set_page_config(
    page_title="AI Ethics & Risk Checker",
    page_icon="",
    layout="wide",
)

# st.title("AI Ethics & Risk Checker")
st.write("Paste any text and get an ethics, privacy, bias, hallucination, and safety review.")

# Get HF token from environment or Streamlit secrets
def get_hf_token() -> Optional[str]:
    """
    1. On Streamlit Cloud: read from st.secrets["HF_TOKEN"]
    2. Locally: read from environment variable HF_TOKEN
    """
    token = None

    # Try Streamlit secrets (Cloud)
    try:
        if "HF_TOKEN" in st.secrets:
            token = st.secrets["HF_TOKEN"]
    except Exception:
        # st.secrets might not be configured locally
        token = None

    # Fallback to environment variable (local)
    if token is None:
        token = os.getenv("HF_TOKEN")

    return token


HF_TOKEN = get_hf_token()

if not HF_TOKEN:
    st.error(
        "Hugging Face token not found.\n\n"
        "Locally: set HF_TOKEN as an environment variable.\n"
        "Streamlit Cloud: add HF_TOKEN to app secrets."
    )
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
# ==========================
# 3. Streamlit UI (FINAL MATERIAL DESIGN v3)
# ==========================

# ---- CLEAN MATERIAL CSS ----
st.markdown("""
<style>

    /* APP BACKGROUND */
    .stApp {
        background-color: #F7F7F8; /* Material light grey surface */
        font-family: "Roboto", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #1F1F1F;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* MATERIAL CARD */
    .md-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.7rem 1.4rem;
        margin-bottom: 1.4rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #1F1F1F;
    }

    /* HEADINGS */
    h2 {
        color: #1F1F1F !important;
        font-size: 1.45rem !important;
        font-weight: 600 !important;
        margin-bottom: .3rem;
    }
    h3 {
        color: #1F1F1F !important;
        font-weight: 500 !important;
        font-size: 1.15rem !important;
        margin-bottom: .5rem;
    }

    /* TEXTAREA (Material Outlined Text Field Style) */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #D2D6DB !important;
        background: #FFFFFF !important;
        color: #1F1F1F !important;
        font-size: .95rem !important;
    }

    textarea::placeholder {
        color: #6B7280 !important;
    }

    /* PRIMARY BUTTON */
    .stButton > button {
        background-color: #1A73E8 !important;
        border-radius: 8px !important;
        border: none !important;
        color: white !important;
        padding: .6rem 1.2rem !important;
        font-size: .95rem !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #1557B0 !important;
    }


    /* METRIC TEXT */
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1F1F1F !important;
    }

    /* PROGRESS BAR */
    [data-testid="stProgressBar"] > div > div {
        background-color: #1A73E8 !important;
    }
    [data-testid="stProgressBar"] > div {
        background-color: #E5E7EB !important;
    }

</style>
""", unsafe_allow_html=True)


# ---- HEADER CARD ----
st.markdown("""
<div class="md-card">
    <h2>AI Ethics & Risk Checker</h2>
    <p style="color:#4B5563; font-size:.93rem; margin-bottom:0;">
        Analyze any text for ethics, privacy, bias, hallucination, and safety risks.
        Receive a safe rewritten version and a summary report.
    </p>
</div>
""", unsafe_allow_html=True)



# ---- INPUT CARD ----
st.markdown("### Input text")

default_text = """Hi team,

We collected everyone's full names, phone numbers, and home addresses to share with our new marketing partner.
They will use this list to run targeted ads and may also pass the data to other companies if needed.

Don't worry about consent, this is just standard practice.

Thanks,
Manager
"""

text = st.text_area(
    "",
    value=default_text,
    height=200,
    placeholder="Paste any email, policy, or text…",
)

st.markdown("<br>", unsafe_allow_html=True)
clicked = st.button("Run ethics & risk check", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)



# ---- RESULTS CARD ----
st.markdown("### Results")

if not clicked:
    st.info("Paste text and run the analysis to see results.")
else:
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_text_structured(text)

        # Risk
        st.markdown("#### Overall risk")
        r_level = result.get("overall_risk") or "Unknown"
        r_score = int(result.get("risk_score") or 0)

        # Decide bar colour based on level
        level_lower = r_level.lower()
        if "low" in level_lower:
            bar_color = "#22C55E"   # green
        elif "medium" in level_lower:
            bar_color = "#FACC15"   # yellow
        elif "high" in level_lower:
            bar_color = "#EF4444"   # red
        else:
            bar_color = "#3B82F6"   # fallback blue

        colA, colB = st.columns([1, 3])

        with colA:
            st.metric("Risk level", r_level)

        with colB:
            st.write(f"Risk score: {r_score}/100")

            # Custom coloured bar
            bar_html = f"""
            <div style="
                background-color:#E5E7EB;
                border-radius:999px;
                height:10px;
                overflow:hidden;
                margin-top:0.25rem;
            ">
                <div style="
                    width:{max(0, min(r_score, 100))}%;
                    background-color:{bar_color};
                    height:100%;
                "></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        st.divider()

        # Issues
                # ---- DETAILED ISSUES (no expanders, just headings) ----
        st.markdown("#### Detailed issues")

        def show_issue_block(title: str, items: list[str]):
            # Section title
            st.markdown(f"**{title}**")

            # Content
            if not items:
                st.write("No issues found.")
            else:
                for issue in items:
                    st.write(f"- {issue}")

            # Light Material-style divider
            st.markdown(
                "<hr style='border:0; border-top:1px solid #E5E7EB; margin:0.6rem 0 0.8rem;'>",
                unsafe_allow_html=True
            )

        show_issue_block("Ethics issues", result["ethics_issues"])
        show_issue_block("Privacy issues", result["privacy_issues"])
        show_issue_block("Bias issues", result["bias_issues"])
        show_issue_block("Hallucination / factual risk", result["hallucination_issues"])
        show_issue_block("Safety issues", result["safety_issues"])

        # st.divider()

        # Safe rewrite
        st.markdown("#### Safe rewritten version")
        st.code(result["safe_rewrite"], language="markdown")

        # Summary report
        st.markdown("#### Summary report")
        st.write(result["short_report"])

st.markdown("</div>", unsafe_allow_html=True)