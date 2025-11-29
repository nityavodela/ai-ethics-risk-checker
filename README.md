A simple Generative AI–powered tool that scans any text (email, policy, AI output, document, message) and flags:

- Ethical risks  
- Privacy issues  
- Bias / unfair or discriminatory language  
- Hallucination / factual overconfidence risks  
- Harmful or unsafe content

For each input, the app provides:

- An overall **risk level** (Low / Medium / High) + numeric score  
- Bullet-point list of issues by category  
- A **safe rewritten version** of the text  
- A short **summary report** explaining the risks

---

**What this project demonstrates**

This project is built as a beginner-friendly, end-to-end GenAI application:

- Uses a free, open-source LLM from Hugging Face (`meta-llama/Llama-3.2-3B-Instruct`)
- Implements a **structured system prompt** for AI ethics and safety analysis
- Parses the model output into a clean Python dictionary
- Exposes the logic via a **Streamlit web app**
- Deployed for free on **Streamlit Community Cloud**

It’s designed as a learning + portfolio project around **AI Safety, Ethics, and Compliance**.

---

**How it works (high level)**

1. **User input**:  
   The user pastes any text into the Streamlit app and clicks **“Run Ethics & Risk Check”**.

2. **LLM prompt & analysis**:  
   The app sends the text to an open-source LLM on Hugging Face with a detailed system prompt that instructs the model to:
   - Identify ethics, privacy, bias, hallucination, and safety issues  
   - Respond in a fixed, structured format

3. **Parsing & scoring**:  
   The raw model output is parsed into:

   ```python
   {
       "overall_risk": "High",
       "risk_score": 90,
       "ethics_issues": [...],
       "privacy_issues": [...],
       "bias_issues": [...],
       "hallucination_issues": [...],
       "safety_issues": [...],
       "safe_rewrite": "...",
       "short_report": "...",
   }

	
4.	**UI display**:
Streamlit displays:
	•	Overall risk + progress bar
	•	Expandable sections per issue type
	•	Safe rewritten version
	•	Short narrative report

No training is done; this project uses inference only with an existing open-source LLM.

⸻

**Tech stack**
	•	Language: Python
	•	LLM provider: Hugging Face Inference API
	•	Model: meta-llama/Llama-3.2-3B-Instruct (or similar small instruct model)
	•	Web UI: Streamlit
	•	Hosting: Streamlit Community Cloud
	•	Notebook experiments: Google Colab

⸻

**Project structure**

ai-ethics-risk-checker/
├─ app.py              # Streamlit app (core logic + UI)
├─ requirements.txt    # Python dependencies
└─ README.md           # This file

(Optional) A Colab notebook can also be added under notebooks/ for experimentation.

⸻
**Run locally**

1. Clone the repository

git clone https://github.com/<your-username>/ai-ethics-risk-checker.git
cd ai-ethics-risk-checker

2. (Optional) Create and activate a virtual environment

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Set your Hugging Face token

Create a free account on Hugging Face￼, generate an access token with read / inference rights, then:

macOS / Linux:

export HF_TOKEN="hf_xxx_your_token_here"

Windows (CMD):

set HF_TOKEN=hf_xxx_your_token_here

5. Run the app

streamlit run app.py

Open the URL shown in the terminal (usually http://localhost:8501).

⸻

**Deployment (Streamlit Community Cloud)**
	1.	Push this repo to GitHub (public).
	2.	Go to Streamlit Community Cloud￼ and create a new app:
	•	Repo: your-username/ai-ethics-risk-checker
	•	Main file: app.py
	3.	In App → Settings → Secrets, add:
  
  HF_TOKEN = "hf_xxx_your_token_here"
  
  4. 	Deploy. You’ll get a public URL to share.




⸻

**Limitations & future improvements**
	•	The tool is not a legal or compliance authority – it’s an AI assistant that can miss issues or be overly strict.
	•	It currently relies on a single general-purpose LLM; domain-specific models or extra rule-based checks could improve reliability.
	•	Hallucination / factual risk is assessed qualitatively from the text itself; no external fact-checking is performed.
	•	Privacy detection is pattern- and instruction-based; more advanced PII detection could be added.

Possible extensions:
	•	Add export of the report as PDF or Markdown
	•	Add a small RAG component with internal policies (e.g., company guidelines)
	•	Support multi-language input
	•	Integrate a second, smaller “safety filter” model as an additional guardrail

⸻

**Learning goals**

This project was built to learn:
	•	How to call open-source LLMs via Hugging Face from Python
	•	How to design structured prompts for AI ethics & safety tasks
	•	How to parse LLM output into structured Python data
	•	How to build and deploy a simple GenAI app using Streamlit

Feel free to fork, experiment with prompts, swap models, or build your own version of an AI safety tool.



