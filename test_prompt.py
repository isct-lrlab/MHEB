# =========================================================
# Answer Extraction Prompt
# =========================================================

EXTRACT_ANSWER = {
    "system_prompt": """
You will be given a multiple-choice question (A–D) and the model's previous-stage output (which may include reasoning and/or a drafted answer).
Your task is only to extract the final selected option letter from the previous-stage output. Do NOT solve the problem again.

Rules:
- If the previous output explicitly states a final choice (e.g., "Final Answer: Yes", "Answer: No", "Final Answer(after applying the memory restriction):Yes"), output that letter.
- If both "Final Answer(before applying the memory restriction)" and "Final Answer(after applying the memory restriction)" are present, extract the option letter from "Final Answer(after applying the memory restriction)" and ignore "Answer(before applying the memory restriction)".
- If no final choice is stated, pick the single letter that the previous output most strongly indicates.

Respond with **only 'Yes' or 'No'** — no additional text.
""".strip(),
    "user_prompt": """
### Question:
{QUESTION}

### Previous stage output:
{REASONING}
""".strip()
}

# =========================================================
# Baselines
# =========================================================

P1 = {
    "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.  
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.  
You must assume **you have never learned them**.

Think silently. Then respond with **only 'Yes' or 'No'**, based on what you knew **before** {cutoff_year}.

Do **not** include any explanation, context, or additional text — just 'Yes' or 'No'.
""".strip(),

    "user_prompt": """
### Question:
{QUESTION}
""".strip()
}


PS = {
    "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.  
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.  
You must assume **you have never learned them**.

Task:
1. Let's first understand the quesiton and devise a plan(2–5 steps) to solve the question.
2. Let's carry out the plan and solve the question step by step.
3. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Plan: <output the plan part>
Solve: <output the solve part>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

    "user_prompt": """
### Question:
{QUESTION}
""".strip()
}

PS_PIPELINE = {"steps": [PS, EXTRACT_ANSWER]}



ZS_COT = {
    "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.  
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.  
You must assume **you have never learned them**.

Task:
1. Let's think step by step to solve the question.
2. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Reasoning: <output the step-by-step reasoning>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

    "user_prompt": """
### Question:
{QUESTION}
""".strip()
}


ZS_COT_PIPELINE = {"steps": [ZS_COT, EXTRACT_ANSWER]}


# =========================================================
# Proposed Method
# =========================================================

SR = {
  "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.
You must assume **you have never learned them**.

Task:
1. Output ONE short self-positioning statement that makes your stance explicit.
2. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Stance: <one short sentence explicitly stating the knowledge cutoff and restriction>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

  "user_prompt": """
### Question:
{QUESTION}
""".strip()
}

SR_PIPELINE = {
    "steps": [SR, EXTRACT_ANSWER]
}

QR = {
  "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.
You must assume **you have never learned them**.

Task:
1. Output ONE short overview of any events, facts, or conditions relevant to this question, including the year associated with each (If no notable events occurred before {cutoff_year}, simply state that).
2. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Overview: <one short sentence highlighting what’s relevant to this question>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

  "user_prompt": """
### Question:
{QUESTION}
""".strip()
}

QR_PIPELINE = {
    "steps": [QR, EXTRACT_ANSWER]
}

SR_QR = {
  "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.
You must assume **you have never learned them**.

Task:
1. Output ONE short self-positioning statement that makes your stance explicit.
2. Output ONE short overview of any events, facts, or conditions relevant to this question, including the year associated with each (If no notable events occurred before {cutoff_year}, simply state that).
3. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Stance: <one short sentence explicitly stating the knowledge cutoff and restriction>
Overview: <one short sentence highlighting what’s relevant to this question>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

  "user_prompt": """
### Question:
{QUESTION}
""".strip()
}

SR_QR_PIPELINE = {
    "steps": [SR_QR, EXTRACT_ANSWER]
}


QR_SR = {
  "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {cutoff_year}**.
Specifically, your memory ends on **December 31, {cutoff_year_minus_1}**, and you have no access to anything that occurred in {cutoff_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.
You must assume **you have never learned them**.

Task:
1. Output ONE short overview of any events, facts, or conditions relevant to this question, including the year associated with each (If no notable events occurred before {cutoff_year}, simply state that).
2. Output ONE short self-positioning statement that makes your stance explicit.
3. Answer in the form of the most reasonable 'Yes' or 'No' response.

Output format:
Overview: <one short sentence highlighting what’s relevant to this question>
Stance: <one short sentence explicitly stating the knowledge cutoff and restriction>
Answer: X
where X is exactly one of 'Yes' or 'No'.

Do not include any other text.
""".strip(),

  "user_prompt": """
### Question:
{QUESTION}
""".strip()
}

QR_SR_PIPELINE = {
    "steps": [QR_SR, EXTRACT_ANSWER]
}