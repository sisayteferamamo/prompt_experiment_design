import pandas as pd
import json
import hashlib
import random
import os
import time
import re
from google import genai
from google.genai import types

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

API_KEY = "MY_API_KEY"

MODEL = "gemini-3-flash-preview"

VALIDATION_FILE = "validation_courses_73.csv"

SLEEP_TIME = 1.6  # avoid rate limits

# ------------------------------------------------
# INITIALIZE CLIENT
# ------------------------------------------------


client = genai.Client(api_key=API_KEY)

# ------------------------------------------------
# PROMPT VERSIONS
# ------------------------------------------------


PROMPTS = {

"prompt_v1_basic":
"""
ROLE

You are an academic curriculum evaluation specialist analyzing university course syllabus descriptions.
Your task is to systematically assess how strongly specific competencies are developed within each course.

Evaluate only what is explicitly described in the syllabus text. Do not assume competencies that are not supported by clear evidence.

------------------------------------------------

SCORING SCALE

0 = Not present  
The competency is not mentioned and there is no evidence in the course description.

1 = Mentioned  
The competency is referenced briefly but not clearly developed through activities.

2 = Developed  
The competency is practiced through assignments, projects, labs, coursework, or applied activities.

3 = Core learning objective  
The competency is a central focus of the course and appears as a main learning objective or repeated emphasis.

------------------------------------------------

COMPETENCIES

V1 – Technical expertise  
Depth of disciplinary knowledge, engineering concepts, models, algorithms, or technical systems.

V2 – Analytical problem solving  
Problem analysis, modelling, evaluation methods, and quantitative reasoning.

V3 – Specialized methods  
Use of specialized tools, software, laboratories, simulations, programming, or technical methodologies.

H1 – Teamwork  
Collaborative work, group projects, or team-based learning activities.

H2 – Communication  
Presentations, reports, documentation, discussions, or written/oral communication.

H3 – Interdisciplinary integration  
Integration of knowledge across disciplines, systems thinking, or cross-domain perspectives.

H4 – Project-based learning  
Projects, applied coursework, design tasks, or practical implementation.

H5 – Sector or industry engagement  
Industry collaboration, real-world case studies, professional practice, or partnerships with external organizations.

------------------------------------------------

EVALUATION PROCEDURE

Follow this reasoning process internally:

1. Carefully read the course description.
2. Identify any explicit evidence related to the competencies.
3. Determine the strongest level of development present for each competency.
4. Assign a score from 0–3 using the defined scale.

Base the score strictly on evidence found in the text.

------------------------------------------------

OUTPUT REQUIREMENTS

Return ONLY a valid JSON list containing a single object with the competency scores.

Do NOT include explanations, comments, or additional text.

------------------------------------------------

Example output

[
{{"V1":2,"V2":2,"V3":1,"H1":1,"H2":1,"H3":0,"H4":2,"H5":1}}
]

------------------------------------------------

COURSE DESCRIPTION

{text}
""",

"prompt_v2_structured":

"""
ROLE

You are an academic curriculum evaluation specialist assessing university course syllabi to identify competency development.

Evaluate only competencies that are explicitly supported by evidence in the course description. Do not assume competencies that are not described.

------------------------------------------------

SCORING SCALE

0 = Not present  
The competency is not mentioned.

1 = Mentioned  
The competency appears briefly but is not clearly practiced.

2 = Developed  
The competency is practiced through coursework, assignments, labs, projects, or applied activities.

3 = Core learning objective  
The competency is central to the course and strongly emphasized.

------------------------------------------------

COMPETENCIES

V1 – Technical expertise  
Disciplinary knowledge, engineering concepts, technical systems.

V2 – Analytical problem solving  
Problem analysis, modelling, evaluation methods, quantitative reasoning.

V3 – Specialized methods  
Use of tools, software, laboratories, simulations, programming, or technical methodologies.

H1 – Teamwork  
Collaborative work, group projects, team-based learning.

H2 – Communication  
Presentations, reports, documentation, discussions, written/oral communication.

H3 – Interdisciplinary integration  
Integration of knowledge across disciplines or systems thinking.

H4 – Project-based learning  
Design projects, applied coursework, practical implementation.

H5 – Sector / industry engagement  
Industry collaboration, professional practice, real-world applications.

------------------------------------------------

EVALUATION PROCEDURE

For the course below:

1. Read the description carefully.
2. Identify explicit evidence for each competency.
3. Determine the strongest level of development present.
4. Assign scores between 0 and 3.
5. If information is ambiguous, default to the lower score to maintain conservative academic rigor.

------------------------------------------------

SCORING EXAMPLE

Example course:

Smart Mobility Systems Project

Description summary:
Students work in interdisciplinary teams to design smart mobility solutions for urban transport systems.  
The course integrates transport engineering, data analytics, and urban planning.  
Teams develop project proposals and present results through reports and presentations.

Evidence-based scoring:

V1 = 2  
Evidence: transport engineering, system design

V2 = 2  
Evidence: design and evaluation of mobility systems

V3 = 2  
Evidence: data analytics methods

H1 = 3  
Evidence: interdisciplinary team collaboration

H2 = 2  
Evidence: presentations and written reports

H3 = 3  
Evidence: integration of engineering, planning, and analytics

H4 = 3  
Evidence: project development and solution design

H5 = 1  
Evidence: real-world mobility context but no direct industry partnership

Expected output format:

------------------------------------------------

OUTPUT FORMAT

Return ONLY a valid JSON object with competency scores.

Example:

{{"V1":2,"V2":1,"V3":2,"H1":0,"H2":1,"H3":0,"H4":2,"H5":0}}

Do not include explanations or additional text.

------------------------------------------------

COURSE DESCRIPTION

{text}
""",


"prompt_v3_reasoned":
"""
ROLE

You are an academic curriculum evaluation specialist assessing university course syllabi to identify competency development.

Your task is to evaluate how strongly each competency is represented in the course description.

Evaluate only competencies supported by explicit textual evidence.

Do NOT assume competencies that are not described.

------------------------------------------------

SCORING SCALE

0 = Not present  
No evidence of the competency.

1 = Mentioned  
The competency appears briefly but is not clearly practiced.

2 = Developed  
The competency is practiced through assignments, labs, coursework, projects, or applied activities.

3 = Core learning objective  
The competency is central to the course and strongly emphasized.

------------------------------------------------

COMPETENCIES

V1 – Technical expertise  
Engineering concepts, technical systems, disciplinary knowledge.

V2 – Analytical problem solving  
Problem analysis, modelling, evaluation methods, quantitative reasoning.

V3 – Specialized methods  
Use of tools, software, laboratories, simulations, programming, or technical methods.

H1 – Teamwork  
Group work, collaborative learning, team projects.

H2 – Communication  
Presentations, reports, written or oral communication.

H3 – Interdisciplinary integration  
Integration of knowledge across disciplines or systems thinking.

H4 – Project-based learning  
Design projects, applied coursework, or practical implementation.

H5 – Sector / industry engagement  
Industry collaboration, professional practice, or real-world application.

------------------------------------------------

EVALUATION PROCEDURE

For the course description below:

1. Identify textual evidence related to each competency.
2. Determine the strongest level of competency development.
3. Assign a score between 0 and 3 based only on the evidence.
4. Avoid over-inferring; only score what is explicitly supported by the text.

------------------------------------------------

SCORING EXAMPLE

Example course:

Smart Mobility Systems Project

Description summary:
Students work in interdisciplinary teams to design smart mobility solutions for urban transport systems.  
The course integrates transport engineering, data analytics, and urban planning.  
Teams develop project proposals and present their solutions through presentations and written reports.

Evidence-based evaluation:

V1 = 2 (transport engineering, system design)  
V2 = 2 (designing mobility solutions)  
V3 = 2 (data analytics methods)  
H1 = 3 (interdisciplinary team collaboration)  
H2 = 2 (presentations and written reports)  
H3 = 3 (integration of engineering, planning, analytics)  
H4 = 3 (project development and solution design)  
H5 = 1 (real-world problem context but no industry collaboration)

Expected output:

{{"V1":2,"V2":2,"V3":2,"H1":3,"H2":2,"H3":3,"H4":3,"H5":1}}

------------------------------------------------

OUTPUT FORMAT

Return ONLY a JSON object with the competency scores.

Example:

{{"V1":2,"V2":1,"V3":2,"H1":0,"H2":1,"H3":0,"H4":2,"H5":0}}

Do not include explanations or additional text.

------------------------------------------------

COURSE DESCRIPTION

{text}
""",

"prompt_v4_reasoned":
"""
ROLE

You are an academic researcher performing a systematic content analysis of course syllabi. Your objective is to objectively measure the alignment of the curriculum with "T-shaped learning principles"—specifically the balance between deep disciplinary expertise (Vertical) and cross-functional competencies (Horizontal).

Your task is to evaluate how strongly each competency is represented in the course description.

Use strict evidence-based judgment.  

Only assign scores supported by explicit textual evidence.

Do NOT assume competencies that are not described in the text.

------------------------------------------------

SCORING SCALE

0 = Not present  
No evidence of the competency.

1 = Mentioned  
The competency appears briefly but is not clearly practiced.

2 = Developed  
The competency is practiced through coursework, assignments, labs, projects, or applied activities.

3 = Core learning objective  
The competency is central to the course and strongly emphasized.

------------------------------------------------

COMPETENCIES

V1 – Technical expertise  
Engineering concepts, disciplinary knowledge, technical systems.

V2 – Analytical problem solving  
Problem analysis, modeling, evaluation methods, quantitative reasoning.

V3 – Specialized methods  
Tools, software, laboratories, simulations, programming, or technical methods.

H1 – Teamwork  
Group work, collaborative learning, team projects.

H2 – Communication  
Presentations, reports, documentation, written or oral communication.

H3 – Interdisciplinary integration  
Integration of knowledge across disciplines or systems thinking.

H4 – Project-based learning  
Design projects, applied coursework, practical implementation.

H5 – Sector / industry engagement  
Industry collaboration, professional practice, or real-world applications.

------------------------------------------------

EVALUATION PROCEDURE

Perform the following reasoning steps internally:

STEP 1 – Identify Evidence  
Read the course description carefully and identify any phrases that indicate the presence of the competencies.

STEP 2 – Determine Competency Strength  
Based on the rubric, determine whether each competency is:
not present, mentioned, developed, or a core objective.

STEP 3 – Consistency Check  
Verify that each score matches the rubric definitions and is supported by evidence in the text.

STEP 4 – Maintain academic rigor
Do not infer. If a project is mentioned but "teamwork" is not explicitly stated, H1 must be 0 or 1.

Step 5 – Assign Scores  
Assign a score from 0 to 3 for each competency.

------------------------------------------------

SCORING EXAMPLE

Example course: Smart Mobility Systems Project

Description summary:
Students work in interdisciplinary teams to design smart mobility solutions for urban transport systems.  
The course integrates transport engineering, data analytics, and urban planning.  
Teams develop project proposals and present results through presentations and written reports.

Evidence-based evaluation:

V1 = 2 (transport engineering, system design)  
V2 = 2 (design and evaluation of mobility systems)  
V3 = 2 (data analytics methods)  
H1 = 3 (interdisciplinary team collaboration)  
H2 = 2 (presentations and written reports)  
H3 = 3 (integration of engineering, planning, analytics)  
H4 = 3 (project development and solution design)  
H5 = 1 (real-world mobility context but no direct industry collaboration)

Expected output:

{{"V1":2,"V2":2,"V3":2,"H1":3,"H2":2,"H3":3,"H4":3,"H5":1}}

------------------------------------------------

OUTPUT FORMAT

Return ONLY a valid JSON object with competency scores.

Example:

{{"V1":2,"V2":1,"V3":2,"H1":0,"H2":1,"H3":0,"H4":2,"H5":0}}

Do not include explanations or additional text.

------------------------------------------------

COURSE DESCRIPTION

{text}
""",

"prompt_v5_reasoned":

"""
ROLE

You are an academic curriculum evaluation specialist analyzing university course syllabi to identify competency development.

Your task is to evaluate how strongly each competency is represented in the course description.

Use strict evidence-based judgment.
Only assign scores supported by explicit textual evidence from the course description.

------------------------------------------------

SCORING SCALE

0 = Not present
No evidence of the competency.

1 = Mentioned
The competency appears briefly but is not clearly practiced.

2 = Developed
The competency is practiced through coursework, assignments, labs, projects, simulations, modeling or applied activities.

3 = Core learning objective
The competency is central to the course and strongly emphasized.

------------------------------------------------

COMPETENCIES

V1 – Technical expertise
Engineering concepts, disciplinary knowledge, technical systems.

V2 – Analytical problem solving
Problem analysis, modeling, evaluation methods, quantitative reasoning.

V3 – Specialized methods
Tools, software, laboratories, simulations, programming, or technical methods.

H1 – Teamwork
Group work, collaborative learning, team projects.

H2 – Communication
Presentations, reports, documentation, written or oral communication.

H3 – Interdisciplinary integration
Integration of knowledge across disciplines or systems thinking.

H4 – Project-based learning
Design projects, applied coursework, practical implementation.

H5 – Sector / industry engagement
Industry collaboration, professional practice, real-world applications.


------------------------------------------------

EVALUATION PROCEDURE

Rules:

1. Read the description carefully.
2. Identify explicit evidence for each competency and use short evidence phrases taken from the text
3. Do not invent information
4. Determine the strongest level of development present.
5. Assign scores between 0 and 3.
6. If information is ambiguous, default to the lower score to maintain conservative academic rigor.

Follow these reasoning steps internally before producing the result.

STEP 1 – Identify Evidence
Read the course description and identify phrases indicating competencies.

STEP 2 – Determine Strength
Use the scoring rubric to determine the correct level.

STEP 3 – Consistency Check
Ensure the score is justified by explicit evidence in the text.



------------------------------------------------

INPUT FORMAT

Each course will be provided as:

course:
keywords:
abstract:
objectives:



------------------------------------------------

OUTPUT FORMAT (STRICT)

Return ONLY a valid JSON object with competency scores.

Example:

{{"V1":2,"V2":1,"V3":2,"H1":0,"H2":1,"H3":0,"H4":2,"H5":0}}

Do not include explanations or additional text.


------------------------------------------------

SCORING EXAMPLE

Example Input

course: Smart Mobility Systems Project
keywords: system design, data analytics, planning
abstract: students work in interdisciplinary teams to design mobility solutions
objectives: teams develop project proposals and present results

Example Output

Evaluation:
V1 = 2 (transport engineering, system design)
V2 = 2 (design and evaluation of mobility systems)
V3 = 2 (data analytics methods)
H1 = 3 (interdisciplinary team collaboration)
H2 = 2 (presentations and written reports)
H3 = 3 (integration of engineering, planning, analytics)
H4 = 3 (project development and solution design)
H5 = 1 (real-world mobility context)

Expected output:

{{"V1":2,"V2":2,"V3":2,"H1":3,"H2":2,"H3":3,"H4":3,"H5":1}}

------------------------------------------------

COURSE DESCRIPTION

{text}
"""
}

# ------------------------------------------------
# SAFE LLM SCORING FUNCTION
# ------------------------------------------------
 
def gemini_score(course_text, prompt_name):

    prompt = PROMPTS[prompt_name].replace("{text}", course_text)

    MAX_RETRIES = 5
    RETRY_DELAY = 5

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(

                model=MODEL,

                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ],

                config=types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH"
                    )
                )
            )

            raw_text = response.text

            match = re.search(r"\{[\s\S]*?\}", raw_text)

            if not match:
                raise ValueError("No JSON detected")

            parsed = json.loads(match.group())

            competencies = ["V1","V2","V3","H1","H2","H3","H4","H5"]

            parsed = {k:int(parsed.get(k,0)) for k in competencies}

            return parsed

        except Exception as e:

            print(f"Attempt {attempt+1} failed:", e)

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached.")

                return {
                    "V1":0,"V2":0,"V3":0,
                    "H1":0,"H2":0,"H3":0,"H4":0,"H5":0
                }

# ------------------------------------------------
# LOAD VALIDATION DATA
# ------------------------------------------------

df = pd.read_csv(VALIDATION_FILE)

print("Validation dataset loaded:", len(df), "courses")

# ------------------------------------------------
# BUILD MERGED TEXT FIELD
# ------------------------------------------------

df["text"] = (
    "Keywords: " + df["keywords"].fillna("") +
    "\nAbstract: " + df["abstract"].fillna("") +
    "\nObjectives: " + df["objectives"].fillna("")
).str.lower().str.slice(0,3000)
print("Text column created for scoring")

# ------------------------------------------------
# RUN PROMPT EXPERIMENTS
# ------------------------------------------------

for prompt_name in PROMPTS:

    print("\nTesting:", prompt_name)

    scores = []

    for i, row in df.iterrows():

        print(f"{prompt_name} → Course {i+1}/{len(df)}")

        result = gemini_score(row["text"], prompt_name)

        result["course_index"] = i

        scores.append(result)

        time.sleep(SLEEP_TIME)

    prompt_df = pd.DataFrame(scores)

    output_file = f"validation_scores_{prompt_name}.csv"

    prompt_df.to_csv(output_file, index=False)

    print("Saved:", output_file)

print("\nPrompt experiment complete.")