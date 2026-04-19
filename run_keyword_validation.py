import pandas as pd
import re

# ------------------------------------------------
# LOAD VALIDATION DATASET
# ------------------------------------------------

df = pd.read_csv("validation_courses_73.csv")

print("Validation dataset loaded:", len(df), "courses")



# ------------------------------------------------
# NORMALIZE TEXT FIELDS
# ------------------------------------------------

for col in ["keywords","abstract","objectives"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# ------------------------------------------------
# KEYWORD DICTIONARIES (FROM YOUR ORIGINAL SCRIPT)
# ------------------------------------------------

vertical_keywords = {

"V1":[
"transport","transportation","infrastructure","traffic",
"railway","aviation","aircraft","airframe","aerodynamic",
"propulsion","navigation","engineering","mechanic","structur",
"static","thermodynamic","control system","logistic",
"supply chain","safety","regulat","legislation",
"law","convention","icao","easa"
],

"V2":[
"analysis","analys","optimization","optimiz","algorithm",
"statistics","statistical","model","modeling","modelling",
"problem solving","problem","evaluation","evaluat","calculation",
"calculat","assessment","assess","quantitative","quantitativ",
"risk analysis","risk management","logic","derivative",
"mathematics","math","economic","economics","finance"
],

"V3":[
"software","programming","python","cad","gis",
"simulation","simulat","data analysis","data processing",
"database","sensor","instrumentation","measurement",
"laboratory","experiment","iot","automation",
"navigation system","uav","drone","ui","ux"
]
}

horizontal_keywords = {

"H1":[
"team","teamwork","group work","group project",
"collaborat","cooperat","joint","peer",
"leadership","interpersonal","collectiv",
"working with others","team-based","soft skill"
],

"H2":[
"communication","presentation","present",
"report","documentation","discussion",
"written","oral","seminar",
"terminology","defense","thesis",
"technical writing","public speaking"
],

"H3":[
"interdisciplinary","multidisciplinary",
"cross-domain","cross-disciplinary",
"integration","integrat",
"system","systems thinking","system approach",
"complex system","holistic",
"system perspective","interconnect"
],

"H4":[
"project","design project","case study",
"assignment","applied","practical",
"hands-on","laboratory","workshop",
"field work","seminar project",
"internship","design task"
],

"H5":[
"industry","professional practice","stakeholder",
"company","corporate","firm",
"market","commercial",
"expert","guest lecture",
"certification","regulation",
"sector","real-world application"
]
}

# ------------------------------------------------
# KEYWORD SCORING FUNCTION (FROM YOUR SCRIPT)
# ------------------------------------------------

def score_competency(text, keywords):

    words = len(text.split())

    # prevent division by zero
    if words == 0:
        return 0

    matches = {
        word for word in keywords
        if re.search(r"\b" + re.escape(word) + r"\b", text)
    }

    density = len(matches) / words

    if density == 0:
        return 0
    elif density < 0.003:
        return 1
    elif density < 0.008:
        return 2
    else:
        return 3

# ------------------------------------------------
# APPLY KEYWORD SCORING (SECTION WEIGHTED)
# ------------------------------------------------

for k,v in vertical_keywords.items():

    keyword_score = df["keywords"].apply(lambda x: score_competency(x.lower(), v))
    abstract_score = df["abstract"].apply(lambda x: score_competency(x.lower(), v))
    objective_score = df["objectives"].apply(lambda x: score_competency(x.lower(), v))

    df[k] = (keyword_score + 2*abstract_score + 3*objective_score) / 6


for k,v in horizontal_keywords.items():

    keyword_score = df["keywords"].apply(lambda x: score_competency(x.lower(), v))
    abstract_score = df["abstract"].apply(lambda x: score_competency(x.lower(), v))
    objective_score = df["objectives"].apply(lambda x: score_competency(x.lower(), v))

    df[k] = (keyword_score + 2*abstract_score + 3*objective_score) / 6


# ------------------------------------------------
# DISCRETIZE SCORES TO MATCH MANUAL / LLM SCALE
# ------------------------------------------------

competencies = ["V1","V2","V3","H1","H2","H3","H4","H5"]

df[competencies] = df[competencies].round().clip(0,3).astype(int)

# ------------------------------------------------
# EXPORT RESULTS
# ------------------------------------------------

scores = df[competencies]

scores.to_csv(
    "validation_scores_keyword.csv",
    index=False
)

print("Keyword scoring completed")
print("Saved: validation_scores_keyword.csv")