import pandas as pd

# --------------------------------
# SETTINGS
# --------------------------------

DATA_FILE = "final_thesis_data.csv"
VALIDATION_SIZE = 73
RANDOM_SEED = 42

# --------------------------------
# LOAD DATASET
# --------------------------------

df = pd.read_csv(DATA_FILE)

print("Total courses in dataset:", len(df))

# --------------------------------
# NORMALIZE TEXT FIELDS
# --------------------------------

for col in ["keywords", "abstract", "objectives"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# --------------------------------
# REMOVE COURSES WITH EMPTY TEXT
# --------------------------------

df = df[
    (df["keywords"] != "") &
    (df["abstract"] != "") &
    (df["objectives"] != "")
]

print("Courses with complete syllabus text:", len(df))

# merged text field (same format used in main project)

df["text"] = (
    "Keywords: " + df["keywords"] +
    "\nAbstract: " + df["abstract"] +
    "\nObjectives: " + df["objectives"]
).str.lower()

df["text"] = df["text"].str.slice(0, 3000)


# --------------------------------
# STRATIFIED RANDOM SAMPLE BY STUDY PROGRAMME
# --------------------------------

program_counts = df["study_programme"].value_counts()

# compute proportional sample sizes
program_sample_sizes = (
    program_counts / len(df) * VALIDATION_SIZE
).round().astype(int)

# adjust if rounding causes mismatch
difference = VALIDATION_SIZE - program_sample_sizes.sum()

if difference != 0:
    program_sample_sizes.iloc[0] += difference

samples = []

for program, size in program_sample_sizes.items():

    subset = df[df["study_programme"] == program]

    sample = subset.sample(
        n=size,
        random_state=RANDOM_SEED
    )

    samples.append(sample)

validation_df = pd.concat(samples).reset_index(drop=True)

print("Validation dataset created:", len(validation_df), "courses")

# --------------------------------
# EXPORT DATASET
# --------------------------------

validation_df.to_csv(
    "validation_courses_73.csv",
    index=False
)

print("File saved: validation_courses_73.csv")

# --------------------------------
# CREATE MANUAL SCORING TEMPLATE
# --------------------------------

competencies = ["V1","V2","V3","H1","H2","H3","H4","H5"]

manual_template = validation_df[
    ["code", "name","keywords","abstract","objectives","text"]
].copy()

for c in competencies:
    manual_template[c+"_manual"] = ""

manual_template.to_csv(
    "validation_manual_scoring_template.csv",
    index=False
)

print("Manual scoring template created")