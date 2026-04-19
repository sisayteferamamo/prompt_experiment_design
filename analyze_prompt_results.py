import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

MANUAL_FILE = "validation_manual_scores.csv"

PROMPT_FILES = {
    "Keyword":"validation_scores_keyword.csv",
    "Prompt_V1": "validation_scores_prompt_v1_basic.csv",
    "Prompt_V2": "validation_scores_prompt_v2_structured.csv",
    "Prompt_V3": "validation_scores_prompt_v3_reasoned.csv",
    "Prompt_V4": "validation_scores_prompt_v4_reasoned.csv",
    "Prompt_V5": "validation_scores_prompt_v5_reasoned.csv"
}

COMPETENCIES = ["V1","V2","V3","H1","H2","H3","H4","H5"]

sns.set_theme(style="whitegrid")

# ------------------------------------------------
# LOAD MANUAL SCORES
# ------------------------------------------------

manual = pd.read_csv(MANUAL_FILE)

print("Manual scores loaded:", len(manual), "courses")

# ------------------------------------------------
# RESULTS STORAGE
# ------------------------------------------------

results = []

# ------------------------------------------------
# COMPUTE METRICS FOR EACH PROMPT
# ------------------------------------------------

for prompt_name, file in PROMPT_FILES.items():

    llm = pd.read_csv(file)

    print("Analyzing:", prompt_name)

    for comp in COMPETENCIES:

        manual_scores = manual[comp]
        llm_scores = llm[comp]

        # Cohen Kappa
        kappa = cohen_kappa_score(manual_scores, llm_scores)

        # Exact agreement
        agreement = (manual_scores == llm_scores).mean()

        # Mean Absolute Error
        mae = np.abs(manual_scores - llm_scores).mean()

        # Correlation
        corr = np.corrcoef(manual_scores, llm_scores)[0,1]

        results.append({
            "Prompt": prompt_name,
            "Competency": comp,
            "Kappa": round(kappa,3),
            "Agreement": round(agreement,3),
            "MAE": round(mae,3),
            "Correlation": round(corr,3)
        })

# ------------------------------------------------
# CREATE RESULTS TABLE
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("prompt_comparison_detailed.csv", index=False)

print("Detailed comparison saved")


# ------------------------------------------------
# AVERAGE PERFORMANCE PER PROMPT
# ------------------------------------------------

summary = results_df.groupby("Prompt").agg({
    "Kappa":"mean",
    "Agreement":"mean",
    "MAE":"mean",
    "Correlation":"mean"
}).reset_index()

summary = summary.round(3)

summary.to_csv("prompt_comparison_summary.csv", index=False)

print("Summary table saved")

print(summary)


# ------------------------------------------------
# PROMPT LEADERBOARD
# ------------------------------------------------

leaderboard = summary.copy()

# rank metrics
leaderboard["Rank_Kappa"] = leaderboard["Kappa"].rank(ascending=False)
leaderboard["Rank_Agreement"] = leaderboard["Agreement"].rank(ascending=False)
leaderboard["Rank_MAE"] = leaderboard["MAE"].rank(ascending=True)
leaderboard["Rank_Correlation"] = leaderboard["Correlation"].rank(ascending=False)

# overall score
leaderboard["Overall_Score"] = (
    leaderboard["Rank_Kappa"] +
    leaderboard["Rank_Agreement"] +
    leaderboard["Rank_MAE"] +
    leaderboard["Rank_Correlation"]
)

leaderboard = leaderboard.sort_values("Overall_Score")

leaderboard.to_csv(
    "prompt_leaderboard.csv",
    index=False
)

print("\nPrompt Leaderboard")
print(leaderboard)

# ------------------------------------------------
# VISUALIZATION: KAPPA COMPARISON
# ------------------------------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    data=summary,
    x="Prompt",
    y="Kappa"
)

plt.title("Prompt Reliability Comparison (Cohen Kappa)")
plt.ylabel("Average Kappa")

plt.savefig("fig_prompt_kappa_comparison.png", dpi=300)

plt.close()

# ------------------------------------------------
# VISUALIZATION: ERROR COMPARISON
# ------------------------------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    data=summary,
    x="Prompt",
    y="MAE"
)

plt.title("Prompt Mean Absolute Error Comparison")
plt.ylabel("Average MAE")

plt.savefig("fig_prompt_error_comparison.png", dpi=300)

plt.close()

# ------------------------------------------------
# COMPETENCY LEVEL KAPPA HEATMAP
# ------------------------------------------------

pivot = results_df.pivot(
    index="Competency",
    columns="Prompt",
    values="Kappa"
)

plt.figure(figsize=(8,6))

sns.heatmap(
    pivot,
    annot=True,
    cmap="viridis",
    fmt=".2f"
)

plt.title("Cohen Kappa by Competency and Prompt")

plt.savefig("fig_prompt_kappa_heatmap.png", dpi=300)

plt.close()

print("Analysis complete.")

# ------------------------------------------------
# CONFUSION MATRICES
# ------------------------------------------------

print("\nGenerating confusion matrices...")

for prompt_name, file in PROMPT_FILES.items():

    llm = pd.read_csv(file)

    for comp in COMPETENCIES:

        manual_scores = manual[comp]
        llm_scores = llm[comp]

        cm = confusion_matrix(
            manual_scores,
            llm_scores,
            labels=[0,1,2,3]
        )

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            fmt="d",
            xticklabels=[0,1,2,3],
            yticklabels=[0,1,2,3]
        )

        plt.xlabel("LLM Score")
        plt.ylabel("Manual Score")

        plt.title(f"Confusion Matrix: {comp} ({prompt_name})")

        filename = f"fig_confusion_{prompt_name}_{comp}.png"

        plt.savefig(filename, dpi=300)

        plt.close()

print("Confusion matrices generated.")

# ------------------------------------------------
# HUMAN vs LLM SCATTER PLOT
# ------------------------------------------------

BEST_PROMPT = "Prompt_v5_reasoned"

print("\nGenerating Human vs LLM reliability plot using:", BEST_PROMPT)

predicted = pd.read_csv(f"validation_scores_{BEST_PROMPT.lower()}.csv")

competencies = ["V1","V2","V3","H1","H2","H3","H4","H5"]

human_scores = []
llm_scores = []

for comp in competencies:

    human_scores.extend(manual[comp].astype(int))
    llm_scores.extend(predicted[comp].astype(int))

plot_df = pd.DataFrame({
    "Human": human_scores,
    "LLM": llm_scores
})

plt.figure(figsize=(7,7))

sns.scatterplot(
    data=plot_df,
    x="Human",
    y="LLM",
    alpha=0.6
)

# perfect agreement line
plt.plot([0,3],[0,3], linestyle="--")

plt.xlabel("Human Score")
plt.ylabel("LLM Score")

plt.title("Human vs LLM Competency Scoring Agreement")

plt.xlim(-0.1,3.1)
plt.ylim(-0.1,3.1)

plt.grid(True)

plt.tight_layout()

plt.savefig("fig_human_vs_llm_scatter.png", dpi=300)

plt.close()

print("Human vs LLM scatter plot saved")

# ------------------------------------------------
# PROMPT LEADERBOARD VISUALIZATION
# ------------------------------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    data=leaderboard,
    x="Prompt",
    y="Kappa"
)

plt.title("Prompt Reliability Leaderboard (Average Cohen Kappa)")
plt.ylabel("Average Kappa")

plt.savefig(
    "fig_prompt_leaderboard.png",
    dpi=300
)

plt.close()