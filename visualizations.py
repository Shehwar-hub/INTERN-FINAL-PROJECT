import pandas as pd
import ast
import plotly.express as px


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/jobs_with_skills.csv")

df["Date Posted"] = pd.to_datetime(
    df["Date Posted"],
    errors="coerce"
)

df["Extracted Skills"] = df["Extracted Skills"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)


# ==========================================
# NORMALIZE DATA
# ==========================================

df["Department"] = (
    df["Department"]
    .str.strip()
    .str.replace(r"\s+Jobs?$", "", regex=True)
    .str.strip()
)

df["Job Type"] = (
    df["Job Type"]
    .str.strip()
    .str.replace(r"\s+Jobs?$", "", regex=True)
    .str.strip()
)


# ==========================================
# 1. TOP SKILLS
# ==========================================

skill_counts = {}

for skills in df["Extracted Skills"]:

    for skill in skills:

        if skill != "API":
            skill_counts[skill] = (
                skill_counts.get(skill, 0) + 1
            )

skill_df = pd.DataFrame(
    list(skill_counts.items()),
    columns=["Skill", "Job Count"]
)

skill_df = skill_df.sort_values(
    "Job Count",
    ascending=False
).head(10)


fig1 = px.bar(
    skill_df,
    x="Job Count",
    y="Skill",
    orientation="h",
    title="Top 10 In-Demand Skills"
)

fig1.show()


# ==========================================
# 2. JOBS BY CITY
# ==========================================

city_df = (
    df["City"]
    .value_counts()
    .head(10)
    .reset_index()
)

city_df.columns = ["City", "Job Count"]

fig2 = px.bar(
    city_df,
    x="City",
    y="Job Count",
    title="Top 10 Cities by Job Opportunities"
)

fig2.show()


# ==========================================
# 3. EXPERIENCE DISTRIBUTION
# ==========================================

experience_df = (
    df["Experience Category"]
    .value_counts()
    .reset_index()
)

experience_df.columns = [
    "Experience",
    "Job Count"
]

fig3 = px.pie(
    experience_df,
    names="Experience",
    values="Job Count",
    title="Experience Requirements"
)

fig3.show()


# ==========================================
# 4. DEPARTMENT DISTRIBUTION
# ==========================================

department_df = (
    df["Department"]
    .value_counts()
    .head(10)
    .reset_index()
)

department_df.columns = [
    "Department",
    "Job Count"
]

fig4 = px.bar(
    department_df,
    x="Job Count",
    y="Department",
    orientation="h",
    title="Top Job Departments"
)

fig4.show()


# ==========================================
# 5. MONTHLY JOB TREND
# ==========================================

df["Month"] = (
    df["Date Posted"]
    .dt.to_period("M")
    .astype(str)
)

monthly_df = (
    df["Month"]
    .value_counts()
    .sort_index()
    .reset_index()
)

monthly_df.columns = [
    "Month",
    "Job Count"
]

fig5 = px.line(
    monthly_df,
    x="Month",
    y="Job Count",
    markers=True,
    title="Job Posting Trend"
)

fig5.show()


# ==========================================
# 6. JOB TYPE
# ==========================================

jobtype_df = (
    df["Job Type"]
    .value_counts()
    .reset_index()
)

jobtype_df.columns = [
    "Job Type",
    "Job Count"
]

fig6 = px.pie(
    jobtype_df,
    names="Job Type",
    values="Job Count",
    title="Job Type Distribution"
)

fig6.show()