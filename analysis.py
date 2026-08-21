import pandas as pd
import ast

# 1. LOAD DATA

df = pd.read_csv("data/jobs_with_skills.csv")

# Convert Date Posted back to datetime
df["Date Posted"] = pd.to_datetime(
    df["Date Posted"],
    errors="coerce"
)

print("Dataset Loaded:", df.shape)

# 2. CONVERT EXTRACTED SKILLS BACK TO LIST

df["Extracted Skills"] = df["Extracted Skills"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# 3. BASIC KPIs

total_jobs = len(df)

total_companies = df["Company Name"].nunique()

total_cities = df["City"].nunique()

total_departments = df["Department"].nunique()


print("\n" + "=" * 60)
print("JOBLENS MARKET OVERVIEW")
print("=" * 60)

print("Total Jobs:", total_jobs)
print("Total Companies:", total_companies)
print("Total Cities:", total_cities)
print("Total Departments:", total_departments)

# 4. TOP JOB ROLES

top_jobs = (
    df["Job Name"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 60)
print("TOP JOB ROLES")
print("=" * 60)

print(top_jobs)

# 5. TOP CITIES

top_cities = (
    df["City"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 60)
print("TOP CITIES")
print("=" * 60)

print(top_cities)

# 6. TOP DEPARTMENTS

df["Department"] = (
    df["Department"]
    .str.strip()
    .str.replace(r"\s+Jobs?$", "", regex=True)
    .str.strip()
)

df["Department"] = df["Department"].replace({
    "IT": "IT",
    "Computer Software": "Computer Software",
    "Customer Service": "Customer Service",
    "Web & E-commerce": "Web & E-commerce",
    "Sales": "Sales",
    "Marketing": "Marketing",
    "Admin": "Admin"
})
top_departments = (
    df["Department"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 60)
print("TOP DEPARTMENTS")
print("=" * 60)

print(top_departments)

# 7. JOB TYPE DISTRIBUTION

df["Job Type"] = (
    df["Job Type"]
    .str.strip()
    .str.replace(r"\s+Jobs?$", "", regex=True)
    .str.strip()
)
job_types = (
    df["Job Type"]
    .value_counts()
)

print("\n" + "=" * 60)
print("JOB TYPES")
print("=" * 60)

print(job_types)

# 8. EXPERIENCE DISTRIBUTION

experience = (
    df["Experience Category"]
    .value_counts()
)

print("\n" + "=" * 60)
print("EXPERIENCE DISTRIBUTION")
print("=" * 60)

print(experience)

# 9. TOP COMPANIES

top_companies = (
    df["Company Name"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 60)
print("TOP COMPANIES")
print("=" * 60)

print(top_companies)


# ==========================================
# 10. SKILL DEMAND
# ==========================================

skill_counts = {}

for skills in df["Extracted Skills"]:

    for skill in skills:

        skill_counts[skill] = (
            skill_counts.get(skill, 0) + 1
        )


skill_demand = pd.DataFrame(
    list(skill_counts.items()),
    columns=["Skill", "Job Count"]
)


skill_demand = skill_demand.sort_values(
    by="Job Count",
    ascending=False
)


# Remove generic API from ranking

skill_demand = skill_demand[
    skill_demand["Skill"] != "API"
]


print("\n" + "=" * 60)
print("TOP 20 IN-DEMAND SKILLS")
print("=" * 60)

print(
    skill_demand.head(20).to_string(index=False)
)


# ==========================================
# 11. MONTHLY JOB POSTINGS
# ==========================================

df["Month"] = df["Date Posted"].dt.to_period("M").astype(str)

monthly_jobs = (
    df["Month"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("MONTHLY JOB POSTINGS")
print("=" * 60)

print(monthly_jobs)

# 12. SAVE ANALYTICS DATA

top_cities.to_csv(
    "data/top_cities.csv"
)

top_departments.to_csv(
    "data/top_departments.csv"
)

job_types.to_csv(
    "data/job_types.csv"
)

experience.to_csv(
    "data/experience_distribution.csv"
)

top_companies.to_csv(
    "data/top_companies.csv"
)

skill_demand.to_csv(
    "data/skill_demand.csv",
    index=False
)

monthly_jobs.to_csv(
    "data/monthly_jobs.csv"
)


print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)
