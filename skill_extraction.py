import pandas as pd
import re

# 1. LOAD CLEANED DATA

df = pd.read_csv("data/cleaned_jobs.csv")

print("Dataset Loaded:", df.shape)

# 2. SKILL DICTIONARY

SKILLS = [
    # Programming Languages
    "Python",
    "Java",
    "JavaScript",
    "C++",
    "C#",
    "PHP",
    "Ruby",
    "Swift",
    "Kotlin",

    # Web Development
    "HTML",
    "CSS",
    "React",
    "Angular",
    "Vue.js",
    "Node.js",
    "Django",
    "Flask",
    "Laravel",
    "WordPress",

    # Data & AI
    "SQL",
    "MySQL",
    "MongoDB",
    "PostgreSQL",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Science",
    "Data Analysis",
    "Power BI",
    "Tableau",
    "Excel",

    # Cloud & DevOps
    "AWS",
    "Azure",
    "Google Cloud",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",

    # Software Engineering
    "REST API",
    "API",
    "OOP",
    "Software Development",
    "Software Engineering",

    # Databases
    "Firebase",
    "Oracle",
    "SQL Server",

    # Professional Skills
    "Communication",
    "Leadership",
    "Teamwork",
    "Problem Solving",
    "Project Management",
    "Time Management",

    # Marketing
    "Marketing",
    "Digital Marketing",
    "SEO",

    # .NET / Microsoft
".NET Core",
".NET",
".Netcore",
"ASP.NET",
"ASP.NET Core",
"AngularJS",
"Flutter",
# Common IT Skills
"TypeScript",
"Bootstrap",
"jQuery",
"GitLab",
"Jenkins",
"Linux",
"Windows Server",

# Business / Office
"MS Office",
"Microsoft Office",
"PowerPoint",
"Word",

# Sales / Customer Support
"Sales",
"Customer Service",
"Customer Support",
"Business Development",

# HR / Management
"Recruitment",
"Human Resources",
"HR",
"Administration",

# Finance
"Accounting",
"Finance",
"QuickBooks",

# Design
"Graphic Design",
"Adobe Photoshop",
"Illustrator",
"UI/UX",
"Figma",
]

def extract_skills(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    found_skills = []

    for skill in SKILLS:

        skill_lower = skill.lower()

        # Special handling for skills containing symbols
        if any(char in skill_lower for char in [".", "+", "#"]):
            pattern = re.escape(skill_lower)
        else:
            pattern = r"\b" + re.escape(skill_lower) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills

df["Search Text"] = (
    df["Job Name"].fillna("") + " " +
    df["JD"].fillna("")
)

df["Extracted Skills"] = df["Search Text"].apply(
    extract_skills
)

EXCLUDED_FROM_RANKING = [
    "API"
]

# ==========================================
# 5. DISPLAY SAMPLE RESULTS
# ==========================================

print("\n" + "=" * 60)
print("SAMPLE SKILL EXTRACTION")
print("=" * 60)

for i in range(10):

    print("\nJob:", df.iloc[i]["Job Name"])

    print(
        "Skills:",
        df.iloc[i]["Extracted Skills"]
    )

    print("-" * 60)

skill_counts = {}

for skills in df["Extracted Skills"]:

    for skill in skills:

        skill_counts[skill] = (
            skill_counts.get(skill, 0) + 1
        )


# Convert to DataFrame

skill_demand = pd.DataFrame(
    list(skill_counts.items()),
    columns=["Skill", "Job Count"]
)
skill_demand = skill_demand[
    ~skill_demand["Skill"].isin(EXCLUDED_FROM_RANKING)
]



# Sort highest demand first

skill_demand = skill_demand.sort_values(
    by="Job Count",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 20 IN-DEMAND SKILLS")
print("=" * 60)

print(
    skill_demand.head(20).to_string(index=False)
)

df.to_csv(
    "data/jobs_with_skills.csv",
    index=False
)

skill_demand.to_csv(
    "data/skill_demand.csv",
    index=False
)

print("\n" + "=" * 60)
print("SKILL EXTRACTION COMPLETED")
print("=" * 60)

print("Saved:")
print("→ data/jobs_with_skills.csv")
print("→ data/skill_demand.csv")
