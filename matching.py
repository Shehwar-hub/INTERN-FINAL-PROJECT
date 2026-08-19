import pandas as pd
import ast


DATA_PATH = "data/jobs_with_skills.csv"

df = pd.read_csv(DATA_PATH)


def parse_skills(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

        return []

    except (ValueError, SyntaxError):
        return []


df["Extracted Skills"] = df["Extracted Skills"].apply(parse_skills)


def normalize_skill(skill):
    if not isinstance(skill, str):
        return ""

    return skill.strip().lower()


CAREER_GROUPS = {
    "Data & Analytics": {
        "python",
        "sql",
        "excel",
        "pandas",
        "numpy",
        "power bi",
        "tableau",
        "data analysis",
        "data science",
        "machine learning",
        "scikit-learn",
        "mysql",
        "postgresql",
        "sql server",
        "oracle"
    },

    "Web Development": {
        "html",
        "css",
        "javascript",
        "typescript",
        "react",
        "angular",
        "angularjs",
        "vue.js",
        "node.js",
        "django",
        "flask",
        "php",
        "laravel",
        "wordpress",
        "bootstrap",
        "jquery"
    },

    "Software Development": {
        "python",
        "java",
        "c++",
        "c#",
        ".net",
        ".net core",
        "asp.net",
        "asp.net core",
        "javascript",
        "typescript",
        "git",
        "github",
        "oop",
        "software development",
        "software engineering",
        "sql",
        "sql server"
    },

    "Marketing": {
        "marketing",
        "digital marketing",
        "seo",
        "sales",
        "business development",
        "social media marketing"
    }
}


def detect_career_group(user_skills):
    normalized_skills = {
        normalize_skill(skill)
        for skill in user_skills
        if normalize_skill(skill)
    }

    scores = {}

    for group, skills in CAREER_GROUPS.items():
        matched = normalized_skills.intersection(skills)

        if matched:
            scores[group] = len(matched)

    if not scores:
        return "General"

    return max(scores, key=scores.get)


def get_skill_demand(data):
    skill_counts = {}

    for skills in data["Extracted Skills"]:
        for skill in skills:
            skill = normalize_skill(skill)

            if not skill or skill == "api":
                continue

            skill_counts[skill] = (
                skill_counts.get(skill, 0) + 1
            )

    return skill_counts


def analyze_skill_gap(user_skills, data):
    user_skills = {
        normalize_skill(skill)
        for skill in user_skills
        if normalize_skill(skill)
    }

    career_group = detect_career_group(user_skills)

    career_skills = CAREER_GROUPS.get(
        career_group,
        set()
    )

    relevant_skill_counts = {}
    relevant_jobs = 0

    for _, row in data.iterrows():

        job_skills = {
            normalize_skill(skill)
            for skill in row["Extracted Skills"]
            if normalize_skill(skill)
        }

        job_skills.discard("api")

        if not job_skills:
            continue

        matched = user_skills.intersection(job_skills)

        if not matched:
            continue

        relevant_jobs += 1

        missing = job_skills - user_skills

        for skill in missing:

            if skill == "api":
                continue

            if career_group != "General":
                if skill not in career_skills:
                    continue

            relevant_skill_counts[skill] = (
                relevant_skill_counts.get(skill, 0) + 1
            )

    missing_skills = sorted(
        relevant_skill_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    matched_skills = sorted(user_skills)

    return (
        matched_skills,
        missing_skills,
        relevant_jobs,
        career_group
    )


def calculate_job_matches(user_skills, data):
    user_skills = {
        normalize_skill(skill)
        for skill in user_skills
        if normalize_skill(skill)
    }

    career_group = detect_career_group(user_skills)
    career_skills = CAREER_GROUPS.get(career_group, set())

    results = []

    for _, row in data.iterrows():

        job_skills = {
            normalize_skill(skill)
            for skill in row["Extracted Skills"]
            if normalize_skill(skill)
        }

        job_skills.discard("api")

        if not job_skills:
            continue

        matched = user_skills.intersection(job_skills)

        if not matched:
            continue

        missing = job_skills - user_skills

        career_matches = job_skills.intersection(
            career_skills
        )

        if career_group != "General" and not career_matches:
            continue

        user_coverage = (
            len(matched) / len(user_skills)
        ) * 100

        job_coverage = (
            len(matched) / len(job_skills)
        ) * 100

        if career_group != "General":
            career_relevance = (
                len(career_matches) /
                max(len(job_skills), 1)
            ) * 100
        else:
            career_relevance = 50

        match_score = (
            user_coverage * 0.45
            + job_coverage * 0.30
            + career_relevance * 0.25
        )

        if match_score < 25:
            continue

        results.append({
            "Job Name": row["Job Name"],
            "Company Name": row["Company Name"],
            "City": row["City"],
            "Experience": row["Experience Required"],
            "Match Score": round(match_score, 1),
            "Career Relevance": round(
                career_relevance,
                1
            ),
            "Matched Skills": ", ".join(
                sorted(matched)
            ),
            "Missing Skills": ", ".join(
                sorted(missing)
            )
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by=[
                "Match Score",
                "Career Relevance"
            ],
            ascending=False
        )

        results_df = results_df.reset_index(
            drop=True
        )

    return results_df
if __name__ == "__main__":

    my_skills = [
        "Python",
        "SQL",
        "Excel"
    ]

    matched, missing, relevant_jobs, career_group = (
        analyze_skill_gap(
            my_skills,
            df
        )
    )

    print("\n" + "=" * 60)
    print("SKILL GAP ANALYSIS")
    print("=" * 60)

    print("\nYour matched skills:")
    print(matched)

    print(f"\nDetected Career Path: {career_group}")

    print(f"\nRelevant jobs found: {relevant_jobs}")

    print("\nTop recommended skills:")

    if missing:
        for skill, count in missing[:10]:
            print(
                f"{skill} → appears in "
                f"{count} relevant jobs"
            )
    else:
        print("No additional skills found.")

    matches = calculate_job_matches(
        my_skills,
        df
    )

    print("\n" + "=" * 60)
    print("TOP JOB MATCHES")
    print("=" * 60)

    if matches.empty:

        print("\nNo matching jobs found.")

    else:

        print(
            matches[
                [
                    "Job Name",
                    "Company Name",
                    "City",
                    "Experience",
                    "Match Score",
                    "Matched Skills",
                    "Missing Skills"
                ]
            ].head(10).to_string(
                index=False
            )
        )

    print("\n" + "=" * 60)
    print("MATCHING COMPLETED")
    print("=" * 60)