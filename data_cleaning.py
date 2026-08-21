import pandas as pd

# 1. LOAD DATASET

df = pd.read_csv("data/Pakistan_Jobs.csv")

print("Original Shape:", df.shape)

# 2. REMOVE DUPLICATES

duplicates = df.duplicated().sum()

print("Duplicated rows:", duplicates)

df = df.drop_duplicates().copy()

# 3. HANDLE MISSING VALUES

df["Company Name"] = df["Company Name"].fillna("Unknown Company")

df = df.drop(columns=["label"])

# 4. CLEAN TEXT COLUMNS

text_columns = [
    "Job Name",
    "Company Name",
    "Job Type",
    "Experience Required",
    "Department",
    "JD",
    "City"
]

for column in text_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

# 5. CONVERT DATE

df["Date Posted"] = pd.to_datetime(
    df["Date Posted"],
    format="%d-%b-%y",
    errors="coerce"
)

# 6. EXPERIENCE PROCESSING

def extract_min_experience(value):

    value = str(value).strip()

    if "< 1 Year" in value:
        return 0

    if "Fresh Graduates" in value:
        return 0

    if "Students" in value:
        return 0

    if "Year Job Exp." in value or "Years Job Exp." in value:
        return int(value.split()[0])

    return 0


def extract_max_experience(value):

    value = str(value).strip()

    if "< 1 Year" in value:
        return 1

    if "Fresh Graduates" in value:
        return 0

    if "Students" in value:
        return 0

    if "Year Job Exp." in value or "Years Job Exp." in value:
        return int(value.split()[0])

    return 0


df["Min Experience"] = df["Experience Required"].apply(
    extract_min_experience
)

df["Max Experience"] = df["Experience Required"].apply(
    extract_max_experience
)

# 7. EXPERIENCE CATEGORY

def categorize_experience(value):

    value = str(value).strip()

    if "Students" in value:
        return "Student"

    if "Fresh Graduates" in value:
        return "Fresh Graduate"

    if "< 1 Year" in value:
        return "Entry Level"

    return "Experienced"


df["Experience Category"] = df["Experience Required"].apply(
    categorize_experience
)

# 8. CHECK EXPERIENCE

print("\nExperience Analysis:")

print(
    df[
        [
            "Experience Required",
            "Min Experience",
            "Max Experience",
            "Experience Category"
        ]
    ].head(20)
)


print("\nExperience Categories:")

print(
    df["Experience Category"].value_counts()
)

# 9. FINAL DATASET CHECK

print("\nCleaned Shape:", df.shape)

print("\nMissing values after cleaning:")

print(df.isnull().sum())

print("\nData Types:")

print(df.dtypes)

# 10. SAVE CLEAN DATASET

df.to_csv(
    "data/cleaned_jobs.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
