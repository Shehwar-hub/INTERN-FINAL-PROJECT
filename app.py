import streamlit as st
import pandas as pd
import ast
import joblib
import plotly.express as px
from matching import analyze_skill_gap, calculate_job_matches


st.set_page_config(
    page_title="JobLens",
    page_icon="J",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
.stApp {
    background: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-right: 3rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
}

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #ffffff;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #ffffff !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #111827 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #111827 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    color: #111827 !important;
    fill: #111827 !important;
    stroke: #111827 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] svg path {
    fill: #111827 !important;
    stroke: #111827 !important;
}

.brand {
    display: inline-block;
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -1px;
    margin: 12px 0 6px 0;
    white-space: nowrap;
    overflow: visible;
}
.subtitle {
    font-size: 17px;
    font-weight: 600;
    color: #475569;
    margin-bottom: 4px;
}

.description {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 18px;
}

.section-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 8px;
}

.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

.job-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.job-card h3 {
    margin-top: 0;
    margin-bottom: 8px;
    color: #0f172a;
}

.job-card p {
    color: #475569;
    line-height: 1.55;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #eef2ff;
    color: #3730a3 !important;
    font-size: 12px;
    font-weight: 600;
    margin-right: 5px;
    margin-bottom: 5px;
}

.score-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 30px;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    font-weight: 600;
}

div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = pd.read_csv("data/jobs_with_skills.csv")

    data["Date Posted"] = pd.to_datetime(
        data["Date Posted"],
        errors="coerce"
    )

    data["Extracted Skills"] = data["Extracted Skills"].apply(
        lambda x: ast.literal_eval(x)
        if isinstance(x, str)
        else []
    )

    return data


@st.cache_resource
@st.cache_resource
def load_ml_model():
    model = joblib.load(
        "models/job_category_model.pkl"
    )

    vectorizer = joblib.load(
        "models/job_category_vectorizer.pkl"
    )

    return model, vectorizer


df = load_data()
model, vectorizer = load_ml_model()


st.markdown(
    '<div class="brand">JobLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Pakistan Job Market Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Data-driven insights into job demand, skills, career paths and opportunities.'
    '</div>',
    unsafe_allow_html=True
)


st.sidebar.markdown("## Filters")

cities = sorted(
    df["City"].dropna().unique().tolist()
)

selected_city = st.sidebar.selectbox(
    "City",
    ["All"] + cities,
    index=0
)

experience_options = [
    "All",
    "Experienced",
    "Entry Level",
    "Fresh Graduate",
    "Student"
]

selected_experience = st.sidebar.selectbox(
    "Experience Level",
    experience_options,
    index=0
)


filtered_df = df.copy()

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["City"] == selected_city
    ]

if selected_experience != "All":
    filtered_df = filtered_df[
        filtered_df["Experience Category"] == selected_experience
    ]


total_jobs = len(filtered_df)
total_companies = filtered_df["Company Name"].nunique()
total_cities = filtered_df["City"].nunique()
total_departments = filtered_df["Department"].nunique()


tab_overview, tab_gap, tab_ai, tab_jobs = st.tabs(
    [
        "Overview",
        "Skill Gap",
        "ML Predictor",
        "Job Explorer"
    ]
)


with tab_overview:

    st.markdown(
        '<div class="section-label">Market Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Jobs",
        f"{total_jobs:,}"
    )

    col2.metric(
        "Companies",
        f"{total_companies:,}"
    )

    col3.metric(
        "Cities",
        f"{total_cities:,}"
    )

    col4.metric(
        "Departments",
        f"{total_departments:,}"
    )

    st.markdown("")

    skill_counts = {}

    for skills in filtered_df["Extracted Skills"]:
        for skill in skills:
            if skill != "API":
                skill_counts[skill] = (
                    skill_counts.get(skill, 0) + 1
                )

    skill_df = pd.DataFrame(
        list(skill_counts.items()),
        columns=["Skill", "Job Count"]
    )

    if not skill_df.empty:
        skill_df = skill_df.sort_values(
            "Job Count",
            ascending=False
        ).head(10)

    city_df = (
        filtered_df["City"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    city_df.columns = [
        "City",
        "Job Count"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Top In-Demand Skills")

        if not skill_df.empty:

            fig = px.bar(
                skill_df,
                x="Job Count",
                y="Skill",
                orientation="h"
            )

            fig.update_layout(
                yaxis=dict(
                    categoryorder="total ascending"
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No skill data available."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Top Job Cities")

        if not city_df.empty:

            fig = px.bar(
                city_df,
                x="City",
                y="Job Count"
            )

            fig.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No city data available."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Experience Requirements")

        experience_df = (
            filtered_df["Experience Category"]
            .value_counts()
            .reset_index()
        )

        experience_df.columns = [
            "Experience",
            "Job Count"
        ]

        if not experience_df.empty:

            fig = px.pie(
                experience_df,
                names="Experience",
                values="Job Count",
                hole=0.45
            )

            fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Job Posting Trend")

        trend_df = filtered_df.copy()

        trend_df["Month"] = (
            trend_df["Date Posted"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_df = (
            trend_df["Month"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        monthly_df.columns = [
            "Month",
            "Job Count"
        ]

        if not monthly_df.empty:

            fig = px.line(
                monthly_df,
                x="Month",
                y="Job Count",
                markers=True
            )

            fig.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=10,
                    b=10
                ),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


with tab_gap:

    st.markdown(
        '<div class="section-label">Personalized Career Analysis</div>',
        unsafe_allow_html=True
    )

    st.header("Skill Gap Analyzer")

    st.write(
        "Enter your current skills to identify relevant jobs, "
        "your career path and market-demanded skills."
    )

    user_input = st.text_input(
        "Current Skills",
        placeholder="Example: Python, SQL, Excel"
    )

    if st.button(
        "Analyze Skills",
        key="analyze_skills"
    ):

        if not user_input.strip():

            st.warning(
                "Please enter at least one skill."
            )

        else:

            user_skills = [
                skill.strip()
                for skill in user_input.split(",")
                if skill.strip()
            ]

            (
                matched_skills,
                missing_skills,
                relevant_jobs,
                career_group
            ) = analyze_skill_gap(
                user_skills,
                filtered_df
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Your Skills",
                len(user_skills)
            )

            col2.metric(
                "Matched Skills",
                len(matched_skills)
            )

            col3.metric(
                "Relevant Jobs",
                relevant_jobs
            )

            st.info(
                f"Detected Career Path: {career_group}"
            )

            st.subheader("Your Skills")

            if matched_skills:

                skill_text = "  ".join(
                    f"`{skill.title()}`"
                    for skill in matched_skills
                )

                st.markdown(
                    skill_text
                )

            else:

                st.info(
                    "None of your entered skills were found "
                    "in the current dataset."
                )

            st.subheader("Recommended Skills")

            if missing_skills:

                recommendation_df = pd.DataFrame(
                    missing_skills[:10],
                    columns=[
                        "Skill",
                        "Relevant Job Count"
                    ]
                )

                recommendation_df["Skill"] = (
                    recommendation_df["Skill"].str.title()
                )

                st.dataframe(
                    recommendation_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "Your skills cover the identified relevant jobs."
                )

            st.divider()

            st.subheader("Best Matching Jobs")

            matches = calculate_job_matches(
                user_skills,
                filtered_df
            )

            if matches.empty:

                st.info(
                    "No matching jobs found. Try adding more skills."
                )

            else:

                for _, match in matches.head(10).iterrows():

                    score = match["Match Score"]

                    if score >= 75:
                        rating = "Excellent Match"
                    elif score >= 55:
                        rating = "Good Match"
                    else:
                        rating = "Partial Match"

                    st.markdown(
                        f"""
                        <div class="job-card">
                            <h3>{match["Job Name"]}</h3>
                            <p>
                                <b>Company:</b> {match["Company Name"]}
                                &nbsp;&nbsp;|&nbsp;&nbsp;
                                <b>City:</b> {match["City"]}
                                &nbsp;&nbsp;|&nbsp;&nbsp;
                                <b>Experience:</b> {match["Experience"]}
                            </p>
                            <p>
                                <b>Match Score:</b> {score}%
                                &nbsp;&nbsp;
                                <span class="badge">{rating}</span>
                            </p>
                            <p>
                                <b>Matched Skills:</b>
                                {match["Matched Skills"] or "None"}
                            </p>
                            <p>
                                <b>Missing Skills:</b>
                                {match["Missing Skills"] or "None"}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


with tab_ai:

    st.markdown(
        '<div class="section-label">Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.header("ML Job Category Predictor")

    st.write(
     "Predict the most likely broad job category "
     "using a trained machine-learning model."
    )

    col1, col2 = st.columns(2)

    with col1:

        prediction_title = st.text_input(
            "Job Title",
            placeholder="Example: Python Data Analyst",
            key="prediction_title"
        )

    with col2:

        prediction_description = st.text_area(
            "Job Description",
            placeholder=(
                "Example: Looking for a Python developer "
                "with SQL, Pandas and data analysis experience."
            ),
            height=160,
            key="prediction_description"
        )

    if st.button(
        "Predict Job Category",
        key="predict_category"
    ):

        if (
            not prediction_title.strip()
            and not prediction_description.strip()
        ):

            st.warning(
                "Please enter a job title or job description."
            )

        else:

            prediction_text = (
                prediction_title + " " +
                prediction_description
            )

            prediction_vector = vectorizer.transform(
                [prediction_text]
            )

            probabilities = model.predict_proba(
                prediction_vector
            )[0]

            class_probabilities = list(
                zip(
                    model.classes_,
                    probabilities
                )
            )

            class_probabilities.sort(
                key=lambda x: x[1],
                reverse=True
            )

            top_prediction = class_probabilities[0][0]

            top_confidence = (
                class_probabilities[0][1] * 100
            )

            st.metric(
    "Predicted Category",
    top_prediction
)

            st.success(f"JobLens predicts this job belongs to the **{top_prediction}** category.")

            st.subheader("Top 3 Predictions")

            top_three = class_probabilities[:3]

            prediction_df = pd.DataFrame(
                top_three,
                columns=["Category", "Probability"]
            )

            prediction_df["Probability"] = (
                prediction_df["Probability"] * 100
            ).round(1)

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )


with tab_jobs:

    st.markdown(
        '<div class="section-label">Job Search</div>',
        unsafe_allow_html=True
    )

    st.header("Job Explorer")

    st.write(
        "Search and filter job postings by title, company, city and experience."
    )

    col1, col2 = st.columns(2)

    with col1:

        search_job = st.text_input(
            "Search Job Title",
            placeholder="Example: Python Developer",
            key="search_job"
        )

    with col2:

        search_company = st.text_input(
            "Search Company",
            placeholder="Example: ibex",
            key="search_company"
        )

    col3, col4 = st.columns(2)

    with col3:

        search_city = st.selectbox(
            "Select City",
            ["All"] + sorted(
                df["City"].dropna().unique().tolist()
            ),
            key="explorer_city"
        )

    with col4:

        search_experience = st.selectbox(
            "Experience Level",
            ["All"] + sorted(
                df["Experience Category"]
                .dropna()
                .unique()
                .tolist()
            ),
            key="explorer_experience"
        )

    explorer_df = df.copy()

    if search_job:

        explorer_df = explorer_df[
            explorer_df["Job Name"].str.contains(
                search_job,
                case=False,
                na=False
            )
        ]

    if search_company:

        explorer_df = explorer_df[
            explorer_df["Company Name"].str.contains(
                search_company,
                case=False,
                na=False
            )
        ]

    if search_city != "All":

        explorer_df = explorer_df[
            explorer_df["City"] == search_city
        ]

    if search_experience != "All":

        explorer_df = explorer_df[
            explorer_df["Experience Category"]
            == search_experience
        ]

    has_search_filter = (
        bool(search_job)
        or bool(search_company)
        or search_city != "All"
        or search_experience != "All"
    )

    if not has_search_filter:

        st.info(
            "Use the search box or filters above to explore specific jobs."
        )

    elif explorer_df.empty:

        st.warning(
            "No jobs found. Try changing your search or filters."
        )

    else:

        st.markdown(
            f"### {len(explorer_df):,} Jobs Found"
        )

        for _, job in explorer_df.head(20).iterrows():

            skills = job["Extracted Skills"]

            if isinstance(skills, str):

                try:
                    skills = ast.literal_eval(skills)

                except (ValueError, SyntaxError):
                    skills = []

            st.markdown(
                '<div class="job-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"### {job['Job Name']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"<b>Company:</b> {job['Company Name']}",
                    unsafe_allow_html=True
                )

            with col2:

                st.write(
                    f"<b>City:</b> {job['City']}",
                    unsafe_allow_html=True
                )

            with col3:

                st.write(
                    f"<b>Experience:</b> {job['Experience Required']}",
                    unsafe_allow_html=True
                )

            st.write(
                f"<b>Department:</b> {job['Department']}",
                unsafe_allow_html=True
            )

            if pd.notna(job["Date Posted"]):

                st.write(
                    f"<b>Posted:</b> "
                    f"{job['Date Posted'].strftime('%d %b %Y')}",
                    unsafe_allow_html=True
                )

            if skills:

                st.write(
                    "**Required Skills**"
                )

                badge_html = ""

                for skill in skills[:10]:

                    badge_html += (
                        f'<span class="badge">'
                        f'{skill}'
                        f'</span>'
                    )

                st.markdown(
                    badge_html,
                    unsafe_allow_html=True
                )

            with st.expander(
                "View Job Description"
            ):

                st.write(
                    job["JD"]
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


st.markdown(
    '<div class="footer">'
    'JobLens | Pakistan Job Market Intelligence'
    '</div>',
    unsafe_allow_html=True
)