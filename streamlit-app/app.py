# =============================================================================
# STUDENT PERFORMANCE INTELLIGENCE SYSTEM
# =============================================================================
# Features: Pass/Fail, Leaderboard, Teacher Dashboard, Smart Insights,
#           Offline Chatbot, Gemini AI, Early Warning, Trend Analysis,
#           Multi-Student Comparison, PDF Export, UI Improvements
# =============================================================================

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch
from io import StringIO, BytesIO
import re

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional Kaleido (for chart-in-PDF export)
try:
    import plotly.io as pio
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Smart Student Performance Intelligence System")

# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_sample_csv():
    return """Name,Math,Science,English,Physics,Chemistry,Computer,Attendance,Study_Hours,Previous_Score,Assignments,Sports,Participation,Sleep_Hours,Screen_Time,Stress_Level
Aarav,78,85,80,82,79,88,90,3,75,80,1,7,7,3,4
Diya,45,50,48,42,46,55,60,1,52,50,0,4,5,6,8
Rahul,88,92,90,91,89,95,95,4,85,90,1,9,7,2,3
Ananya,60,58,62,65,61,67,70,2,65,60,0,5,6,4,5
Kiran,35,40,38,32,36,42,55,1,45,40,0,3,5,7,9
Sneha,82,78,85,80,83,87,88,3,80,85,1,8,7,3,4
Meera,91,89,94,93,92,96,96,5,90,95,1,10,8,2,2
Simran,92,95,94,96,93,98,98,5,91,96,1,10,8,2,2
Arjun,55,60,58,54,57,63,65,2,58,55,0,5,6,5,6
Priya,72,68,74,70,73,76,80,3,70,72,1,7,7,3,5
Rohan,40,45,43,38,41,48,58,1,48,42,0,4,5,7,8
Kavya,85,87,89,84,86,91,92,4,83,88,1,9,7,2,3
"""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("📊 Controls")

st.sidebar.download_button(
    "📥 Download Sample Dataset",
    data=get_sample_csv(),
    file_name="students_sample.csv",
    mime="text/csv",
)

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

def load_sample():
    return pd.read_csv(StringIO(get_sample_csv()))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded")
else:
    df = load_sample()
    st.sidebar.info("ℹ️ Using sample dataset")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Gemini AI Settings")
gemini_api_key = st.sidebar.text_input(
    "Google Gemini API Key", type="password", placeholder="Enter API key (optional)"
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove duplicate column names
df = df.loc[:, ~df.columns.duplicated()]

# -----------------------------
# MISSING VALUES ANALYSIS
# -----------------------------
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

# Store columns having missing values
missing_info = missing_values[missing_values > 0]

# -----------------------------
# DUPLICATE ANALYSIS
# -----------------------------
duplicate_count = df.duplicated().sum()

# Remove duplicate rows
df = df.drop_duplicates()

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

# Numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Fill numeric missing values with mean
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Name column
name_col = [col for col in df.columns if "name" in col][0]
# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
target_cols = ["attendance", "study_hours", "previous_score"]
subject_cols = [col for col in numeric_cols if col not in target_cols]

df["Average"] = df[subject_cols].mean(axis=1).round(2)
prev_score_col = "previous_score" if "previous_score" in df.columns else None
df["Trend"] = (df["Average"] - df[prev_score_col]).round(2) if prev_score_col else 0

# Pass / Fail
df["Pass_Fail"] = df["Average"].apply(lambda x: "Pass" if x >= 50 else "Fail")

# Risk
def get_risk(row):
    att = row.get("attendance", 100)
    if row["Average"] < 40 or att < 60:
        return "High"
    elif row["Average"] < 60:
        return "Moderate"
    return "Low"

df["Risk"] = df.apply(get_risk, axis=1)

# Rank
df["Rank"] = df["Average"].rank(ascending=False, method="min").astype(int)
df_ranked = df.sort_values("Rank")

# ─────────────────────────────────────────────
# ML MODEL (Linear Regression)
# ─────────────────────────────────────────────
features = subject_cols.copy()
extra_cols = [
    "attendance", "study_hours", "assignments",
    "sports", "participation", "sleep_hours",
    "screen_time", "stress_level",
]
for col in extra_cols:
    if col in df.columns:
        features.append(col)

features = list(dict.fromkeys(features))

model = LinearRegression()
model.fit(df[features], df["Average"])

# ─────────────────────────────────────────────
# SMART AI INSIGHTS (Rule-based)
# ─────────────────────────────────────────────
def generate_smart_insights(row):
    insights = []
    avg = row.get("Average", 0)
    att = row.get("attendance", 100)
    stress = row.get("stress_level", 0)
    study = row.get("study_hours", 0)
    participation = row.get("participation", 5)
    assignments = row.get("assignments", 1)
    trend = row.get("Trend", 0)

    # Attendance
    if att < 60:
        insights.append(("⚠️ Critical Attendance", f"Attendance is only {att}%. Must attend classes regularly to avoid academic penalties."))
    elif att < 75:
        insights.append(("⚠️ Low Attendance", f"Attendance is {att}%. Try to attend more classes — aim for 85%+."))
    else:
        insights.append(("✅ Good Attendance", f"Attendance is {att}%. Keep it up!"))

    # Stress
    if stress >= 8:
        insights.append(("🔴 High Stress Alert", "Stress level is critically high. Consider counseling, exercise, and better sleep habits."))
    elif stress >= 6:
        insights.append(("🟠 Elevated Stress", "Stress level is above average. Practice mindfulness and time management."))

    # Study hours
    if study < 1.5:
        insights.append(("📚 Increase Study Hours", f"Only {study}h/day of study time. Aim for at least 3 hours daily for improvement."))
    elif study >= 4:
        insights.append(("📚 Dedicated Learner", f"{study}h/day study time shows strong dedication. Maintain structured revision."))

    # Participation
    if participation < 4:
        insights.append(("🗣️ Low Participation", "Low classroom participation. Engaging in discussions improves understanding significantly."))

    # Assignments
    if assignments == 0:
        insights.append(("📝 Assignments Incomplete", "No assignments completed. Completing assignments is crucial for exam preparation."))

    # Trend
    if trend > 5:
        insights.append(("📈 Improving Rapidly", f"Performance improved by {trend} points. Great momentum — keep it going!"))
    elif trend < -5:
        insights.append(("📉 Declining Performance", f"Performance dropped by {abs(trend)} points. Review previous topics and seek teacher guidance."))

    # Weak subjects
    weak = min(subject_cols, key=lambda x: row[x])
    weak_score = row[weak]
    if weak_score < 50:
        insights.append(("📖 Weak Subject Focus", f"{weak.title()} needs urgent attention (score: {weak_score:.0f}). Schedule extra practice."))

    # Overall performance
    if avg >= 85:
        insights.append(("🏆 Excellent Performance", "Outstanding results! Consider mentoring peers and exploring advanced topics."))
    elif avg >= 70:
        insights.append(("👍 Good Performance", "Solid performance. Focus on weak subjects to reach excellence."))
    elif avg >= 50:
        insights.append(("🔁 Average Performance", "Average performance. Consistent effort and targeted study will help improve scores."))
    else:
        insights.append(("🚨 Below Average", "Performance needs significant improvement. Seek teacher support and create a study plan."))

    return insights

# ─────────────────────────────────────────────
# OFFLINE CHATBOT
# ─────────────────────────────────────────────
def offline_chatbot(question: str, data: pd.DataFrame) -> str:
    q = question.lower().strip()

    avg_col = "Average"
    pass_pct = (data["Pass_Fail"] == "Pass").mean() * 100
    class_avg = data[avg_col].mean()
    top_student = data.loc[data[avg_col].idxmax(), name_col]
    bottom_student = data.loc[data[avg_col].idxmin(), name_col]
    top_score = data[avg_col].max()
    bottom_score = data[avg_col].min()
    high_risk_count = len(data[data["Risk"] == "High"])
    high_risk_names = ", ".join(data[data["Risk"] == "High"][name_col].tolist())

    subject_avgs = {col: data[col].mean() for col in subject_cols}
    weakest_subject = min(subject_avgs, key=subject_avgs.get)
    strongest_subject = max(subject_avgs, key=subject_avgs.get)

    # Extract student name if mentioned
    mentioned_student = None
    for sname in data[name_col]:
        if sname.lower() in q:
            mentioned_student = sname
            break

    # ── top student
    if any(kw in q for kw in ["top student", "best student", "highest scorer", "topper", "rank 1"]):
        return (f"🏆 The top student is **{top_student}** with an average score of **{top_score:.1f}**. "
                f"They rank 1st in the class!")

    # ── weakest / lowest
    if any(kw in q for kw in ["weakest student", "lowest scorer", "worst", "bottom student", "lowest"]):
        return (f"📉 The lowest scoring student is **{bottom_student}** with an average of **{bottom_score:.1f}**. "
                f"They may need additional academic support.")

    # ── average / class average
    if any(kw in q for kw in ["class average", "average marks", "average score", "class performance", "overall average"]):
        return (f"📊 The class average score is **{class_avg:.2f}**. "
                f"Pass rate is **{pass_pct:.1f}%**. "
                f"There are **{high_risk_count}** high-risk students.")

    # ── pass percentage
    if any(kw in q for kw in ["pass", "fail", "pass rate", "pass percentage"]):
        fail_pct = 100 - pass_pct
        return (f"✅ Pass Rate: **{pass_pct:.1f}%** | ❌ Fail Rate: **{fail_pct:.1f}%**. "
                f"Students who pass have an average of 50 or above.")

    # ── subject averages
    if any(kw in q for kw in ["subject", "subject average", "subject score"]):
        lines = [f"- **{s.title()}**: {v:.1f}" for s, v in subject_avgs.items()]
        return "📚 Subject-wise averages:\n" + "\n".join(lines)

    # ── weakest subject
    if any(kw in q for kw in ["weakest subject", "worst subject", "subject to improve"]):
        return (f"📖 The weakest subject for the class is **{weakest_subject.title()}** "
                f"with an average of **{subject_avgs[weakest_subject]:.1f}**. "
                f"The strongest is **{strongest_subject.title()}** ({subject_avgs[strongest_subject]:.1f}).")

    # ── high risk
    if any(kw in q for kw in ["high risk", "at risk", "danger", "struggling", "need help"]):
        if high_risk_count > 0:
            return (f"🚨 There are **{high_risk_count}** high-risk students: {high_risk_names}. "
                    f"They need immediate academic intervention.")
        else:
            return "✅ Great news — no students are currently classified as high-risk!"

    # ── attendance
    if any(kw in q for kw in ["attendance", "absent", "present"]):
        if "attendance" in data.columns:
            avg_att = data["attendance"].mean()
            low_att = data[data["attendance"] < 75]
            return (f"📅 Average class attendance is **{avg_att:.1f}%**. "
                    f"**{len(low_att)}** students have attendance below 75% and may face academic risk.")
        return "Attendance data is not available in the dataset."

    # ── improvement tips
    if any(kw in q for kw in ["improve", "tips", "suggestion", "advice", "how to"]):
        return ("💡 General improvement tips:\n"
                "- 📚 Study at least 3 hours daily with focused sessions\n"
                "- 📅 Maintain 85%+ attendance\n"
                "- 🧘 Keep stress levels low with regular breaks\n"
                "- 📝 Complete all assignments on time\n"
                "- 🗣️ Participate actively in class discussions\n"
                "- 😴 Get 7–8 hours of sleep for optimal brain function")

    # ── specific student query
    if mentioned_student:
        s_row = data[data[name_col] == mentioned_student].iloc[0]
        return (f"👤 **{mentioned_student}**:\n"
                f"- Average: **{s_row[avg_col]:.2f}**\n"
                f"- Rank: **{s_row['Rank']}**\n"
                f"- Risk Level: **{s_row['Risk']}**\n"
                f"- Pass/Fail: **{s_row['Pass_Fail']}**\n"
                f"- Attendance: **{s_row.get('attendance', 'N/A')}%**\n"
                f"- Study Hours: **{s_row.get('study_hours', 'N/A')}h/day**")

    # ── rank query
    if any(kw in q for kw in ["rank", "ranking", "position"]):
        top5 = df_ranked.head(5)[[name_col, "Rank", avg_col, "Risk"]]
        lines = [f"- **#{int(r['Rank'])}** {r[name_col]} — {r[avg_col]:.1f}" for _, r in top5.iterrows()]
        return "🏅 Top 5 students by rank:\n" + "\n".join(lines)

    # ── trend
    if any(kw in q for kw in ["trend", "improving", "declining", "progress"]):
        if "Trend" in data.columns:
            improving = data[data["Trend"] > 0]
            declining = data[data["Trend"] < 0]
            return (f"📈 **{len(improving)}** students are improving compared to their previous score. "
                    f"📉 **{len(declining)}** students are showing a declining trend. "
                    f"Monitor declining students closely.")
        return "Trend data is not available."

    # ── comparison
    if any(kw in q for kw in ["compare", "comparison", "vs", "versus", "better"]):
        return ("🔍 You can compare multiple students in the **Insights** tab using the multi-student comparison feature. "
                "Select students from the dropdown and explore subject, attendance, and performance charts.")

    # ── default
    return (f"🤖 I can help with questions about class performance, specific students, subjects, risk levels, and more. "
            f"Try asking: 'Who is the top student?', 'What is the class average?', 'Which students are high risk?', "
            f"or mention a student's name like 'Tell me about {df[name_col].iloc[0]}'.")

# ─────────────────────────────────────────────
# GEMINI AI INTEGRATION
# ─────────────────────────────────────────────
def ask_gemini(question: str, context_summary: str, api_key: str) -> str:
    if not GEMINI_AVAILABLE:
        return "Google Generative AI package not installed. Using offline chatbot instead."
    try:
        genai.configure(api_key=api_key)
        model_g = genai.GenerativeModel("gemini-pro")
        prompt = f"""You are an expert educational AI assistant analyzing student performance data.

Dataset Summary:
{context_summary}

User Question: {question}

Provide a helpful, specific, data-driven answer. Be concise and use bullet points where appropriate."""
        response = model_g.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}. Falling back to offline analysis."

def get_dataset_context(data: pd.DataFrame) -> str:
    avg_col = "Average"
    ctx = f"""
- Total students: {len(data)}
- Class average: {data[avg_col].mean():.2f}
- Pass rate: {(data['Pass_Fail'] == 'Pass').mean() * 100:.1f}%
- High risk students: {len(data[data['Risk'] == 'High'])}
- Top student: {data.loc[data[avg_col].idxmax(), name_col]} ({data[avg_col].max():.1f})
- Weakest student: {data.loc[data[avg_col].idxmin(), name_col]} ({data[avg_col].min():.1f})
- Subject averages: {', '.join([f'{c}: {data[c].mean():.1f}' for c in subject_cols])}
"""
    if "attendance" in data.columns:
        ctx += f"- Average attendance: {data['attendance'].mean():.1f}%\n"
    if "stress_level" in data.columns:
        ctx += f"- Average stress level: {data['stress_level'].mean():.1f}/10\n"
    return ctx

# ─────────────────────────────────────────────
# CHART EXPORT HELPER (for PDF)
# ─────────────────────────────────────────────
def fig_to_bytes(fig) -> bytes | None:
    if not KALEIDO_AVAILABLE:
        return None
    try:
        return fig.to_image(format="png", width=700, height=350)
    except Exception:
        return None

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def make_pdf_overview(data: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    normal = styles["Normal"]
    content = []

    content.append(Paragraph("📊 Student Intelligence System — Overview Report", title_style))
    content.append(Spacer(1, 10))

    # Key Metrics
    content.append(Paragraph("Class Metrics", h2_style))
    pass_pct = (data["Pass_Fail"] == "Pass").mean() * 100
    metrics = [
        ["Metric", "Value"],
        ["Class Average Score", f"{data['Average'].mean():.2f}"],
        ["Top Score", f"{data['Average'].max():.2f}"],
        ["Lowest Score", f"{data['Average'].min():.2f}"],
        ["Pass Rate", f"{pass_pct:.1f}%"],
        ["High Risk Students", str(len(data[data["Risk"] == "High"]))],
        ["Total Students", str(len(data))],
    ]
    t = Table(metrics, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2196F3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0f4f8"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 16))

    # Early Warning
    content.append(Paragraph("🚨 Early Warning — Students At Risk", h2_style))
    warn_cols = ["attendance", "stress_level", "study_hours"]
    at_risk = data[data["Risk"].isin(["High", "Moderate"])]
    if len(at_risk) > 0:
        cols_to_show = [name_col, "Average", "Risk", "Pass_Fail"] + [c for c in warn_cols if c in data.columns]
        warn_data = [cols_to_show] + [[str(r[c]) if c not in ["Average"] else f"{r[c]:.2f}" for c in cols_to_show]
                                       for _, r in at_risk.iterrows()]
        wt = Table(warn_data, colWidths=[inch] * len(cols_to_show))
        wt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f44336")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#fff3f3"), rl_colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(wt)
    else:
        content.append(Paragraph("No high-risk students found.", normal))
    content.append(Spacer(1, 16))

    # Leaderboard
    content.append(Paragraph("🏆 Top 10 Leaderboard", h2_style))
    top10 = data.nsmallest(10, "Rank")[[name_col, "Rank", "Average", "Risk"]]
    lb_data = [["Rank", "Name", "Average", "Risk"]] + [
        [str(int(r["Rank"])), r[name_col], f"{r['Average']:.2f}", r["Risk"]]
        for _, r in top10.iterrows()
    ]
    lt = Table(lb_data, colWidths=[1 * inch, 2.5 * inch, 1.5 * inch, 1.5 * inch])
    lt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#4CAF50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0fff0"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(lt)
    content.append(Spacer(1, 16))

    # Pass/Fail Summary
    content.append(Paragraph("Pass / Fail Summary", h2_style))
    pf_counts = data["Pass_Fail"].value_counts()
    pf_data = [["Status", "Count", "Percentage"]] + [
        [status, str(cnt), f"{cnt / len(data) * 100:.1f}%"]
        for status, cnt in pf_counts.items()
    ]
    pft = Table(pf_data, colWidths=[2.5 * inch, 1.5 * inch, 2.5 * inch])
    pft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#FF9800")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(pft)

    doc.build(content)
    return buf.getvalue()


def make_pdf_student(row_data: pd.Series, insights: list) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    normal = styles["Normal"]
    content = []

    content.append(Paragraph(f"Student Report — {row_data[name_col]}", title_style))
    content.append(Spacer(1, 10))

    # Core metrics
    content.append(Paragraph("Performance Summary", h2_style))
    metrics = [
        ["Field", "Value"],
        ["Average Score", f"{row_data['Average']:.2f}"],
        ["Rank", f"#{row_data['Rank']}"],
        ["Risk Level", row_data["Risk"]],
        ["Pass / Fail", row_data["Pass_Fail"]],
        ["Trend vs Previous", f"{row_data.get('Trend', 0):+.2f}"],
        ["Attendance", f"{row_data.get('attendance', 'N/A')}%"],
        ["Study Hours/Day", str(row_data.get("study_hours", "N/A"))],
        ["Stress Level", str(row_data.get("stress_level", "N/A"))],
    ]
    t = Table(metrics, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#9C27B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f8f0ff"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 12))

    # Subject scores
    content.append(Paragraph("Subject-wise Scores", h2_style))
    subj_data = [["Subject", "Score"]] + [
        [sub.title(), f"{row_data[sub]:.1f}"] for sub in subject_cols
    ]
    st_tbl = Table(subj_data, colWidths=[3 * inch, 3 * inch])
    st_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2196F3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0f8ff"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(st_tbl)
    content.append(Spacer(1, 12))

    # AI Insights
    content.append(Paragraph("Smart AI Insights", h2_style))
    for title, text in insights:
        content.append(Paragraph(f"<b>{title}</b>: {text}", normal))
        content.append(Spacer(1, 4))

    doc.build(content)
    return buf.getvalue()


def make_pdf_teacher(data: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    normal = styles["Normal"]
    content = []

    content.append(Paragraph("Teacher Dashboard Report", title_style))
    content.append(Spacer(1, 10))

    pass_pct = (data["Pass_Fail"] == "Pass").mean() * 100

    # Summary metrics
    content.append(Paragraph("Class Summary", h2_style))
    summary = [
        ["Metric", "Value"],
        ["Class Average", f"{data['Average'].mean():.2f}"],
        ["Pass Rate", f"{pass_pct:.1f}%"],
        ["High Risk Count", str(len(data[data["Risk"] == "High"]))],
    ]
    if "attendance" in data.columns:
        summary.append(["Avg Attendance", f"{data['attendance'].mean():.1f}%"])
    if "study_hours" in data.columns:
        summary.append(["Avg Study Hours", f"{data['study_hours'].mean():.1f}h/day"])
    t = Table(summary, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#009688")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e8f5e9"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 14))

    # High risk students
    content.append(Paragraph("High-Risk Students", h2_style))
    hr = data[data["Risk"] == "High"][[name_col, "Average", "Pass_Fail"]]
    if len(hr) > 0:
        hr_data = [["Name", "Average", "Pass/Fail"]] + [
            [r[name_col], f"{r['Average']:.2f}", r["Pass_Fail"]] for _, r in hr.iterrows()
        ]
        ht = Table(hr_data, colWidths=[2.5 * inch, 2 * inch, 2 * inch])
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f44336")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(ht)
    else:
        content.append(Paragraph("No high-risk students.", normal))
    content.append(Spacer(1, 14))

    # Top performers
    content.append(Paragraph("Top Performers", h2_style))
    top = data[data["Risk"] == "Low"].nsmallest(5, "Rank")[[name_col, "Average", "Rank"]]
    tp_data = [["Name", "Average", "Rank"]] + [
        [r[name_col], f"{r['Average']:.2f}", f"#{int(r['Rank'])}"] for _, r in top.iterrows()
    ]
    tt = Table(tp_data, colWidths=[2.5 * inch, 2 * inch, 2 * inch])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#4CAF50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(tt)
    content.append(Spacer(1, 14))

    # Subject analysis
    content.append(Paragraph("Subject-wise Class Averages", h2_style))
    subj_avgs = [(sub.title(), f"{data[sub].mean():.2f}") for sub in subject_cols]
    sa_data = [["Subject", "Class Average"]] + list(subj_avgs)
    sat = Table(sa_data, colWidths=[3.5 * inch, 3 * inch])
    sat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#FF9800")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#fff8e1"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(sat)

    doc.build(content)
    return buf.getvalue()


def make_pdf_insights(data: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    normal = styles["Normal"]
    content = []

    content.append(Paragraph("Insights & Trend Report", title_style))
    content.append(Spacer(1, 10))

    # Trend summary
    content.append(Paragraph("Performance Trend Summary", h2_style))
    if "Trend" in data.columns:
        imp = data[data["Trend"] > 0]
        dec = data[data["Trend"] < 0]
        content.append(Paragraph(
            f"Students improving: {len(imp)} | Students declining: {len(dec)}", normal
        ))
        content.append(Spacer(1, 6))
        trend_data = [[name_col.title(), "Average", "Trend", "Risk"]] + [
            [r[name_col], f"{r['Average']:.2f}", f"{r['Trend']:+.2f}", r["Risk"]]
            for _, r in data.sort_values("Trend", ascending=False).iterrows()
        ]
        tdt = Table(trend_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        tdt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3F51B5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e8eaf6"), rl_colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(tdt)
        content.append(Spacer(1, 14))

    # Risk distribution
    content.append(Paragraph("Risk Distribution", h2_style))
    risk_dist = data["Risk"].value_counts().reset_index()
    risk_dist.columns = ["Risk Level", "Count"]
    rd_data = [["Risk Level", "Count", "Percentage"]] + [
        [r["Risk Level"], str(r["Count"]), f"{r['Count'] / len(data) * 100:.1f}%"]
        for _, r in risk_dist.iterrows()
    ]
    rdt = Table(rd_data, colWidths=[2.5 * inch, 1.5 * inch, 2.5 * inch])
    rdt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E91E63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(rdt)

    doc.build(content)
    return buf.getvalue()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "👤 Student Analysis",
    "📈 Insights",
    "👩‍🏫 Teacher Dashboard",
    "📂 Dataset",
    "🤖 AI Chatbot",
])
# ═══════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Class Dashboard")

    pass_pct = (df["Pass_Fail"] == "Pass").mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Score", round(df["Average"].mean(), 2))
    col2.metric("Top Score", round(df["Average"].max(), 2))
    col3.metric("Lowest Score", round(df["Average"].min(), 2))
    col4.metric("High Risk", len(df[df["Risk"] == "High"]))
    col5.metric("Pass Rate", f"{pass_pct:.1f}%")

    st.markdown("---")

    # ═════════════════════════════════════
    # DATASET QUALITY REPORT
    # ═════════════════════════════════════
    st.subheader("🧹 Dataset Quality Report")

    q1, q2, q3 = st.columns(3)

    q1.metric("Missing Values", int(total_missing))
    q2.metric("Duplicate Rows Removed", int(duplicate_count))
    q3.metric("Final Dataset Size", f"{df.shape[0]} rows")

    # Missing values details
    if total_missing > 0:
        st.warning("⚠️ Missing values detected and handled automatically.")

        missing_df = pd.DataFrame({
            "Column": missing_info.index,
            "Missing Values": missing_info.values
        })

        st.dataframe(missing_df, use_container_width=True)

    else:
        st.success("✅ No missing values found in dataset.")

    # Duplicate rows details
    if duplicate_count > 0:
        st.info(f"ℹ️ {duplicate_count} duplicate rows were removed automatically.")
    else:
        st.success("✅ No duplicate rows found.")

    st.markdown("---")

    # ═════════════════════════════════════
    # CHARTS
    # ═════════════════════════════════════
    colA, colB = st.columns(2)

    with colA:
        fig_hist = px.histogram(
            df,
            x="Average",
            nbins=10,
            color_discrete_sequence=["#2196F3"],
            title="Score Distribution"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with colB:
        fig_pf = px.pie(
            df,
            names="Pass_Fail",
            title="Pass / Fail Distribution",
            color_discrete_map={
                "Pass": "#4CAF50",
                "Fail": "#f44336"
            }
        )
        st.plotly_chart(fig_pf, use_container_width=True)

    colC, colD = st.columns(2)

    with colC:
        fig_risk = px.pie(
            df,
            names="Risk",
            title="Risk Level Distribution",
            color_discrete_map={
                "Low": "#4CAF50",
                "Moderate": "#FF9800",
                "High": "#f44336"
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with colD:
        subj_avg_vals = {
            col.title(): df[col].mean()
            for col in subject_cols
        }

        fig_subj = px.bar(
            x=list(subj_avg_vals.keys()),
            y=list(subj_avg_vals.values()),
            title="Subject-wise Class Averages",
            labels={"x": "Subject", "y": "Average"},
            color_discrete_sequence=["#9C27B0"]
        )

        st.plotly_chart(fig_subj, use_container_width=True)

    st.markdown("---")

    # ═════════════════════════════════════
    # EARLY WARNING SYSTEM
    # ═════════════════════════════════════
    st.subheader("🚨 Early Warning System")

    warn_conditions = {
        "Low Attendance (<75%)":
            df["attendance"] < 75
            if "attendance" in df.columns
            else pd.Series([False] * len(df)),

        "Low Average (<50)":
            df["Average"] < 50,

        "High Stress (>7)":
            df["stress_level"] > 7
            if "stress_level" in df.columns
            else pd.Series([False] * len(df)),

        "Low Study Hours (<1.5h)":
            df["study_hours"] < 1.5
            if "study_hours" in df.columns
            else pd.Series([False] * len(df)),
    }

    warn_cols = st.columns(len(warn_conditions))

    for i, (label, mask) in enumerate(warn_conditions.items()):
        count = mask.sum()
        warn_cols[i].metric(label, int(count))

    at_risk_students = df[df["Risk"].isin(["High", "Moderate"])]

    if len(at_risk_students) > 0:

        st.warning(
            f"⚠️ {len(at_risk_students)} students require attention"
        )

        risk_display_cols = [
            name_col,
            "Average",
            "Risk",
            "Pass_Fail"
        ] + [
            c for c in [
                "attendance",
                "study_hours",
                "stress_level"
            ]
            if c in df.columns
        ]

        st.dataframe(
            at_risk_students[risk_display_cols]
            .sort_values("Average"),
            use_container_width=True
        )

    else:
        st.success("✅ No students in the warning zone.")

    st.markdown("---")

    # ═════════════════════════════════════
    # LEADERBOARD
    # ═════════════════════════════════════
    st.subheader("🏆 Top 10 Leaderboard")

    top10 = df_ranked.head(10)[
        [name_col, "Rank", "Average", "Risk", "Pass_Fail"]
    ]

    top_performer = top10.iloc[0]

    st.success(
        f"🥇 Top Performer: "
        f"**{top_performer[name_col]}** — "
        f"Average: **{top_performer['Average']:.2f}**"
    )

    if len(top10) > 1:
        second = top10.iloc[1]

        st.info(
            f"🥈 Runner-up: "
            f"**{second[name_col]}** — "
            f"Average: **{second['Average']:.2f}**"
        )

    st.dataframe(
        top10.reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")

    # ═════════════════════════════════════
    # PDF DOWNLOAD
    # ═════════════════════════════════════
    pdf_ov = make_pdf_overview(df)

    st.download_button(
        "📄 Download Overview Report (PDF)",
        data=pdf_ov,
        file_name="overview_report.pdf",
        mime="application/pdf",
    )


# TAB 2 — STUDENT ANALYSIS
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("👤 Student Analysis")

    student = st.selectbox("Select Student", df[name_col].tolist())
    row = df[df[name_col] == student].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average", round(row["Average"], 2))
    col2.metric("Trend vs Previous", f"{row.get('Trend', 0):+.2f}")
    col3.metric("Pass / Fail", row["Pass_Fail"])
    col4.metric("Rank", f"#{int(row['Rank'])}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Risk Level", row["Risk"])
    col6.metric("Attendance", f"{row.get('attendance', 'N/A')}%")
    col7.metric("Stress Level", row.get("stress_level", "N/A"))

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig_bar = px.bar(
            x=[c.title() for c in subject_cols],
            y=[row[c] for c in subject_cols],
            title=f"{student}'s Subject Scores",
            labels={"x": "Subject", "y": "Score"},
            color_discrete_sequence=["#2196F3"],
        )
        fig_bar.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Pass Line")
        st.plotly_chart(fig_bar, use_container_width=True)

    with colB:
        # Radar chart
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in subject_cols] + [row[subject_cols[0]]],
            theta=[c.title() for c in subject_cols] + [subject_cols[0].title()],
            fill="toself",
            name=student,
            line_color="#9C27B0",
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"Performance Radar — {student}",
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # Progress bars
    st.subheader("📊 Subject Progress")
    for sub in subject_cols:
        score = int(row[sub])
        color = "normal" if score >= 50 else "inverse"
        st.write(f"**{sub.title()}** — {score}/100")
        st.progress(score / 100)

    weak = min(subject_cols, key=lambda x: row[x])
    strong = max(subject_cols, key=lambda x: row[x])
    st.error(f"📌 Weakest Subject: **{weak.title()}** ({row[weak]:.0f}/100)")
    st.success(f"⭐ Strongest Subject: **{strong.title()}** ({row[strong]:.0f}/100)")

    pred = model.predict([[row[col] for col in features]])
    st.info(f"🔮 Predicted Next Score: **{round(pred[0], 2)}**")

    st.markdown("---")

    # ── Smart AI Insights
    st.subheader("💡 Smart AI Insights")
    insights = generate_smart_insights(row)
    for title, text in insights:
        if "⚠️" in title or "📉" in title or "🔴" in title or "🚨" in title:
            st.warning(f"**{title}**: {text}")
        elif "✅" in title or "🏆" in title or "📈" in title or "👍" in title:
            st.success(f"**{title}**: {text}")
        else:
            st.info(f"**{title}**: {text}")

    st.markdown("---")
    pdf_stu = make_pdf_student(row, insights)
    st.download_button(
        "📄 Download Student Report (PDF)",
        data=pdf_stu,
        file_name=f"student_report_{student.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

# ═══════════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Class Insights & Trend Analysis")

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).drop(columns=["Rank"], errors="ignore").corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                          title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Subject-wise line chart
    st.markdown("#### Subject Score Comparison Across Students")
    fig_line = px.line(df, x=name_col, y=subject_cols, markers=True,
                       title="Subject Scores by Student")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # Trend analysis
    st.markdown("#### Performance Trend (vs Previous Score)")
    if "Trend" in df.columns and "previous_score" in df.columns:
        fig_trend = px.scatter(
            df, x="previous_score", y="Average", color="Risk", text=name_col,
            title="Previous Score vs Current Average",
            color_discrete_map={"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#f44336"},
        )
        fig_trend.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                             line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig_trend, use_container_width=True)

    # Attendance vs score
    if "attendance" in df.columns:
        st.markdown("#### Attendance vs Average Score")
        fig_att = px.scatter(
            df, x="attendance", y="Average", color="Risk", text=name_col,
            title="Attendance vs Performance",
            color_discrete_map={"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#f44336"},
        )
        st.plotly_chart(fig_att, use_container_width=True)

    # Stress vs performance
    if "stress_level" in df.columns:
        st.markdown("#### Stress Level vs Performance")
        fig_stress = px.scatter(
            df, x="stress_level", y="Average", color="Pass_Fail", text=name_col,
            title="Stress vs Performance",
            color_discrete_map={"Pass": "#4CAF50", "Fail": "#f44336"},
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("---")

    # ── Multi-Student Comparison
    st.subheader("🔀 Multi-Student Comparison")
    all_students = df[name_col].tolist()
    selected_students = st.multiselect(
        "Select 2+ students to compare",
        all_students,
        default=all_students[:min(3, len(all_students))],
    )

    if len(selected_students) >= 2:
        cdf = df[df[name_col].isin(selected_students)]

        # Grouped bar chart
        fig_compare = go.Figure()
        for _, srow in cdf.iterrows():
            fig_compare.add_trace(go.Bar(
                name=srow[name_col],
                x=[c.title() for c in subject_cols],
                y=[srow[c] for c in subject_cols],
            ))
        fig_compare.update_layout(barmode="group", title="Subject Score Comparison",
                                   xaxis_title="Subject", yaxis_title="Score")
        st.plotly_chart(fig_compare, use_container_width=True)

        # Radar comparison
        fig_radar_cmp = go.Figure()
        for _, srow in cdf.iterrows():
            fig_radar_cmp.add_trace(go.Scatterpolar(
                r=[srow[c] for c in subject_cols] + [srow[subject_cols[0]]],
                theta=[c.title() for c in subject_cols] + [subject_cols[0].title()],
                fill="toself",
                name=srow[name_col],
            ))
        fig_radar_cmp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Multi-Student Radar Comparison",
        )
        st.plotly_chart(fig_radar_cmp, use_container_width=True)

        # Additional attributes comparison
        compare_attrs = ["Average", "attendance", "study_hours", "stress_level"]
        valid_attrs = [a for a in compare_attrs if a in cdf.columns]
        fig_attr = px.bar(
            cdf, x=name_col, y=valid_attrs, barmode="group",
            title="Additional Attributes Comparison"
        )
        st.plotly_chart(fig_attr, use_container_width=True)

    elif len(selected_students) == 1:
        st.info("Please select at least 2 students for comparison.")

    st.markdown("---")
    # Risk filter
    st.markdown("#### Filter by Risk Level")
    risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Moderate", "Low"])
    filtered = df if risk_filter == "All" else df[df["Risk"] == risk_filter]
    st.dataframe(filtered, use_container_width=True)

    st.markdown("---")
    pdf_ins = make_pdf_insights(df)
    st.download_button(
        "📄 Download Insights Report (PDF)",
        data=pdf_ins,
        file_name="insights_report.pdf",
        mime="application/pdf",
    )

# ═══════════════════════════════════════════════════
# TAB 4 — TEACHER DASHBOARD
# ═══════════════════════════════════════════════════
with tab4:
    st.subheader("👩‍🏫 Teacher Dashboard")

    pass_pct_t = (df["Pass_Fail"] == "Pass").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class Average", round(df["Average"].mean(), 2))
    c2.metric("Pass Rate", f"{pass_pct_t:.1f}%")
    if "attendance" in df.columns:
        c3.metric("Avg Attendance", f"{df['attendance'].mean():.1f}%")
    if "study_hours" in df.columns:
        c4.metric("Avg Study Hours", f"{df['study_hours'].mean():.1f}h")

    st.markdown("---")

    # High risk table
    st.markdown("#### 🚨 High-Risk Students")
    hr_students = df[df["Risk"] == "High"]
    if len(hr_students) > 0:
        st.dataframe(hr_students[[name_col, "Average", "Pass_Fail", "Risk"] +
                                  [c for c in ["attendance", "study_hours", "stress_level"] if c in df.columns]],
                     use_container_width=True)
    else:
        st.success("✅ No high-risk students currently.")

    st.markdown("---")

    # Top performers
    st.markdown("#### ⭐ Top Performers")
    top_perf = df[df["Risk"] == "Low"].nsmallest(5, "Rank")
    if len(top_perf) > 0:
        st.dataframe(top_perf[[name_col, "Average", "Rank", "Pass_Fail"]], use_container_width=True)
    else:
        st.info("No low-risk (top) performers found.")

    st.markdown("---")

    # Weakest subject
    subj_avgs = {col: df[col].mean() for col in subject_cols}
    weakest_subj = min(subj_avgs, key=subj_avgs.get)
    strongest_subj = max(subj_avgs, key=subj_avgs.get)
    st.info(f"📖 Weakest Subject (class): **{weakest_subj.title()}** — Avg: {subj_avgs[weakest_subj]:.1f}")
    st.success(f"⭐ Strongest Subject (class): **{strongest_subj.title()}** — Avg: {subj_avgs[strongest_subj]:.1f}")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig_subj_t = px.bar(
            x=[s.title() for s in subj_avgs.keys()],
            y=list(subj_avgs.values()),
            title="Subject-wise Class Average",
            labels={"x": "Subject", "y": "Average"},
            color_discrete_sequence=["#009688"],
        )
        fig_subj_t.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Pass Line")
        st.plotly_chart(fig_subj_t, use_container_width=True)

    with colB:
        fig_risk_t = px.pie(
            df, names="Risk", title="Risk Distribution",
            color_discrete_map={"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#f44336"}
        )
        st.plotly_chart(fig_risk_t, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig_pf_t = px.pie(
            df, names="Pass_Fail", title="Pass / Fail Distribution",
            color_discrete_map={"Pass": "#4CAF50", "Fail": "#f44336"}
        )
        st.plotly_chart(fig_pf_t, use_container_width=True)

    with colD:
        if "attendance" in df.columns:
            fig_att_t = px.histogram(
                df, x="attendance", nbins=10, color_discrete_sequence=["#2196F3"],
                title="Attendance Distribution"
            )
            st.plotly_chart(fig_att_t, use_container_width=True)

    # Box plots per subject
    st.markdown("#### Subject Score Distribution (Box Plot)")
    fig_box = go.Figure()
    for sub in subject_cols:
        fig_box.add_trace(go.Box(y=df[sub], name=sub.title()))
    fig_box.update_layout(title="Score Distribution per Subject", yaxis_title="Score")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    pdf_tc = make_pdf_teacher(df)

st.download_button(
    label="📄 Download Teacher Dashboard Report (PDF)",
    data=pdf_tc,
    file_name="teacher_dashboard_report.pdf",
    mime="application/pdf",
    key="teacher_pdf_download"
    title="Download Teacher Dashboard"
)
# ═══════════════════════════════════════════════════
# TAB 5 — DATASET
# ═══════════════════════════════════════════════════
with tab5:
    st.subheader("📂 Dataset View")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Export full dataset
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Processed Dataset (CSV)",
        data=csv_export,
        file_name="processed_students.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════
# TAB 6 — AI CHATBOT
# ═══════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 AI Chatbot — Ask About Your Students")

    use_gemini = bool(gemini_api_key and GEMINI_AVAILABLE)

    if use_gemini:
        st.success("🟢 Gemini AI active — powered by Google Gemini Pro")
    else:
        st.info("🔵 Offline mode — smart offline chatbot active. Enter a Gemini API key in the sidebar for AI-enhanced responses.")

    # ── Ask Gemini section
    if use_gemini:
        st.markdown("#### 🌟 Ask Gemini AI")
        gemini_q = st.text_input("Ask Gemini anything about the dataset or education:")
        if gemini_q:
            with st.spinner("Gemini is thinking..."):
                context = get_dataset_context(df)
                gemini_resp = ask_gemini(gemini_q, context, gemini_api_key)
            if "error" in gemini_resp.lower() or "gemini error" in gemini_resp.lower():
                st.warning(gemini_resp)
                st.markdown("**Offline fallback response:**")
                st.write(offline_chatbot(gemini_q, df))
            else:
                st.markdown(gemini_resp)

        st.markdown("---")

    # ── Offline chatbot
    st.markdown("#### 💬 Offline Smart Chatbot")
    st.caption("Powered by rule-based analysis of your dataset")

    sample_questions = [
        "Who is the top student?",
        "What is the class average?",
        "Which students are high risk?",
        "What is the pass percentage?",
        "Which subject is the weakest?",
        "Give me improvement tips",
        "What is the attendance situation?",
    ]

    selected_q = st.selectbox("💡 Sample questions (or type your own below):", [""] + sample_questions)
    user_q = st.text_input("Ask a question about the dataset:", value=selected_q)

    if user_q:
        with st.spinner("Analyzing..."):
            response = offline_chatbot(user_q, df)
        st.markdown("**Answer:**")
        st.markdown(response)

    st.markdown("---")
    st.markdown("**You can ask about:**")
    st.markdown(
        "- Top/weakest student\n"
        "- Class average, pass rate\n"
        "- Subject averages & weakest subject\n"
        "- High-risk students\n"
        "- Attendance insights\n"
        "- Improvement tips\n"
        "- Specific student details (e.g. 'Tell me about Aarav')\n"
        "- Rankings and trends"
    )
