# =============================================================================
# STUDENT PERFORMANCE INTELLIGENCE SYSTEM
# =============================================================================
# Features: Pass/Fail, Leaderboard, Teacher Dashboard, Smart Insights,
#           Offline Chatbot, Gemini AI, Early Warning, Trend Analysis,
#           Multi-Student Comparison, PDF Export, UI Improvements,
#           Goal Tracking with personalized study plans
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
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.loc[:, ~df.columns.duplicated()]

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
name_col = [col for col in df.columns if "name" in col][0]

df = df.fillna(df.mean(numeric_only=True))

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Overview",
    "👤 Student Analysis",
    "📈 Insights",
    "👩‍🏫 Teacher Dashboard",
    "📂 Dataset",
    "🤖 AI Chatbot",
    "🎯 Goal Tracking",
    "📡 Attendance Forecast",
    "📜 Report Cards",
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

    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(
            df, x="Average", nbins=10, color_discrete_sequence=["#2196F3"],
            title="Score Distribution"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        fig_pf = px.pie(
            df, names="Pass_Fail", title="Pass / Fail Distribution",
            color_discrete_map={"Pass": "#4CAF50", "Fail": "#f44336"}
        )
        st.plotly_chart(fig_pf, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig_risk = px.pie(df, names="Risk", title="Risk Level Distribution",
                          color_discrete_map={"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#f44336"})
        st.plotly_chart(fig_risk, use_container_width=True)
    with colD:
        subj_avg_vals = {col.title(): df[col].mean() for col in subject_cols}
        fig_subj = px.bar(
            x=list(subj_avg_vals.keys()), y=list(subj_avg_vals.values()),
            title="Subject-wise Class Averages", labels={"x": "Subject", "y": "Average"},
            color_discrete_sequence=["#9C27B0"]
        )
        st.plotly_chart(fig_subj, use_container_width=True)

    st.markdown("---")

    # ── Early Warning System
    st.subheader("🚨 Early Warning System")

    warn_conditions = {
        "Low Attendance (<75%)": df["attendance"] < 75 if "attendance" in df.columns else pd.Series([False] * len(df)),
        "Low Average (<50)": df["Average"] < 50,
        "High Stress (>7)": df["stress_level"] > 7 if "stress_level" in df.columns else pd.Series([False] * len(df)),
        "Low Study Hours (<1.5h)": df["study_hours"] < 1.5 if "study_hours" in df.columns else pd.Series([False] * len(df)),
    }

    warn_cols = st.columns(len(warn_conditions))
    for i, (label, mask) in enumerate(warn_conditions.items()):
        count = mask.sum()
        warn_cols[i].metric(label, int(count), delta=None)

    at_risk_students = df[df["Risk"].isin(["High", "Moderate"])]
    if len(at_risk_students) > 0:
        st.warning(f"⚠️ {len(at_risk_students)} students require attention")
        risk_display_cols = [name_col, "Average", "Risk", "Pass_Fail"] + [
            c for c in ["attendance", "study_hours", "stress_level"] if c in df.columns
        ]
        st.dataframe(at_risk_students[risk_display_cols].sort_values("Average"), use_container_width=True)
    else:
        st.success("✅ No students in the warning zone.")

    st.markdown("---")

    # ── Leaderboard
    st.subheader("🏆 Top 10 Leaderboard")
    top10 = df_ranked.head(10)[[name_col, "Rank", "Average", "Risk", "Pass_Fail"]]

    top_performer = top10.iloc[0]
    st.success(f"🥇 Top Performer: **{top_performer[name_col]}** — Average: **{top_performer['Average']:.2f}**")

    if len(top10) > 1:
        second = top10.iloc[1]
        st.info(f"🥈 Runner-up: **{second[name_col]}** — Average: **{second['Average']:.2f}**")

    st.dataframe(top10.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    pdf_ov = make_pdf_overview(df)
    st.download_button(
        "📄 Download Overview Report (PDF)",
        data=pdf_ov,
        file_name="overview_report.pdf",
        mime="application/pdf",
    )

# ═══════════════════════════════════════════════════
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
        "📄 Download Teacher Dashboard Report (PDF)",
        data=pdf_tc,
        file_name="teacher_dashboard_report.pdf",
        mime="application/pdf",
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

# ═══════════════════════════════════════════════════
# TAB 7 — GOAL TRACKING
# ═══════════════════════════════════════════════════
with tab7:
    st.subheader("🎯 Goal Tracking & Study Plans")
    st.caption("Set target scores per subject for each student and track progress toward those goals.")

    # ── Session state for persisting goals
    if "goals" not in st.session_state:
        st.session_state["goals"] = {}

    # ── Mode selector
    mode = st.radio(
        "View mode",
        ["Individual Student", "Class Overview"],
        horizontal=True,
    )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # INDIVIDUAL STUDENT MODE
    # ─────────────────────────────────────────────
    if mode == "Individual Student":
        goal_student = st.selectbox("Select Student", df[name_col].tolist(), key="goal_student_select")
        s_row = df[df[name_col] == goal_student].iloc[0]

        st.markdown(f"#### Set Target Scores for **{goal_student}**")
        st.caption("Adjust the sliders to set the target score for each subject. Current scores are shown alongside.")

        # Retrieve saved goals or default to current score + 10 (capped at 100)
        saved = st.session_state["goals"].get(goal_student, {})

        targets = {}
        cols_per_row = 3
        subject_chunks = [subject_cols[i:i+cols_per_row] for i in range(0, len(subject_cols), cols_per_row)]

        for chunk in subject_chunks:
            row_cols = st.columns(len(chunk))
            for col_ui, sub in zip(row_cols, chunk):
                current_score = float(s_row[sub])
                default_target = min(100, int(current_score) + 10)
                saved_target = saved.get(sub, default_target)
                with col_ui:
                    targets[sub] = st.slider(
                        f"{sub.title()}",
                        min_value=0,
                        max_value=100,
                        value=int(saved_target),
                        key=f"goal_{goal_student}_{sub}",
                        help=f"Current: {current_score:.0f}",
                    )

        if st.button("💾 Save Goals", key="save_goals_btn"):
            st.session_state["goals"][goal_student] = targets
            st.success(f"Goals saved for {goal_student}!")

        # Use the in-progress slider values for display
        st.session_state["goals"][goal_student] = targets

        st.markdown("---")

        # ── Gap analysis
        st.markdown(f"#### 📊 Progress Toward Goals — {goal_student}")

        gap_data = []
        for sub in subject_cols:
            current = float(s_row[sub])
            target = float(targets[sub])
            gap = target - current
            pct = min(100, (current / target * 100)) if target > 0 else 100
            status = "✅ Achieved" if current >= target else "⏳ In Progress"
            gap_data.append({
                "Subject": sub.title(),
                "Current": current,
                "Target": target,
                "Gap": gap,
                "Progress %": round(pct, 1),
                "Status": status,
            })

        gap_df = pd.DataFrame(gap_data)

        # Bullet / progress chart
        fig_bullet = go.Figure()
        for _, grow in gap_df.iterrows():
            color = "#4CAF50" if grow["Current"] >= grow["Target"] else "#2196F3"
            fig_bullet.add_trace(go.Bar(
                name=grow["Subject"],
                x=[grow["Subject"]],
                y=[grow["Current"]],
                marker_color=color,
                text=f"{grow['Current']:.0f}",
                textposition="outside",
            ))
            fig_bullet.add_trace(go.Scatter(
                x=[grow["Subject"]],
                y=[grow["Target"]],
                mode="markers",
                marker=dict(symbol="line-ew", size=20, color="#f44336",
                            line=dict(width=3, color="#f44336")),
                name=f"{grow['Subject']} Target",
                showlegend=False,
            ))

        fig_bullet.update_layout(
            title=f"Current Score vs Target — {goal_student}",
            yaxis=dict(range=[0, 105], title="Score"),
            xaxis_title="Subject",
            barmode="group",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_bullet, use_container_width=True)

        # Progress percentage bar chart
        fig_pct = px.bar(
            gap_df,
            x="Subject",
            y="Progress %",
            color="Progress %",
            color_continuous_scale=["#f44336", "#FF9800", "#4CAF50"],
            range_color=[0, 100],
            title=f"Goal Completion Rate — {goal_student}",
            text="Progress %",
        )
        fig_pct.add_hline(y=100, line_dash="dash", line_color="green",
                           annotation_text="Goal Reached")
        fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pct.update_layout(yaxis=dict(range=[0, 115]), coloraxis_showscale=False)
        st.plotly_chart(fig_pct, use_container_width=True)

        # Gap table with status
        st.markdown("#### Gap Summary Table")
        display_gap = gap_df.copy()
        display_gap["Gap"] = display_gap["Gap"].apply(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
        st.dataframe(display_gap, use_container_width=True)

        st.markdown("---")

        # ── Personalized study plan
        st.markdown(f"#### 📋 Personalized Study Plan — {goal_student}")

        critical = gap_df[gap_df["Gap"].apply(lambda x: float(str(x).replace("+","")) if isinstance(x, str) else x) > 15].sort_values("Progress %")
        moderate = gap_df[(gap_df["Gap"].apply(lambda x: float(str(x).replace("+","")) if isinstance(x, str) else x) > 0) &
                          (gap_df["Gap"].apply(lambda x: float(str(x).replace("+","")) if isinstance(x, str) else x) <= 15)].sort_values("Progress %")
        achieved = gap_df[gap_df["Status"] == "✅ Achieved"]

        raw_gap_df = pd.DataFrame(gap_data)

        if len(raw_gap_df[raw_gap_df["Gap"] > 15]) > 0:
            st.error("🔴 **Critical Focus Areas** — more than 15 points below target")
            for _, g in raw_gap_df[raw_gap_df["Gap"] > 15].sort_values("Gap", ascending=False).iterrows():
                with st.expander(f"📖 {g['Subject']} — Gap: {g['Gap']:.0f} points (Current: {g['Current']:.0f} → Target: {g['Target']:.0f})"):
                    weeks_needed = max(1, int(g["Gap"] / 5))
                    st.markdown(f"""
**Study Plan for {g['Subject'].title()}:**
- **Daily practice:** Dedicate at least **45–60 minutes/day** to {g['Subject'].title()} exercises
- **Weekly target:** Aim to close **5 points of the gap per week** (~{weeks_needed} week{'s' if weeks_needed > 1 else ''} to reach goal)
- **Resources:** Review textbook chapters, attempt past exam papers, and seek teacher help for difficult topics
- **Technique:** Use active recall — solve problems without looking at notes first, then check answers
- **Track progress:** Re-test yourself every week on {g['Subject'].title()} topics to measure improvement
- **Study group:** Consider forming a study group focused on {g['Subject'].title()} with higher-performing peers
                    """)

        if len(raw_gap_df[(raw_gap_df["Gap"] > 0) & (raw_gap_df["Gap"] <= 15)]) > 0:
            st.warning("🟠 **Moderate Improvement Needed** — within 15 points of target")
            for _, g in raw_gap_df[(raw_gap_df["Gap"] > 0) & (raw_gap_df["Gap"] <= 15)].sort_values("Gap", ascending=False).iterrows():
                with st.expander(f"📗 {g['Subject']} — Gap: {g['Gap']:.0f} points (Current: {g['Current']:.0f} → Target: {g['Target']:.0f})"):
                    weeks_needed = max(1, int(g["Gap"] / 5))
                    st.markdown(f"""
**Study Plan for {g['Subject'].title()}:**
- **Daily practice:** 30 minutes of focused revision on {g['Subject'].title()}
- **Weekly target:** Close **5 points of the gap per week** (~{weeks_needed} week{'s' if weeks_needed > 1 else ''} to reach goal)
- **Technique:** Focus on weak sub-topics within {g['Subject'].title()} — use flashcards or summary notes
- **Assignments:** Complete all assigned practice problems and review any graded feedback carefully
- **Consistency:** Short, regular sessions are more effective than occasional long cramming sessions
                    """)

        if len(raw_gap_df[raw_gap_df["Gap"] <= 0]) > 0:
            st.success("✅ **Goals Already Achieved** — maintain and stretch further")
            achieved_subjects = ", ".join(raw_gap_df[raw_gap_df["Gap"] <= 0]["Subject"].tolist())
            st.markdown(f"Great work on: **{achieved_subjects}**! Consider raising the target to keep challenging yourself.")

        # Overall recommendation
        st.markdown("---")
        st.markdown("#### 🧠 Overall Recommendation")
        total_gap = raw_gap_df["Gap"].clip(lower=0).sum()
        subjects_behind = len(raw_gap_df[raw_gap_df["Gap"] > 0])
        avg_progress = raw_gap_df["Progress %"].mean()

        if avg_progress >= 95:
            st.success(f"🏆 **{goal_student}** is performing exceptionally close to all targets (avg goal completion: {avg_progress:.1f}%). Consider raising the targets to maintain challenge and growth.")
        elif avg_progress >= 80:
            st.info(f"👍 **{goal_student}** is making strong progress (avg goal completion: {avg_progress:.1f}%). Focus on the {subjects_behind} subject(s) still below target. Total gap to close: {total_gap:.0f} points.")
        elif avg_progress >= 60:
            st.warning(f"📚 **{goal_student}** needs consistent effort (avg goal completion: {avg_progress:.1f}%). {subjects_behind} subjects are below target with a total gap of {total_gap:.0f} points. A structured daily study plan is recommended.")
        else:
            st.error(f"🚨 **{goal_student}** is significantly below targets (avg goal completion: {avg_progress:.1f}%). Immediate intervention is needed — consider one-on-one teacher sessions, reduced screen time, and a strict study schedule.")

    # ─────────────────────────────────────────────
    # CLASS OVERVIEW MODE
    # ─────────────────────────────────────────────
    else:
        st.markdown("#### 🏫 Class-wide Goal Overview")
        st.info("Set a single class-wide target for each subject, then see how every student measures up.")

        # Class-wide target sliders
        st.markdown("**Set Class Targets per Subject:**")
        class_targets = {}
        cols_per_row = 3
        subject_chunks = [subject_cols[i:i+cols_per_row] for i in range(0, len(subject_cols), cols_per_row)]
        for chunk in subject_chunks:
            row_cols = st.columns(len(chunk))
            for col_ui, sub in zip(row_cols, chunk):
                default_t = int(df[sub].mean()) + 10
                with col_ui:
                    class_targets[sub] = st.slider(
                        f"{sub.title()} target",
                        min_value=0,
                        max_value=100,
                        value=min(100, default_t),
                        key=f"class_goal_{sub}",
                    )

        st.markdown("---")

        # Build progress matrix for all students
        progress_rows = []
        for _, srow in df.iterrows():
            sname = srow[name_col]
            pcts = []
            for sub in subject_cols:
                t = class_targets[sub]
                pct = min(100, (srow[sub] / t * 100)) if t > 0 else 100
                pcts.append(round(pct, 1))
            avg_pct = round(sum(pcts) / len(pcts), 1)
            subjects_met = sum(1 for p in pcts if p >= 100)
            progress_rows.append({
                "Student": sname,
                **{f"{sub.title()} %": pct for sub, pct in zip(subject_cols, pcts)},
                "Avg Goal %": avg_pct,
                "Subjects Met": f"{subjects_met}/{len(subject_cols)}",
            })

        prog_df = pd.DataFrame(progress_rows).sort_values("Avg Goal %", ascending=False)

        # Heatmap of goal completion
        heat_data = prog_df[["Student"] + [f"{s.title()} %" for s in subject_cols]].set_index("Student")
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale=["#f44336", "#FF9800", "#4CAF50"],
            range_color=[0, 100],
            title="Goal Completion Heatmap (% of Target Reached per Subject)",
            text_auto=True,
            aspect="auto",
        )
        fig_heat.update_layout(coloraxis_colorbar=dict(title="% of Target"))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Avg goal completion bar chart
        fig_avg = px.bar(
            prog_df,
            x="Student",
            y="Avg Goal %",
            color="Avg Goal %",
            color_continuous_scale=["#f44336", "#FF9800", "#4CAF50"],
            range_color=[0, 100],
            title="Average Goal Completion Rate per Student",
            text="Avg Goal %",
        )
        fig_avg.add_hline(y=100, line_dash="dash", line_color="green",
                           annotation_text="All Goals Met")
        fig_avg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_avg.update_layout(yaxis=dict(range=[0, 115]), coloraxis_showscale=False)
        st.plotly_chart(fig_avg, use_container_width=True)

        # Summary table
        st.markdown("#### Student Goal Summary Table")
        st.dataframe(prog_df, use_container_width=True)

        # Students who met all goals
        all_met = prog_df[prog_df["Subjects Met"] == f"{len(subject_cols)}/{len(subject_cols)}"]
        none_met = prog_df[prog_df["Avg Goal %"] < 60]

        if len(all_met) > 0:
            st.success(f"🏆 Students who met ALL subject targets: **{', '.join(all_met['Student'].tolist())}**")
        if len(none_met) > 0:
            st.error(f"🚨 Students below 60% on average goals (need intervention): **{', '.join(none_met['Student'].tolist())}**")

        # Hardest subject to reach target
        subject_hit_rate = {}
        for sub in subject_cols:
            col_name = f"{sub.title()} %"
            hit_rate = (prog_df[col_name] >= 100).mean() * 100
            subject_hit_rate[sub.title()] = round(hit_rate, 1)

        hardest = min(subject_hit_rate, key=subject_hit_rate.get)
        easiest = max(subject_hit_rate, key=subject_hit_rate.get)
        st.info(f"📌 Hardest subject to meet target: **{hardest}** (only {subject_hit_rate[hardest]:.0f}% of students met target)   |   Easiest: **{easiest}** ({subject_hit_rate[easiest]:.0f}% met target)")

        # ── PDF for class overview goals
        st.markdown("---")

        def make_pdf_goals_class(prog_dataframe, targets_dict):
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
            h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
            normal = styles["Normal"]
            content = []

            content.append(Paragraph("Class Goal Tracking Report", title_style))
            content.append(Spacer(1, 10))

            content.append(Paragraph("Class Targets Set", h2_style))
            tgt_data = [["Subject", "Target Score"]] + [
                [s.title(), str(v)] for s, v in targets_dict.items()
            ]
            tt = Table(tgt_data, colWidths=[3 * inch, 3 * inch])
            tt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3F51B5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e8eaf6"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            content.append(tt)
            content.append(Spacer(1, 14))

            content.append(Paragraph("Student Goal Completion Summary", h2_style))
            col_names = ["Student", "Avg Goal %", "Subjects Met"]
            pd_data = [col_names] + [
                [r["Student"], f"{r['Avg Goal %']:.1f}%", r["Subjects Met"]]
                for _, r in prog_dataframe.iterrows()
            ]
            pt = Table(pd_data, colWidths=[2.5 * inch, 2 * inch, 2 * inch])
            pt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#009688")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e0f2f1"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            content.append(pt)

            doc.build(content)
            return buf.getvalue()

        pdf_goals = make_pdf_goals_class(prog_df, class_targets)
        st.download_button(
            "📄 Download Class Goal Report (PDF)",
            data=pdf_goals,
            file_name="class_goal_report.pdf",
            mime="application/pdf",
        )

    # ── Individual student PDF (shown in both modes)
    if mode == "Individual Student" and goal_student:
        st.markdown("---")

        def make_pdf_goals_student(student_name, s_row_data, tgts, raw_gaps):
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
            h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
            normal = styles["Normal"]
            content = []

            content.append(Paragraph(f"Goal Tracking Report — {student_name}", title_style))
            content.append(Spacer(1, 10))

            content.append(Paragraph("Subject Goals & Progress", h2_style))
            goal_table_data = [["Subject", "Current", "Target", "Gap", "Progress %", "Status"]] + [
                [
                    g["Subject"],
                    f"{g['Current']:.1f}",
                    f"{g['Target']:.1f}",
                    f"{g['Gap']:+.1f}",
                    f"{g['Progress %']:.1f}%",
                    g["Status"],
                ]
                for g in raw_gaps
            ]
            gt = Table(goal_table_data, colWidths=[1.2*inch, 0.9*inch, 0.9*inch, 0.7*inch, 1*inch, 1.5*inch])
            gt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#9C27B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f8f0ff"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            content.append(gt)
            content.append(Spacer(1, 14))

            content.append(Paragraph("Study Recommendations", h2_style))
            for g in raw_gaps:
                if g["Gap"] > 15:
                    priority = "CRITICAL"
                    weeks = max(1, int(g["Gap"] / 5))
                    rec = (f"{g['Subject']} needs urgent attention (gap: {g['Gap']:.0f} pts). "
                           f"Dedicate 45-60 min/day. Estimated {weeks} week(s) to reach goal with consistent effort.")
                elif g["Gap"] > 0:
                    weeks = max(1, int(g["Gap"] / 5))
                    priority = "MODERATE"
                    rec = (f"{g['Subject']} needs steady improvement (gap: {g['Gap']:.0f} pts). "
                           f"30 min/day of focused revision. Estimated {weeks} week(s) to reach goal.")
                else:
                    priority = "ACHIEVED"
                    rec = f"{g['Subject']} target achieved! Consider raising the target to keep improving."
                content.append(Paragraph(f"<b>[{priority}] {g['Subject']}:</b> {rec}", normal))
                content.append(Spacer(1, 4))

            doc.build(content)
            return buf.getvalue()

        pdf_stu_goals = make_pdf_goals_student(goal_student, s_row, targets, gap_data)
        st.download_button(
            "📄 Download Student Goal Report (PDF)",
            data=pdf_stu_goals,
            file_name=f"goal_report_{goal_student.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

# ═══════════════════════════════════════════════════
# TAB 8 — ATTENDANCE FORECAST
# ═══════════════════════════════════════════════════
with tab8:
    st.subheader("📡 Attendance Risk Forecasting")
    st.caption(
        "ML-powered forecast identifying which students are at risk of dropping below "
        "the 75% attendance threshold, with teacher alert suggestions."
    )

    # ── Only possible if attendance column exists
    if "attendance" not in df.columns:
        st.warning("⚠️ No 'attendance' column found in the dataset. Please upload a CSV that includes attendance data.")
    else:
        # ─────────────────────────────────────────────
        # BUILD ATTENDANCE RISK MODEL
        # ─────────────────────────────────────────────
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        import warnings
        warnings.filterwarnings("ignore")

        ATT_THRESHOLD = 75

        # Features that correlate with attendance
        att_feature_candidates = [
            "study_hours", "stress_level", "sleep_hours",
            "screen_time", "participation", "assignments",
            "sports", "Average",
        ]
        att_features = [c for c in att_feature_candidates if c in df.columns]

        # Current attendance labels
        df["Att_Risk"] = (df["attendance"] < ATT_THRESHOLD).astype(int)

        # Train regressor to predict attendance from other features
        att_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        att_reg.fit(df[att_features], df["attendance"])

        # Train classifier to flag at-risk students
        att_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        att_clf.fit(df[att_features], df["Att_Risk"])

        # ── Simulate "next period" projections using mild perturbations
        # Stress +1, study_hours −0.3, screen_time +0.5  →  realistic trend for at-risk students
        np.random.seed(0)
        df_proj = df.copy()
        if "stress_level" in df_proj.columns:
            df_proj["stress_level"] = (df_proj["stress_level"] + np.random.uniform(0, 1, len(df_proj))).clip(upper=10)
        if "study_hours" in df_proj.columns:
            df_proj["study_hours"] = (df_proj["study_hours"] - np.random.uniform(0, 0.5, len(df_proj))).clip(lower=0)
        if "screen_time" in df_proj.columns:
            df_proj["screen_time"] = (df_proj["screen_time"] + np.random.uniform(0, 0.5, len(df_proj))).clip(upper=10)

        projected_att = att_reg.predict(df_proj[att_features]).clip(0, 100)
        projected_risk = att_clf.predict(df_proj[att_features])
        projected_prob = att_clf.predict_proba(df_proj[att_features])[:, 1]

        # ── Assemble forecast table
        forecast_df = pd.DataFrame({
            "Student": df[name_col].values,
            "Current Att %": df["attendance"].round(1).values,
            "Projected Att %": projected_att.round(1),
            "Risk Prob %": (projected_prob * 100).round(1),
            "Projected Risk": ["At Risk" if r == 1 else "Safe" for r in projected_risk],
            "Risk Level": df["Risk"].values,
            "Average": df["Average"].round(1).values,
        })
        forecast_df = forecast_df.sort_values("Risk Prob %", ascending=False).reset_index(drop=True)

        # ── Feature importance
        feat_importance = pd.Series(att_reg.feature_importances_, index=att_features).sort_values(ascending=False)

        # ─────────────────────────────────────────────
        # SUMMARY METRICS
        # ─────────────────────────────────────────────
        n_currently_at_risk = int((df["attendance"] < ATT_THRESHOLD).sum())
        n_projected_at_risk = int((projected_att < ATT_THRESHOLD).sum())
        n_high_prob = int((projected_prob >= 0.6).sum())
        avg_projected = projected_att.mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Currently Below 75%", n_currently_at_risk,
                  delta=None)
        c2.metric("Projected Below 75%", n_projected_at_risk,
                  delta=int(n_projected_at_risk - n_currently_at_risk),
                  delta_color="inverse")
        c3.metric("High-Probability At Risk (≥60%)", n_high_prob)
        c4.metric("Avg Projected Attendance", f"{avg_projected:.1f}%")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # RISK GAUGE / PROBABILITY CHART
        # ─────────────────────────────────────────────
        st.markdown("#### 📊 Attendance Drop Risk Probability per Student")

        color_map = forecast_df["Risk Prob %"].apply(
            lambda x: "#f44336" if x >= 60 else ("#FF9800" if x >= 35 else "#4CAF50")
        )

        fig_prob = go.Figure(go.Bar(
            x=forecast_df["Student"],
            y=forecast_df["Risk Prob %"],
            marker_color=color_map.tolist(),
            text=forecast_df["Risk Prob %"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig_prob.add_hline(y=60, line_dash="dash", line_color="#f44336",
                           annotation_text="High-Risk Threshold (60%)")
        fig_prob.add_hline(y=35, line_dash="dash", line_color="#FF9800",
                           annotation_text="Moderate-Risk Threshold (35%)")
        fig_prob.update_layout(
            title="Probability of Falling Below 75% Attendance (Next Period)",
            yaxis=dict(range=[0, 115], title="Risk Probability (%)"),
            xaxis_title="Student",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # ─────────────────────────────────────────────
        # CURRENT vs PROJECTED ATTENDANCE
        # ─────────────────────────────────────────────
        st.markdown("#### 📉 Current vs Projected Attendance")

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Current Attendance %",
            x=forecast_df["Student"],
            y=forecast_df["Current Att %"],
            marker_color="#2196F3",
            text=forecast_df["Current Att %"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig_comp.add_trace(go.Bar(
            name="Projected Attendance %",
            x=forecast_df["Student"],
            y=forecast_df["Projected Att %"],
            marker_color="#FF9800",
            text=forecast_df["Projected Att %"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig_comp.add_hline(y=ATT_THRESHOLD, line_dash="dash", line_color="#f44336",
                           annotation_text="75% Threshold")
        fig_comp.update_layout(
            barmode="group",
            title="Current vs Projected Attendance per Student",
            yaxis=dict(range=[0, 115], title="Attendance %"),
            xaxis_title="Student",
            height=420,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # ─────────────────────────────────────────────
        # FEATURE IMPORTANCE
        # ─────────────────────────────────────────────
        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### 🔍 Key Drivers of Attendance")
            fig_fi = px.bar(
                x=feat_importance.values * 100,
                y=feat_importance.index.str.title(),
                orientation="h",
                title="Feature Importance for Attendance Prediction",
                labels={"x": "Importance (%)", "y": "Factor"},
                color=feat_importance.values,
                color_continuous_scale=["#90CAF9", "#1565C0"],
            )
            fig_fi.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        with colB:
            st.markdown("#### 🗂 Risk Distribution (Projected)")
            risk_counts = forecast_df["Projected Risk"].value_counts().reset_index()
            risk_counts.columns = ["Status", "Count"]
            fig_rd = px.pie(
                risk_counts, names="Status", values="Count",
                color="Status",
                color_discrete_map={"At Risk": "#f44336", "Safe": "#4CAF50"},
                title="Projected Risk Distribution",
            )
            st.plotly_chart(fig_rd, use_container_width=True)

        st.markdown("---")

        # ─────────────────────────────────────────────
        # FORECAST TABLE
        # ─────────────────────────────────────────────
        st.markdown("#### 📋 Full Forecast Table")

        def style_risk(val):
            if val == "At Risk":
                return "background-color:#fff3f3; color:#c62828; font-weight:bold"
            return "background-color:#f0fff0; color:#2e7d32"

        def style_prob(val):
            try:
                v = float(str(val).replace("%", ""))
                if v >= 60:
                    return "color:#c62828; font-weight:bold"
                elif v >= 35:
                    return "color:#e65100; font-weight:bold"
                return "color:#2e7d32"
            except Exception:
                return ""

        display_fc = forecast_df.copy()
        st.dataframe(
            display_fc.style
                .applymap(style_risk, subset=["Projected Risk"])
                .applymap(style_prob, subset=["Risk Prob %"]),
            use_container_width=True,
        )

        st.markdown("---")

        # ─────────────────────────────────────────────
        # AUTOMATED TEACHER ALERTS
        # ─────────────────────────────────────────────
        st.markdown("#### 🚨 Automated Teacher Alerts")
        st.caption("Prioritised action list for teachers based on ML forecast results.")

        high_risk_students = forecast_df[forecast_df["Risk Prob %"] >= 60]
        moderate_risk_students = forecast_df[
            (forecast_df["Risk Prob %"] >= 35) & (forecast_df["Risk Prob %"] < 60)
        ]
        safe_students = forecast_df[forecast_df["Risk Prob %"] < 35]

        if len(high_risk_students) > 0:
            st.error(f"🔴 **URGENT — {len(high_risk_students)} student(s) with HIGH drop risk (≥60% probability)**")
            for _, s in high_risk_students.iterrows():
                delta = s["Projected Att %"] - s["Current Att %"]
                with st.expander(
                    f"🚨 {s['Student']} — Risk: {s['Risk Prob %']:.0f}% | "
                    f"Projected: {s['Projected Att %']:.1f}% ({delta:+.1f}%)"
                ):
                    urgency = "URGENT" if s["Projected Att %"] < 60 else "HIGH PRIORITY"
                    st.markdown(f"**Status:** [{urgency}] Attendance drop forecasted")
                    st.markdown(f"**Current attendance:** {s['Current Att %']:.1f}%")
                    st.markdown(f"**Projected next-period attendance:** {s['Projected Att %']:.1f}%")
                    st.markdown(f"**Academic risk level:** {s['Risk Level']}")
                    st.markdown(f"**Average score:** {s['Average']:.1f}")
                    st.markdown("---")
                    st.markdown("**Recommended Teacher Actions:**")
                    actions = [
                        f"📞 Schedule a one-on-one meeting with **{s['Student']}** and their parents/guardian this week.",
                        "📝 Review recent absence records and identify any recurring patterns (days of week, specific subjects).",
                        "💬 Conduct a welfare check — ask about personal, health, or motivation challenges.",
                        "📚 Provide a catch-up plan for missed content so the student doesn't feel overwhelmed on return.",
                        "🎯 Set a short-term attendance goal (e.g. full attendance for the next 2 weeks) with positive reinforcement.",
                        "🔔 Flag to the school counsellor if attendance has dropped more than 10% in the past month.",
                    ]
                    if "stress_level" in df.columns:
                        stress_val = df[df[name_col] == s["Student"]]["stress_level"].values
                        if len(stress_val) > 0 and stress_val[0] >= 7:
                            actions.append("🧘 Student shows high stress — refer to counselling or wellness program.")
                    for a in actions:
                        st.markdown(f"- {a}")

        if len(moderate_risk_students) > 0:
            st.warning(f"🟠 **MODERATE — {len(moderate_risk_students)} student(s) with moderate drop risk (35–59%)**")
            for _, s in moderate_risk_students.iterrows():
                delta = s["Projected Att %"] - s["Current Att %"]
                with st.expander(
                    f"⚠️ {s['Student']} — Risk: {s['Risk Prob %']:.0f}% | "
                    f"Projected: {s['Projected Att %']:.1f}% ({delta:+.1f}%)"
                ):
                    st.markdown(f"**Current attendance:** {s['Current Att %']:.1f}%")
                    st.markdown(f"**Projected next-period attendance:** {s['Projected Att %']:.1f}%")
                    st.markdown("**Recommended Teacher Actions:**")
                    st.markdown(
                        f"- 💬 Check in with **{s['Student']}** informally — a brief conversation can prevent further disengagement.\n"
                        "- 📣 Send a friendly reminder about the importance of consistent attendance.\n"
                        "- 📊 Monitor attendance weekly and escalate if any further decline is observed.\n"
                        "- 🏫 Ensure the student feels included and engaged in class activities."
                    )

        if len(safe_students) > 0:
            with st.expander(f"✅ {len(safe_students)} student(s) projected as SAFE (risk < 35%)"):
                safe_list = ", ".join(safe_students["Student"].tolist())
                st.success(f"The following students are projected to maintain healthy attendance: **{safe_list}**")
                st.markdown("Continue monitoring and recognise consistent attendance positively.")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # INDIVIDUAL DEEP DIVE
        # ─────────────────────────────────────────────
        st.markdown("#### 🔎 Individual Attendance Deep Dive")
        dive_student = st.selectbox("Select a student to analyse:", df[name_col].tolist(), key="att_dive_select")
        d_row = df[df[name_col] == dive_student].iloc[0]
        d_fc = forecast_df[forecast_df["Student"] == dive_student].iloc[0]

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Current Attendance", f"{d_fc['Current Att %']:.1f}%")
        dc2.metric(
            "Projected Attendance",
            f"{d_fc['Projected Att %']:.1f}%",
            delta=f"{d_fc['Projected Att %'] - d_fc['Current Att %']:+.1f}%",
            delta_color="inverse",
        )
        dc3.metric("Drop Risk Probability", f"{d_fc['Risk Prob %']:.1f}%")

        # Factor breakdown radar for this student
        radar_factors = att_features
        student_vals = [float(d_row.get(f, 0)) for f in radar_factors]
        class_vals = [float(df[f].mean()) for f in radar_factors]

        fig_radar_att = go.Figure()
        fig_radar_att.add_trace(go.Scatterpolar(
            r=student_vals + [student_vals[0]],
            theta=[f.replace("_", " ").title() for f in radar_factors] + [radar_factors[0].replace("_", " ").title()],
            fill="toself",
            name=dive_student,
            line_color="#2196F3",
        ))
        fig_radar_att.add_trace(go.Scatterpolar(
            r=class_vals + [class_vals[0]],
            theta=[f.replace("_", " ").title() for f in radar_factors] + [radar_factors[0].replace("_", " ").title()],
            fill="toself",
            name="Class Average",
            line_color="#9E9E9E",
            opacity=0.5,
        ))
        fig_radar_att.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=f"Factor Profile: {dive_student} vs Class Average",
            height=380,
        )
        st.plotly_chart(fig_radar_att, use_container_width=True)

        # Attendance scenario simulator
        st.markdown("##### 🧪 Scenario Simulator")
        st.caption("Adjust factors below to simulate how changes might affect projected attendance.")

        sim_cols = st.columns(min(4, len(att_features)))
        sim_vals = {}
        for i, feat in enumerate(att_features):
            with sim_cols[i % len(sim_cols)]:
                curr_val = float(d_row.get(feat, df[feat].mean()))
                sim_vals[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=float(df[feat].max()),
                    value=round(curr_val, 1),
                    step=0.5,
                    key=f"sim_{dive_student}_{feat}",
                )

        sim_input = pd.DataFrame([sim_vals])[att_features]
        sim_predicted_att = float(att_reg.predict(sim_input)[0])
        sim_risk_prob = float(att_clf.predict_proba(sim_input)[0][1]) * 100
        sim_predicted_att = max(0, min(100, sim_predicted_att))

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Simulated Attendance", f"{sim_predicted_att:.1f}%",
                   delta=f"{sim_predicted_att - d_fc['Current Att %']:+.1f}%",
                   delta_color="normal")
        sc2.metric("Simulated Drop Risk", f"{sim_risk_prob:.1f}%",
                   delta=f"{sim_risk_prob - d_fc['Risk Prob %']:+.1f}%",
                   delta_color="inverse")
        if sim_predicted_att >= ATT_THRESHOLD:
            sc3.success(f"✅ Above safe threshold ({ATT_THRESHOLD}%)")
        else:
            sc3.error(f"⚠️ Below threshold ({ATT_THRESHOLD}%)")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # PDF REPORT
        # ─────────────────────────────────────────────
        def make_pdf_attendance(fc_df, feat_imp, n_high, n_mod, threshold):
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
            h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
            normal = styles["Normal"]
            content = []

            content.append(Paragraph("Attendance Risk Forecast Report", title_style))
            content.append(Spacer(1, 10))

            # Summary
            content.append(Paragraph("Forecast Summary", h2_style))
            summary = [
                ["Metric", "Value"],
                ["Attendance Threshold", f"{threshold}%"],
                ["High-Risk Students (≥60% drop prob)", str(n_high)],
                ["Moderate-Risk Students (35–59%)", str(n_mod)],
                ["Avg Projected Attendance", f"{fc_df['Projected Att %'].mean():.1f}%"],
                ["Students Projected Below Threshold", str(int((fc_df['Projected Att %'] < threshold).sum()))],
            ]
            t = Table(summary, colWidths=[3.5 * inch, 3 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1565C0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#E3F2FD"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            content.append(t)
            content.append(Spacer(1, 14))

            # Full forecast table
            content.append(Paragraph("Student-level Forecast", h2_style))
            cols = ["Student", "Current Att %", "Projected Att %", "Risk Prob %", "Projected Risk"]
            fc_rows = [cols] + [
                [r["Student"], f"{r['Current Att %']:.1f}%",
                 f"{r['Projected Att %']:.1f}%", f"{r['Risk Prob %']:.1f}%", r["Projected Risk"]]
                for _, r in fc_df.iterrows()
            ]
            ft = Table(fc_rows, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 1.1*inch, 1.2*inch])
            ft.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1565C0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#E3F2FD"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            content.append(ft)
            content.append(Spacer(1, 14))

            # Key drivers
            content.append(Paragraph("Key Drivers of Attendance (Feature Importance)", h2_style))
            fi_rows = [["Factor", "Importance %"]] + [
                [f.replace("_", " ").title(), f"{v*100:.1f}%"]
                for f, v in feat_imp.items()
            ]
            fit = Table(fi_rows, colWidths=[3.5 * inch, 3 * inch])
            fit.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0D47A1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#E8EAF6"), rl_colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            content.append(fit)
            content.append(Spacer(1, 14))

            # High-risk alert list
            hr = fc_df[fc_df["Risk Prob %"] >= 60]
            if len(hr) > 0:
                content.append(Paragraph("🚨 High-Risk Teacher Alerts", h2_style))
                for _, s in hr.iterrows():
                    content.append(Paragraph(
                        f"<b>{s['Student']}</b>: Current {s['Current Att %']:.1f}% → "
                        f"Projected {s['Projected Att %']:.1f}% | Risk {s['Risk Prob %']:.0f}%. "
                        f"Action: Schedule parent meeting, welfare check, and catch-up plan immediately.",
                        normal,
                    ))
                    content.append(Spacer(1, 4))

            doc.build(content)
            return buf.getvalue()

        pdf_att = make_pdf_attendance(
            forecast_df, feat_importance,
            len(high_risk_students), len(moderate_risk_students), ATT_THRESHOLD,
        )
        st.download_button(
            "📄 Download Attendance Forecast Report (PDF)",
            data=pdf_att,
            file_name="attendance_forecast_report.pdf",
            mime="application/pdf",
        )

# ═══════════════════════════════════════════════════
# TAB 9 — REPORT CARDS
# ═══════════════════════════════════════════════════
with tab9:
    st.subheader("📜 End-of-Term Report Card Generator")
    st.caption(
        "Generate professional styled report cards for individual students or download "
        "a complete PDF pack for the entire class."
    )

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def score_to_grade(score: float) -> str:
        if score >= 90: return "A+"
        elif score >= 80: return "A"
        elif score >= 70: return "B+"
        elif score >= 60: return "B"
        elif score >= 50: return "C"
        elif score >= 40: return "D"
        return "F"

    def grade_to_gpa(grade: str) -> str:
        return {"A+": "4.0", "A": "3.7", "B+": "3.3", "B": "3.0",
                "C": "2.0", "D": "1.0", "F": "0.0"}.get(grade, "—")

    def score_to_remark(score: float) -> str:
        if score >= 90: return "Outstanding"
        elif score >= 80: return "Excellent"
        elif score >= 70: return "Very Good"
        elif score >= 60: return "Good"
        elif score >= 50: return "Satisfactory"
        elif score >= 40: return "Needs Improvement"
        return "Unsatisfactory"

    def risk_to_conduct(risk: str) -> str:
        return {"Low": "Excellent", "Moderate": "Satisfactory", "High": "Needs Attention"}.get(risk, "—")

    def generate_teacher_comment(row: pd.Series, subject_cols_list: list) -> str:
        name = row[name_col].split()[0]
        avg = row["Average"]
        risk = row["Risk"]
        pf = row["Pass_Fail"]
        trend = row.get("Trend", 0)
        att = row.get("attendance", None)
        stress = row.get("stress_level", None)
        study = row.get("study_hours", None)
        weak = min(subject_cols_list, key=lambda x: row[x])
        strong = max(subject_cols_list, key=lambda x: row[x])

        opener = {
            "A+": f"{name} has delivered an exceptional performance this term,",
            "A":  f"{name} has performed excellently this term,",
            "B+": f"{name} has shown very good academic progress this term,",
            "B":  f"{name} has demonstrated good performance this term,",
            "C":  f"{name} has achieved satisfactory results this term,",
            "D":  f"{name} has shown some effort this term, though results need improvement,",
            "F":  f"{name} has struggled significantly this term,",
        }[score_to_grade(avg)]

        body = f"achieving an overall average of {avg:.1f}%. "

        if trend > 5:
            body += f"There has been a notable improvement of {trend:.1f} points compared to the previous assessment. "
        elif trend < -5:
            body += f"Performance has declined by {abs(trend):.1f} points from the previous assessment. "

        body += f"Strongest subject: {strong.title()} ({row[strong]:.0f}%). "
        if row[weak] < 60:
            body += f"{weak.title()} requires focused attention ({row[weak]:.0f}%). "

        if att is not None:
            if att >= 90:
                body += f"Attendance has been commendable at {att:.0f}%. "
            elif att < 75:
                body += f"Attendance of {att:.0f}% is a concern and must be addressed urgently. "

        if stress is not None and stress >= 7:
            body += "The student appears to be experiencing elevated stress levels — pastoral support is recommended. "

        if study is not None and study >= 3:
            body += f"Consistent study habits ({study:.1f}h/day) are reflected in the results. "
        elif study is not None and study < 2:
            body += f"Increasing daily study time beyond the current {study:.1f}h would significantly help performance. "

        closer = {
            "Low":      f"We encourage {name} to maintain this momentum and continue setting high goals.",
            "Moderate": f"With consistent effort and focus, {name} has the potential to achieve stronger results next term.",
            "High":     f"We strongly encourage {name} and their family to work closely with the school to address these challenges.",
        }.get(risk, "")

        return opener + body + closer

    # ─────────────────────────────────────────────
    # REPORT CARD PDF BUILDER (single student)
    # ─────────────────────────────────────────────
    def make_report_card_pdf(row: pd.Series, term: str, school: str, teacher: str) -> bytes:
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
            pagesize=(8.27 * 72, 11.69 * 72),   # A4
        )
        styles = getSampleStyleSheet()
        W = 8.27 * 72 - 72   # usable width

        # Custom styles
        school_style   = ParagraphStyle("School",  fontSize=16, fontName="Helvetica-Bold",
                                        alignment=1, textColor=rl_colors.HexColor("#1565C0"), spaceAfter=2)
        title_style    = ParagraphStyle("Title",   fontSize=12, fontName="Helvetica",
                                        alignment=1, textColor=rl_colors.HexColor("#424242"), spaceAfter=10)
        section_style  = ParagraphStyle("Section", fontSize=11, fontName="Helvetica-Bold",
                                        textColor=rl_colors.HexColor("#1565C0"), spaceAfter=4, spaceBefore=8)
        normal         = ParagraphStyle("Normal2", fontSize=9, fontName="Helvetica", leading=13)
        comment_style  = ParagraphStyle("Comment", fontSize=9, fontName="Helvetica",
                                        leading=14, textColor=rl_colors.HexColor("#212121"))
        grade_A_style  = ParagraphStyle("GradeA",  fontSize=9, fontName="Helvetica-Bold",
                                        textColor=rl_colors.HexColor("#2e7d32"))
        grade_F_style  = ParagraphStyle("GradeF",  fontSize=9, fontName="Helvetica-Bold",
                                        textColor=rl_colors.HexColor("#c62828"))

        sname = row[name_col]
        avg   = row["Average"]
        grade = score_to_grade(avg)
        gpa   = grade_to_gpa(grade)
        comment = generate_teacher_comment(row, subject_cols)

        content = []

        # ── Header banner (simulated with a table)
        header_data = [[Paragraph(school, school_style)],
                       [Paragraph(f"Student Academic Report Card  |  {term}", title_style)]]
        header_tbl = Table(header_data, colWidths=[W])
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), rl_colors.HexColor("#E3F2FD")),
            ("TOPPADDING",  (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0,0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("BOX",         (0, 0), (-1, -1), 1.5, rl_colors.HexColor("#1565C0")),
        ]))
        content.append(header_tbl)
        content.append(Spacer(1, 10))

        # ── Student info row
        att_val   = f"{row.get('attendance', '—')}%" if 'attendance' in row.index else "—"
        info_data = [
            [Paragraph("<b>Student Name:</b>", normal),  Paragraph(sname, normal),
             Paragraph("<b>Teacher:</b>", normal),        Paragraph(teacher, normal)],
            [Paragraph("<b>Overall Grade:</b>", normal),  Paragraph(f"<b>{grade}</b>", grade_A_style if avg >= 50 else grade_F_style),
             Paragraph("<b>GPA:</b>", normal),            Paragraph(gpa, normal)],
            [Paragraph("<b>Overall Average:</b>", normal),Paragraph(f"{avg:.1f}%", normal),
             Paragraph("<b>Attendance:</b>", normal),     Paragraph(att_val, normal)],
            [Paragraph("<b>Class Rank:</b>", normal),     Paragraph(f"#{int(row['Rank'])} of {len(df)}", normal),
             Paragraph("<b>Status:</b>", normal),         Paragraph(row["Pass_Fail"], normal)],
        ]
        info_tbl = Table(info_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
        info_tbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [rl_colors.HexColor("#F5F5F5"), rl_colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#BDBDBD")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        content.append(info_tbl)
        content.append(Spacer(1, 10))

        # ── Subject marks table
        content.append(Paragraph("Subject-wise Performance", section_style))
        sub_header = ["Subject", "Score", "Grade", "Remark", "Class Avg", "Rank in Subject"]
        sub_rows   = []
        for sub in subject_cols:
            score     = row[sub]
            cls_avg   = df[sub].mean()
            sub_rank  = int(df[sub].rank(ascending=False, method="min")[df[name_col] == sname].values[0])
            grade_sub = score_to_grade(score)
            remark    = score_to_remark(score)
            sub_rows.append([
                sub.title(),
                f"{score:.0f}%",
                grade_sub,
                remark,
                f"{cls_avg:.1f}%",
                f"#{sub_rank}",
            ])
        # Sort by score descending
        sub_rows.sort(key=lambda x: -float(x[1].replace("%","")))

        sub_data  = [sub_header] + sub_rows
        sub_widths = [W*0.20, W*0.10, W*0.10, W*0.25, W*0.18, W*0.17]
        sub_tbl   = Table(sub_data, colWidths=sub_widths)
        sub_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor("#1565C0")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  rl_colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.HexColor("#F9F9F9"), rl_colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#BDBDBD")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
        ])
        # Colour-code failing subjects
        for i, sr in enumerate(sub_rows, start=1):
            score_val = float(sr[1].replace("%",""))
            if score_val < 50:
                sub_style.add("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#FFEBEE"))
                sub_style.add("TEXTCOLOR",  (2, i), (2, i),  rl_colors.HexColor("#c62828"))
            elif score_val >= 80:
                sub_style.add("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#E8F5E9"))
                sub_style.add("TEXTCOLOR",  (2, i), (2, i),  rl_colors.HexColor("#2e7d32"))
        sub_tbl.setStyle(sub_style)
        content.append(sub_tbl)
        content.append(Spacer(1, 10))

        # ── Behaviour / co-curricular row
        content.append(Paragraph("Conduct & Co-curricular", section_style))
        conduct_data = [
            ["Conduct", "Class Participation", "Study Habits", "Sports/Activities"],
            [
                risk_to_conduct(row["Risk"]),
                score_to_remark(float(row.get("participation", 5)) * 10),
                "Dedicated" if row.get("study_hours", 0) >= 3 else "Developing",
                "Active" if row.get("sports", 0) == 1 else "Not Reported",
            ]
        ]
        ct = Table(conduct_data, colWidths=[W*0.25]*4)
        ct.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor("#0D47A1")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  rl_colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.HexColor("#E8EAF6")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#9FA8DA")),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        content.append(ct)
        content.append(Spacer(1, 10))

        # ── Performance summary bar (visual)
        content.append(Paragraph("Performance Summary", section_style))
        perf_cols  = [s for s in subject_cols] + (["attendance"] if "attendance" in row.index else [])
        bar_labels = [s.title() for s in perf_cols]
        bar_values = [min(100, max(0, float(row.get(s, 0)))) for s in perf_cols]

        BAR_W = W / len(bar_values)
        bar_rows = []
        for lbl, val in zip(bar_labels, bar_values):
            color = rl_colors.HexColor("#4CAF50") if val >= 70 else (
                    rl_colors.HexColor("#FF9800") if val >= 50 else rl_colors.HexColor("#f44336"))
            bar_rows.append(Paragraph(f"<b>{lbl}</b>  {val:.0f}%", normal))
        perf_tbl = Table([bar_rows], colWidths=[BAR_W]*len(bar_rows))
        perf_tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.2, rl_colors.HexColor("#E0E0E0")),
        ]))
        content.append(perf_tbl)
        content.append(Spacer(1, 10))

        # ── Teacher comment
        content.append(Paragraph("Class Teacher's Comment", section_style))
        comment_box_data = [[Paragraph(comment, comment_style)]]
        comment_tbl = Table(comment_box_data, colWidths=[W])
        comment_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), rl_colors.HexColor("#FFFDE7")),
            ("BOX",           (0, 0), (-1, -1), 0.8, rl_colors.HexColor("#F9A825")),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        content.append(comment_tbl)
        content.append(Spacer(1, 14))

        # ── Signature strip
        sig_data = [
            [Paragraph("Class Teacher's Signature", normal),
             Paragraph("Principal's Signature", normal),
             Paragraph("Parent/Guardian Signature", normal)],
            [Paragraph("_____________________", normal),
             Paragraph("_____________________", normal),
             Paragraph("_____________________", normal)],
            [Paragraph(f"<b>{teacher}</b>", normal),
             Paragraph("", normal),
             Paragraph("", normal)],
        ]
        sig_tbl = Table(sig_data, colWidths=[W/3]*3)
        sig_tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEABOVE",     (0, 0), (-1, 0),  0.5, rl_colors.HexColor("#BDBDBD")),
        ]))
        content.append(sig_tbl)

        # ── Footer
        content.append(Spacer(1, 8))
        footer_data = [[Paragraph(
            f"This report is computer-generated by the Student Performance Intelligence System  |  {term}  |  {school}",
            ParagraphStyle("Footer", fontSize=7, textColor=rl_colors.HexColor("#9E9E9E"), alignment=1),
        )]]
        footer_tbl = Table(footer_data, colWidths=[W])
        footer_tbl.setStyle(TableStyle([
            ("LINEABOVE",     (0, 0), (-1, 0),  0.5, rl_colors.HexColor("#BDBDBD")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ]))
        content.append(footer_tbl)

        doc.build(content)
        return buf.getvalue()

    # ─────────────────────────────────────────────
    # UI — SETTINGS
    # ─────────────────────────────────────────────
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        rc_school  = st.text_input("🏫 School Name", value="Greenwood International School")
    with rc2:
        rc_term    = st.text_input("📅 Term / Period", value="Term 2 — Academic Year 2025–26")
    with rc3:
        rc_teacher = st.text_input("👩‍🏫 Class Teacher Name", value="Ms. Priya Sharma")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # PREVIEW — SINGLE STUDENT
    # ─────────────────────────────────────────────
    st.markdown("#### 👤 Individual Report Card Preview")
    rc_student = st.selectbox("Select student to preview:", df[name_col].tolist(), key="rc_student_select")
    rc_row     = df[df[name_col] == rc_student].iloc[0]

    # Live preview in Streamlit
    preview_cols = st.columns([1, 1])
    with preview_cols[0]:
        st.markdown(f"**Student:** {rc_row[name_col]}")
        st.markdown(f"**Overall Average:** {rc_row['Average']:.1f}%")
        st.markdown(f"**Grade:** {score_to_grade(rc_row['Average'])}")
        st.markdown(f"**GPA:** {grade_to_gpa(score_to_grade(rc_row['Average']))}")
        st.markdown(f"**Class Rank:** #{int(rc_row['Rank'])} of {len(df)}")
        st.markdown(f"**Status:** {rc_row['Pass_Fail']}")
        if "attendance" in df.columns:
            st.markdown(f"**Attendance:** {rc_row.get('attendance', '—')}%")
        st.markdown(f"**Conduct:** {risk_to_conduct(rc_row['Risk'])}")

    with preview_cols[1]:
        sub_preview = {s.title(): rc_row[s] for s in subject_cols}
        fig_rc = px.bar(
            x=list(sub_preview.keys()),
            y=list(sub_preview.values()),
            color=list(sub_preview.values()),
            color_continuous_scale=["#f44336", "#FF9800", "#4CAF50"],
            range_color=[0, 100],
            title="Subject Scores",
            labels={"x": "Subject", "y": "Score", "color": "Score"},
        )
        fig_rc.add_hline(y=50, line_dash="dash", line_color="#f44336", annotation_text="Pass")
        fig_rc.update_layout(coloraxis_showscale=False, height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig_rc, use_container_width=True)

    # Grade breakdown table
    grade_rows = []
    for sub in subject_cols:
        score = rc_row[sub]
        cls_avg = df[sub].mean()
        sub_rank = int(df[sub].rank(ascending=False, method="min")[df[name_col] == rc_student].values[0])
        grade_rows.append({
            "Subject": sub.title(),
            "Score": f"{score:.0f}%",
            "Grade": score_to_grade(score),
            "GPA": grade_to_gpa(score_to_grade(score)),
            "Remark": score_to_remark(score),
            "Class Avg": f"{cls_avg:.1f}%",
            "Rank": f"#{sub_rank}",
        })
    grade_rows.sort(key=lambda x: -float(x["Score"].replace("%", "")))
    st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)

    # Teacher comment preview
    st.markdown("**Auto-generated Teacher Comment:**")
    tc = generate_teacher_comment(rc_row, subject_cols)
    st.info(tc)

    # Individual PDF download
    st.markdown("---")
    single_pdf = make_report_card_pdf(rc_row, rc_term, rc_school, rc_teacher)
    st.download_button(
        f"📄 Download {rc_student}'s Report Card (PDF)",
        data=single_pdf,
        file_name=f"report_card_{rc_student.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CLASS OVERVIEW TABLE
    # ─────────────────────────────────────────────
    st.markdown("#### 🏫 Full Class Report Card Summary")

    summary_rows = []
    for _, srow in df.iterrows():
        avg  = srow["Average"]
        grade = score_to_grade(avg)
        summary_rows.append({
            "Student": srow[name_col],
            "Average": f"{avg:.1f}%",
            "Grade": grade,
            "GPA": grade_to_gpa(grade),
            "Rank": f"#{int(srow['Rank'])}",
            "Pass/Fail": srow["Pass_Fail"],
            "Conduct": risk_to_conduct(srow["Risk"]),
            "Attendance": f"{srow.get('attendance', '—')}%" if "attendance" in df.columns else "—",
        })
    summary_rows.sort(key=lambda x: int(x["Rank"].replace("#", "")))
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Grade distribution chart
    grade_dist = summary_df["Grade"].value_counts().reset_index()
    grade_dist.columns = ["Grade", "Count"]
    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    grade_dist["Grade"] = pd.Categorical(grade_dist["Grade"], categories=grade_order, ordered=True)
    grade_dist = grade_dist.sort_values("Grade")

    col_gd1, col_gd2 = st.columns(2)
    with col_gd1:
        fig_gd = px.bar(
            grade_dist, x="Grade", y="Count",
            color="Grade",
            color_discrete_map={
                "A+": "#1B5E20", "A": "#4CAF50", "B+": "#8BC34A",
                "B": "#CDDC39", "C": "#FF9800", "D": "#FF5722", "F": "#f44336",
            },
            title="Grade Distribution Across Class",
            text="Count",
        )
        fig_gd.update_traces(textposition="outside")
        fig_gd.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_gd, use_container_width=True)

    with col_gd2:
        fig_gpd = px.pie(
            grade_dist, names="Grade", values="Count",
            color="Grade",
            color_discrete_map={
                "A+": "#1B5E20", "A": "#4CAF50", "B+": "#8BC34A",
                "B": "#CDDC39", "C": "#FF9800", "D": "#FF5722", "F": "#f44336",
            },
            title="Grade Distribution (Pie)",
        )
        fig_gpd.update_layout(height=300)
        st.plotly_chart(fig_gpd, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # FULL CLASS PDF PACK
    # ─────────────────────────────────────────────
    st.markdown("#### 📦 Download Full Class Report Card Pack")
    st.caption(
        "Generates one PDF containing report cards for all students in the class, "
        "ready for printing or distribution."
    )

    if st.button("🖨️ Generate Full Class PDF Pack", key="gen_class_pack_btn"):
        with st.spinner(f"Generating report cards for all {len(df)} students…"):
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import PageBreak

            pack_buf  = BytesIO()
            pack_doc  = SimpleDocTemplate(
                pack_buf, rightMargin=36, leftMargin=36,
                topMargin=36, bottomMargin=36, pagesize=A4,
            )
            all_content = []

            for i, (_, srow) in enumerate(df.sort_values("Rank").iterrows()):
                # Generate each card's content and insert a page break between cards
                card_buf = BytesIO()
                card_doc = SimpleDocTemplate(
                    card_buf, rightMargin=36, leftMargin=36,
                    topMargin=36, bottomMargin=36, pagesize=A4,
                )
                # Re-use the helper but capture flowables instead
                # Build single card
                single = make_report_card_pdf(srow, rc_term, rc_school, rc_teacher)
                # Each card is its own bytes — merge via concatenation into a multi-page doc
                all_content.append(single)

            # Merge PDFs using simple byte concatenation via reportlab multi-story trick
            # Build a fresh merged document
            merged_buf = BytesIO()
            from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
            from reportlab.lib.pagesizes import A4 as A4ps

            # Simple approach: write each card PDF then concatenate with page breaks
            # Use pypdf if available, otherwise provide individual zipped cards
            try:
                from pypdf import PdfWriter, PdfReader
                writer = PdfWriter()
                for card_bytes in all_content:
                    reader = PdfReader(BytesIO(card_bytes))
                    for page in reader.pages:
                        writer.add_page(page)
                writer.write(merged_buf)
                merged_bytes = merged_buf.getvalue()
                st.success(f"✅ Report card pack generated for {len(df)} students!")
                st.download_button(
                    "📥 Download Full Class PDF Pack",
                    data=merged_bytes,
                    file_name=f"class_report_cards_{rc_term.replace(' ', '_').replace('|','').replace('—','')}.pdf",
                    mime="application/pdf",
                    key="download_class_pack_btn",
                )
            except ImportError:
                # Fallback: download cards as a zip
                import zipfile
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for card_bytes, (_, srow) in zip(all_content, df.sort_values("Rank").iterrows()):
                        fname = f"report_card_{srow[name_col].replace(' ', '_')}.pdf"
                        zf.writestr(fname, card_bytes)
                st.success(f"✅ {len(df)} report cards packed into a ZIP!")
                st.download_button(
                    "📥 Download All Report Cards (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name="class_report_cards.zip",
                    mime="application/zip",
                    key="download_class_zip_btn",
                )
