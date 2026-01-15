import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from study_plan_logic import generate_study_plan

MODEL_PATH = "study_performance_model.pkl"
DATASET_PATH = "student_study_dataset_150.csv"


def train_model():
    df = pd.read_csv(DATASET_PATH)

    df["weak_subject"] = df["weak_subject"].astype("category").cat.codes

    df["performance"] = pd.cut(
        df["final_score"],
        bins=[0, 50, 75, 100],
        labels=[0, 1, 2]
    ).astype(int)

    X = df[[
        "attendance",
        "study_hours_per_week",
        "quiz_avg",
        "assignment_score",
        "midterm_score",
        "weak_subject",
        "difficulty_level",
        "motivation_level"
    ]]

    y = df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="AI Study Plan Chatbot", layout="wide")
st.title("🎓 AI Student Study Plan & Progress Chatbot")
st.write("Enter your academic details to generate a personalized study plan.")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask something...")

def bot_reply(text):
    if "plan" in text.lower():
        return "Sure! Fill academic details in the left sidebar."
    return "Hi! I generate personalized study plans. Use the form on the left."

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    reply = bot_reply(prompt)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()


st.sidebar.header("📘 Academic Details")

with st.sidebar.form("form"):
    attendance = st.number_input("Attendance (%)", 50, 100, 85)
    study_hours = st.number_input("Study Hours / Week", 1, 40, 8)
    quiz_avg = st.number_input("Quiz Avg (%)", 0, 100, 75)
    assignment_score = st.number_input("Assignment Score (%)", 0, 100, 80)
    midterm_score = st.number_input("Midterm Score (%)", 0, 100, 65)
    weak_subject = st.selectbox("Weak Subject", ["Math", "Programming", "DSA", "Database", "Networking"])
    difficulty = st.slider("Difficulty Level", 1, 5, 3)
    motivation = st.slider("Motivation Level", 1, 5, 3)

    submit = st.form_submit_button("Generate Study Plan")

subject_map = {"Math": 0, "Programming": 1, "DSA": 2, "Database": 3, "Networking": 4}

if submit:
    features = {
        "attendance": attendance,
        "study_hours_per_week": study_hours,
        "quiz_avg": quiz_avg,
        "assignment_score": assignment_score,
        "midterm_score": midterm_score,
        "weak_subject": subject_map[weak_subject],
        "difficulty_level": difficulty,
        "motivation_level": motivation
    }

    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]

    perf = ["Weak", "Average", "Strong"]
    plan = generate_study_plan(prediction, features)

    st.subheader("📊 Predicted Performance")
    st.write(f"**Performance Level:** {perf[prediction]}")

    st.subheader("📝 Personalized Study Plan")
    st.json(plan)
