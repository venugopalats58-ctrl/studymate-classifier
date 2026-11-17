import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StudyMate AI Classifier",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_FILE = "studymate_education_classifier.pkl"


# =========================================================
# LOAD ML MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the ML model (`studymate_education_classifier.pkl`).")
    st.code(str(e))
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #0D1B4B 50%, #07111f 100%);
    }

    .block-container {
        max-width: 1150px;
        padding-top: 30px;
        padding-bottom: 50px;
    }

    .hero {
        background: linear-gradient(135deg, #0D1B4B, #1A56B0);
        padding: 30px;
        border-radius: 20px;
        border-bottom: 3px solid #00C9A7;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
        margin-bottom: 25px;
    }

    .badge {
        display: inline-block;
        background: rgba(0, 201, 167, 0.12);
        border: 1px solid rgba(0, 201, 167, 0.35);
        color: #00C9A7;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        color: white;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #CBD5E1;
        font-size: 15px;
        line-height: 1.7;
        max-width: 850px;
    }

    .section-title {
        color: white;
        font-size: 22px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        padding: 35px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📚 StudyMate ML")
    st.caption("Education Query Classifier")
    st.divider()

    st.markdown("### 🧠 ML Pipeline")
    st.write("📝 Student Prompt")
    st.write("↓")
    st.write("🔢 TF-IDF Vectorizer")
    st.write("↓")
    st.write("🤖 Logistic Regression (1,000 samples)")
    st.write("↓")
    st.write("📊 Class & Probability Output")

    st.divider()

    st.markdown("### 🎓 Supported Subjects")
    st.write("📐 Mathematics")
    st.write("⚛️ Physics & Chemistry")
    st.write("🧬 Biology")
    st.write("💻 Computer Science & AI")
    st.write("📊 Economics & History")
    st.write("🌍 Geography & English")

    st.divider()

    st.success("🟢 ML Model Loaded & Ready")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">🤖 MACHINE LEARNING CLASSIFIER</div>
        <div class="hero-title">📚 StudyMate Query Classifier</div>
        <div class="hero-text">
            Enter any prompt to classify whether it is <b>Education-related</b> or
            <b>Non-Education</b> using TF-IDF feature extraction and Logistic Regression.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

st.markdown('<div class="section-title">📊 Model Dashboard</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🧠 Algorithm", "LogReg")

with col2:
    st.metric("🤖 Dataset Size", "1,000")

with col3:
    st.metric("💬 Predictions", st.session_state.question_count)

with col4:
    st.metric("🟢 Status", "Ready")


# =========================================================
# QUICK TEST PROMPTS
# =========================================================

st.markdown('<div class="section-title">⚡ Sample Prompts</div>', unsafe_allow_html=True)

quick_questions = [
    "What is the Pythagorean theorem?",
    "What are Newton's three laws?",
    "How does photosynthesis work?",
    "What is Machine Learning?",
    "Book me a hotel",
    "Order food for me",
    "Who won the cricket match?",
    "Tell me a joke"
]

quick_cols = st.columns(4)

for i, q_text in enumerate(quick_questions):
    with quick_cols[i % 4]:
        if st.button(q_text, key=f"quick_{i}", use_container_width=True):
            st.session_state.quick_question = q_text
            st.rerun()


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.markdown('<div class="section-title">💬 Prediction Results</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# =========================================================
# INPUT
# =========================================================

typed_question = st.chat_input("Enter a question or sentence to classify...")
quick_question = st.session_state.pop("quick_question", None)
question = typed_question or quick_question


# =========================================================
# PROCESS & PREDICT OUTPUT
# =========================================================

if question:
    question = question.strip()
    if not question:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.question_count += 1

    with st.chat_message("user", avatar="🎓"):
        st.markdown(question)

    try:
        prediction = model.predict([question])[0]
        prediction_text = str(prediction).strip()

        edu_prob = 0.0
        non_edu_prob = 0.0
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([question])[0]
            classes = [str(c).lower().strip() for c in model.classes_]

            for cls_name, prob in zip(classes, probabilities):
                if cls_name == "education":
                    edu_prob = prob * 100
                else:
                    non_edu_prob = prob * 100

            confidence = max(probabilities) * 100

    except Exception as e:
        with st.chat_message("assistant", avatar="🤖"):
            st.error("❌ ML model prediction failed.")
            st.code(str(e))
        st.stop()

    normalized_prediction = prediction_text.lower().replace("_", " ").strip()
    is_education = normalized_prediction in {
        "education",
        "educational",
        "education related",
        "education-related",
        "edu",
        "1",
        "true"
    }

    with st.chat_message("assistant", avatar="🤖"):
        if is_education:
            st.success("✅ Predicted Output: **Education**")
            category_label = "Education"
            status_text = "Allowed (Academic Query)"
        else:
            st.warning("⚠️ Predicted Output: **Non-Education**")
            category_label = "Non-Education"
            status_text = "Filtered (Non-Academic Query)"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Class", category_label)
        with m2:
            st.metric("Confidence", f"{confidence:.2f}%")
        with m3:
            st.metric("Decision", "Allowed" if is_education else "Blocked")

        # Class probability breakdown table
        prob_df = pd.DataFrame({
            "Category": ["Education", "Non-Education"],
            "Probability (%)": [round(edu_prob, 2), round(non_edu_prob, 2)]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        summary_md = (
            f"**Input:** {question}\n\n"
            f"- **Predicted Label:** `{prediction_text}` ({category_label})\n"
            f"- **Confidence:** `{confidence:.2f}%`\n"
            f"- **Status:** {status_text}"
        )
        st.session_state.messages.append({"role": "assistant", "content": summary_md})


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        📚 <b>StudyMate ML Classifier</b><br>
        TF-IDF + Logistic Regression (1,000 Training Samples)
    </div>
    """,
    unsafe_allow_html=True
)