
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

model = load_model()

# Application title
st.title("🔍 Fake Review Detection System")

st.write(
    "Enter a product review to predict whether it "
    "matches the fake-labelled or honest-labelled class."
)

st.divider()

# User inputs
review_title = st.text_input("Product Review Title")

review_text = st.text_area(
    "Enter Product Review",
    height=180,
    placeholder="Write or paste your product review here..."
)

# Prediction button
if st.button("Analyze Review", type="primary"):

    if not review_text.strip():
        st.warning("Please enter a review first.")

    else:
        # Combine title and review text
        combined_text = (
            review_title.strip() + " " + review_text.strip()
        ).strip()

        # Predict using trained model
        prediction = model.predict([combined_text])[0]

        st.divider()

        if prediction == 1:
            st.error("Prediction: Fake-labelled Review")
        else:
            st.success("Prediction: Honest-labelled Review")

        st.caption(
            "This prediction reflects patterns learned from "
            "the dataset's reviewer annotations. It does not "
            "independently verify whether a review is fraudulent."
        )

st.divider()

st.caption("Machine Learning Mini Project | TF-IDF + Logistic Regression")
