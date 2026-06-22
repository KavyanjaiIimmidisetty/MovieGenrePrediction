import streamlit as st
import joblib

# Load model
model = joblib.load("models/genre_model.pkl")

# Page settings
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 Movie Genre Predictor")
st.markdown(
    """
    Predict the genre of a movie using **Natural Language Processing (NLP)** and **Machine Learning**.

    **Dataset:** IMDb Genre Classification Dataset (54K+ movie plots)
    """
)

st.divider()

# Example plots
with st.expander("📌 Example Plots"):
    st.write("**Sci-Fi:** An astronaut travels through a wormhole to save humanity.")
    st.write("**Thriller:** A detective investigates a mysterious murder in a small town.")
    st.write("**Romance:** Two young people fall in love despite family opposition.")

# User input
plot = st.text_area(
    "✍️ Enter Movie Plot",
    placeholder="Example: A detective investigates a mysterious murder in a small town...",
    height=200
)
st.info(
    "Note: Some movies can belong to multiple genres. The model predicts the most likely single genre."
)
st.sidebar.title("📊 Dataset")
st.sidebar.write("54K+ Movie Plots")
st.sidebar.write("27 Genres")
st.sidebar.write("Accuracy: 58.76%")

# Predict button
if st.button("🎯 Predict Genre"):

    if plot.strip() == "":
        st.warning("⚠️ Please enter a movie plot.")
    else:
        with st.spinner("Analyzing plot..."):
            prediction = model.predict([plot])

        st.success(f"🎬 Predicted Genre: **{prediction[0].title()}**")

        st.subheader("📖 Your Plot")
        st.write(plot)

st.divider()

st.caption(
    "Built using Python, Scikit-learn, TF-IDF, LinearSVC, and Streamlit."
)