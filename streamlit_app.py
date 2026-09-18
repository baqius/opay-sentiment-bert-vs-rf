import streamlit as st
from preprocessing import preprocessor
import joblib
from PIL import Image

# Load the logo image
logo = Image.open("opay image.jpg")

st.set_page_config(page_title="Opay Sentiment Classifier", page_icon=logo)

# Load model, vectorizer, and encoder once
@st.cache_resource
def load_artifacts():
    rf = joblib.load('rf_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return rf, vectorizer, encoder

rf, vectorizer, encoder = load_artifacts()

# Display logo alongside the title
col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo, width=250)
with col2:
    st.markdown(
        "<h1 style='color:#1B1B4B; font-weight:800;'>"
        "Review Sentiment Classifier"
        "</h1>",
        unsafe_allow_html=True
    )

st.write("Enter a review to predict its sentiment.")

user_input = st.text_area("Review text:", height=120)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocessor(user_input)
        text_tfidf = vectorizer.transform([cleaned])
        pred = rf.predict(text_tfidf)
        pred_label = encoder.inverse_transform(pred)[0]

        proba = rf.predict_proba(text_tfidf)[0]

        st.success(f"Predicted sentiment: **{pred_label.capitalize()}**")

        st.write("Confidence breakdown:")
        for label, prob in zip(encoder.classes_, proba):
            st.write(f"- {label.capitalize()}: {prob:.1%}")

        if max(proba) < 0.5:
            st.info("This review has mixed or ambiguous sentiment — the model has lower confidence here.")

        if pred_label == "positive":
            st.write("😊 Thank you for the kind words! We're glad Opay is working well for you.")
        elif pred_label == "negative":
            st.write("🙏 We're sorry to hear about your experience. Thank you for the honest feedback — issues like this are how the app gets better. Consider reaching out to Opay's support team directly for a faster resolution.")
        else:  # neutral
            st.write("🙏 Thanks for sharing your thoughts — feedback like this, both good and bad, genuinely helps identify what to improve.")