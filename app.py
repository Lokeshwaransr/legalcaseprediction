import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load Model and Vectorizer ---
@st.cache_data
def load_model():
    model = joblib.load('model.pkl')  # Load the trained model
    vectorizer = joblib.load('vectorizer.pkl')  # Load the TF-IDF vectorizer
    return model, vectorizer

model, vectorizer = load_model()

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("land_cases_dataset_1000.csv")
    return df

data = load_data()

# --- Session State Variables ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "details_submitted" not in st.session_state:
    st.session_state.details_submitted = False
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# --- Pages ---
def login_page():
    st.title("🔒 Admin Login")

    admin_id = st.text_input("Admin ID")
    admin_pass = st.text_input("Password", type="password")

    if st.button("Login"):
        if admin_id == "manikandan" and admin_pass == "6379039339":
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Incorrect ID or Password.")

def user_input_page():
    st.title("📄 Enter Case Details")

    name = st.text_input("Enter Name")
    patta_no = st.text_input("Enter Patta Number")
    survey_no = st.text_input("Enter Survey Number")

    if st.button("Submit"):
        st.session_state.user_data['name'] = name
        st.session_state.user_data['patta_no'] = patta_no
        st.session_state.user_data['survey_no'] = survey_no
        st.session_state.details_submitted = True
        st.success("Details submitted successfully!")

def prediction_page():
    st.title("📝 Case Text Analysis and Prediction")

    input_text = st.text_area("Enter Legal Document Text:")

    if st.button("Analyze"):
        if input_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            # --- Predict the Outcome ---
            input_vec = vectorizer.transform([input_text])  # Vectorize input text
            prediction = model.predict(input_vec)  # Predict using the trained model
            prediction_result = prediction[0]  # Get prediction result

            # --- Display Results ---
            result = f"""
            Name: {st.session_state.user_data['name']}
            Patta No: {st.session_state.user_data['patta_no']}
            Survey No: {st.session_state.user_data['survey_no']}
            Prediction: {prediction_result}
            """
            st.text_area("Prediction Result:", value=result, height=200)

# --- Main Control ---
def main():
    if not st.session_state.logged_in:
        login_page()
    elif not st.session_state.details_submitted:
        user_input_page()
    else:
        prediction_page()

if __name__ == "__main__":
    main()
