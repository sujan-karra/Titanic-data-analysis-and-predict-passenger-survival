import pandas as pd
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. Load and Prepare Data
# ----------------------------------------------------
# Cache the data loading so it doesn't reload on every click
@st.cache_data
def load_data():
    # Replacing the Kaggle path with the direct URL for easy running anywhere
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Basic preprocessing for the model
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df


df = load_data()

# ----------------------------------------------------
# 2. Sidebar / Top Statistics (Your original logic)
# ----------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.sidebar.header("Dataset Insights")

# Calculating your exact metrics
survived_fraction = (df["Survived"] == 1).sum() / len(df)
minors_fraction = (df["Age"] < 18).sum() / len(df)

st.sidebar.metric(
    label="Overall Survival Rate", value=f"{survived_fraction:.2%}"
)
st.sidebar.metric(label="Fraction of Minors", value=f"{minors_fraction:.2%}")

# ----------------------------------------------------
# 3. Train a Quick Model
# ----------------------------------------------------
# Simple feature selection for demo purposes
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = df[features]
y = df["Survived"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ----------------------------------------------------
# 4. User Input Interface
# ----------------------------------------------------
st.subheader("Enter Passenger Details to Predict Survival")

col1, col2 = st.columns(2)

with col1:
    passenger_class = st.selectbox(
        "Ticket Class (Pclass)", options=[1, 2, 3], index=2
    )
    gender = st.selectbox("Gender", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=100, value=25)

with col2:
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
    )
    parch = st.number_input(
        "Number of Parents/Children Aboard (Parch)",
        min_value=0,
        max_value=10,
        value=0,
    )

# Map gender text back to numeric for the model
gender_numeric = 1 if gender == "female" else 0

# Create input dataframe for prediction
input_data = pd.DataFrame(
    [[passenger_class, gender_numeric, age, sibsp, parch]], columns=features
)

# ----------------------------------------------------
# 5. Prediction and Output
# ----------------------------------------------------
if st.button("Predict Survival Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")
    if prediction == 1:
        st.success(
            f"🎉 This passenger likely **SURVIVED**! (Chance: {probability:.1%})"
        )
    else:
        st.error(
            f"😔 This passenger likely **DID NOT SURVIVE**. (Survival Chance: {probability:.1%})"
        )

# Optional: Show raw data if the user wants to see it
if st.checkbox("Show Raw Titanic Dataset"):
    st.dataframe(df)
