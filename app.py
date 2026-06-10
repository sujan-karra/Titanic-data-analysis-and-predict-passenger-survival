import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. Load and Prepare Data
# ----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df


df = load_data()

# ----------------------------------------------------
# 2. Sidebar Insights
# ----------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.sidebar.header("Dataset Insights")

survived_fraction = (df["Survived"] == 1).sum() / len(df)
minors_df = df[df["Age"] < 18]
minors_fraction = len(minors_df) / len(df)
minors_survival_rate = (minors_df["Survived"] == 1).sum() / len(minors_df)

st.sidebar.metric(
    label="Overall Survival Rate", value=f"{survived_fraction:.2%}"
)
st.sidebar.metric(label="Fraction of Minors", value=f"{minors_fraction:.2%}")
st.sidebar.metric(
    label="Minor Survival Rate (Actual)", value=f"{minors_survival_rate:.2%}"
)

# ----------------------------------------------------
# 3. Train the Model
# ----------------------------------------------------
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = df[features]
y = df["Survived"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ----------------------------------------------------
# 4. New Option: Select Passenger Type
# ----------------------------------------------------
st.subheader("Passenger Profile")

# This radio button lets the user choose the category directly
passenger_type = st.radio(
    "Select Passenger Age Category:",
    options=["Minor (Under 18)", "Adult (18+)"],
    horizontal=True,
)

st.write("---")
st.subheader("Fine-tune Passenger Details")

col1, col2 = st.columns(2)

with col1:
    passenger_class = st.selectbox(
        "Ticket Class (Pclass)", options=[1, 2, 3], index=2
    )
    gender = st.selectbox("Gender", options=["male", "female"])

    # Dynamically adjust the slider ranges based on the user's selection above
    if passenger_type == "Minor (Under 18)":
        age = st.slider(
            "Age (Locked to Minors)", min_value=0, max_value=17, value=10
        )
    else:
        age = st.slider(
            "Age (Locked to Adults)", min_value=18, max_value=100, value=30
        )

with col2:
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
    )
    # Minors usually travel with parents, so we default this to 1 or 2 if "Minor" is selected
    default_parch = 1 if passenger_type == "Minor (Under 18)" else 0
    parch = st.number_input(
        "Number of Parents/Children Aboard (Parch)",
        min_value=0,
        max_value=10,
        value=default_parch,
    )

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

    if passenger_type == "Minor (Under 18)":
        st.info(f"🧑 **Predicting for a Minor** (Age: {age})")

    if prediction == 1:
        st.success(
            f"🎉 This passenger likely **SURVIVED**! (Chance: {probability:.1%})"
        )
    else:
        st.error(
            f"😔 This passenger likely **DID NOT SURVIVE**. (Survival Chance: {probability:.1%})"
        )
