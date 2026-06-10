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
# 2. Sidebar Insights (Including Minor Specifics)
# ----------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.sidebar.header("Dataset Insights")

# Calculations
survived_fraction = (df["Survived"] == 1).sum() / len(df)
minors_df = df[df["Age"] < 18]
minors_fraction = len(minors_df) / len(df)

# Calculate how many minors actually survived in the training data
minors_survival_rate = (minors_df["Survived"] == 1).sum() / len(minors_df)

# Display stats
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
# 4. User Input Interface
# ----------------------------------------------------
st.subheader("Enter Passenger Details to Predict Survival")

col1, col2 = st.columns(2)

with col1:
    passenger_class = st.selectbox(
        "Ticket Class (Pclass)", options=[1, 2, 3], index=2
    )
    gender = st.selectbox("Gender", options=["male", "female"])

    # Notice the age slider helps determine if they are a minor
    age = st.slider("Age", min_value=0, max_value=100, value=12)

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
        value=1,
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

    # Dynamic messaging if the input age is a minor
    if age < 18:
        st.info(f"🧑 **Minor Passenger Detected** (Age: {age})")

    if prediction == 1:
        st.success(
            f"🎉 This passenger likely **SURVIVED**! (Chance: {probability:.1%})"
        )
    else:
        st.error(
            f"😔 This passenger likely **DID NOT SURVIVE**. (Survival Chance: {probability:.1%})"
        )

# Optional: Show filtered data for minors
if st.checkbox("Show Raw Data for Minors Only"):
    # Convert Sex back to readable text for presentation
    display_df = minors_df.copy()
    display_df["Sex"] = display_df["Sex"].map({0: "male", 1: "female"})
    st.dataframe(display_df)
