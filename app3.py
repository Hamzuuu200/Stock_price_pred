import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# Load model + columns + data
# ======================
model = pickle.load(open("Model2/model1.pkl", "rb"))
columns = pickle.load(open("Model2/columns.pkl", "rb"))
data = pickle.load(open("Model2/data.pkl", "rb"))   # cleaned dataset

st.title("📈 Stock Price Predictor")

st.write("Enter stock details below to predict the **Close Price**:")

# ======================
# Create input fields
# ======================
Open = st.number_input("Open Price", value=500.0)
High = st.number_input("High Price", value=510.0)
Low = st.number_input("Low Price", value=495.0)
AdjClose = st.number_input("Adj Close Price", value=505.0)
Volume = st.number_input("Volume", value=1000000)

Year = st.number_input("Year", value=2025)
Month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=9)
Day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=17)
DayOfWeek = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

# ======================
# Prepare input DataFrame
# ======================
user_input = pd.DataFrame([[Open, High, Low, AdjClose, Volume, Year, Month, Day, DayOfWeek]], 
                          columns=[col for col in columns if col != "Close"])

# ======================
# Prediction button
# ======================
if st.button("Predict Close Price"):
    prediction = model.predict(user_input)
    st.success(f"📊 Predicted Close Price: {prediction[0]:.2f}")

# ======================
# Seaborn plot section
# ======================
st.subheader("📊 Historical Stock Prices")

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=data, x="Year", y="Close", ax=ax)

ax.set_title("Close Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
st.pyplot(fig)


