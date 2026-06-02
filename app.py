import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales & Demand Forecasting Dashboard")
st.subheader("Future Interns - ML Task 1")

df = pd.read_csv("Superstore.csv", encoding="latin1")

st.write("### Dataset Overview")
st.write("Rows and Columns:", df.shape)

df['Order Date'] = pd.to_datetime(df['Order Date'])

monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

monthly_sales['Month_Num'] = range(len(monthly_sales))

X = monthly_sales[['Month_Num']]
y = monthly_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

st.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")

st.write("### Historical Sales Trend")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(monthly_sales['Order Date'], monthly_sales['Sales'])
ax1.set_title("Monthly Sales Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")

st.pyplot(fig1)

future_months = pd.DataFrame({
    'Month_Num': range(
        len(monthly_sales),
        len(monthly_sales) + 12
    )
})

future_predictions = model.predict(future_months)

forecast_dates = pd.date_range(
    start=monthly_sales['Order Date'].max() + pd.offsets.MonthEnd(1),
    periods=12,
    freq='ME'
)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted_Sales': future_predictions
})

st.write("### Future Sales Forecast")

st.dataframe(forecast_df)

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales'],
    label='Historical Sales'
)

ax2.plot(
    forecast_dates,
    future_predictions,
    label='Forecast'
)

ax2.set_title("Sales Forecast for Next 12 Months")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.legend()

st.pyplot(fig2)

st.success("Dashboard Loaded Successfully!")