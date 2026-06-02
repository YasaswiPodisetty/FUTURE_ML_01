import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Superstore.csv", encoding="latin1")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter

monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

print("\nMonthly Sales Data:")
print(monthly_sales.head())

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Order Date'], monthly_sales['Sales'])
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.savefig("sales_trend.png")
plt.close()

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

print("\nModel Trained Successfully!")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error (MAE):", mae)

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

print("\nForecast for Next 12 Months:")
print(forecast_df)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales'],
    label='Historical Sales'
)

plt.plot(
    forecast_dates,
    future_predictions,
    label='Forecast'
)

plt.title("Sales Forecast for Next 12 Months")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.savefig("sales_forecast.png")
plt.close()

print("\nProject Completed Successfully!")