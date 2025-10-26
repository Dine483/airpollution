# run_analysis.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 300

df = pd.DataFrame({
    "PM2.5": np.random.randint(20, 150, n),
    "PM10": np.random.randint(30, 200, n),
    "NO2": np.random.randint(10, 80, n),
    "SO2": np.random.randint(5, 40, n),
    "O3": np.random.randint(15, 100, n),
    "Temperature": np.random.randint(20, 40, n),
    "Humidity": np.random.randint(40, 90, n),
})

df["RespiratoryCases"] = (
    0.3 * df["PM2.5"] +
    0.2 * df["PM10"] +
    0.25 * df["NO2"] +
    0.15 * df["SO2"] -
    0.1 * df["O3"] +
    0.05 * df["Temperature"] -
    0.03 * df["Humidity"] +
    np.random.normal(0, 10, n)
).round().astype(int)
df["RespiratoryCases"] = df["RespiratoryCases"].clip(lower=0)

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
df.to_csv("data/air_resp_data.csv", index=False)
print("Saved synthetic dataset to data/air_resp_data.csv (first 5 rows):")
print(df.head())

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("plots/corr_matrix.png")
plt.close()

X = df.drop(columns=['RespiratoryCases'])
y = df['RespiratoryCases']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\\nModel performance on test set:")
print("MSE:", round(mean_squared_error(y_test, y_pred),2))
print("R2 :", round(r2_score(y_test, y_pred),3))

coefs = pd.DataFrame({'feature': X.columns, 'coef': model.coef_}).sort_values(by='coef', key=abs, ascending=False)
print("\\nCoefficients:")
print(coefs)

plt.figure(figsize=(6,4))
sns.barplot(x='coef', y='feature', data=coefs)
plt.title("Feature coefficients")
plt.tight_layout()
plt.savefig("plots/coefficients.png")
plt.close()