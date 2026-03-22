import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("./gdp_per_capita.csv")

# Fix column names
df.columns = df.columns.str.strip()

gdp_per_capita = df["GDP per capita (USD)"]
life_expectancy = df["Life expectancy"]

# Reshape (QUAN TRỌNG trong sklearn)
X = gdp_per_capita.values.reshape(-1, 1)
y = life_expectancy.values.reshape(-1, 1)

# Plot
plt.figure(figsize=(16, 8))
sns.scatterplot(x=gdp_per_capita, y=life_expectancy)

plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life Expectancy")
plt.title("GDP vs Life Expectancy")

plt.show()

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
X_new = np.array([[22587]])
prediction = model.predict(X_new)

print("Prediction (Life Expectancy):", prediction)