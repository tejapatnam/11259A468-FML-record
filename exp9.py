
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("data.csv")

# Display the original dataset
print("Original Dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing numerical values with the column mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select numerical columns for normalization
columns = ['Hours_Studied', 'Attendance', 'Assignments', 'Marks']
X = data[columns].copy()

# Normalize the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the normalized data into a DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=columns, index=data.index)

# Display the normalized dataset
print("\nNormalized Dataset:")
print(X_scaled.head())
