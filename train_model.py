import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("mood_music_dataset.csv")

print("Dataset Shape:", df.shape)

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le_mood = LabelEncoder()
le_time = LabelEncoder()

df["Mood"] = le_mood.fit_transform(df["Mood"])
df["Time of Day"] = le_time.fit_transform(df["Time of Day"])

# Features and target
X = df.drop("Mood", axis=1)
y = df["Mood"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_mood, "label_encoder.pkl")
joblib.dump(le_time, "time_encoder.pkl")

print("\nModel saved successfully.")