import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the already-processed dataset
df = pd.read_csv(r"C:\Users\Charuta\Codes\Python\Health Screen\Data\dataset_processed.csv.csv")

# Separate features and target
X = df.drop(columns=["Deficiency"])
y = df["Deficiency"]

# Fit label encoder on the numeric target (0-6 → deficiency names)
# Since Deficiency is already numeric, we create a manual mapping instead
DEFICIENCY_NAMES = {0: "Calcium", 1: "Folate", 2: "Iron", 3: "Magnesium", 4: "Vitamin B12", 5: "Vitamin D", 6: "Zinc"}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
print("Model trained!")
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
print(classification_report(y_test, model.predict(X_test)))

# Save model and label map
with open("nutriscreen_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(DEFICIENCY_NAMES, f)

print("✅ nutriscreen_model.pkl saved")
print("✅ label_encoder.pkl saved")