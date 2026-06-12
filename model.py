import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

FEATURES = ["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE"]

df = pd.read_csv("features.csv")

X = df[FEATURES]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
print()
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

print("Feature importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print()
print("Model saved to classifier.pkl")