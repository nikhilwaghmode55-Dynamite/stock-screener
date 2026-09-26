import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

FEATURES = ["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE", "Return_3d", "Price_vs_EMA20", "Volatility_10d"]

df = pd.read_csv("features.csv")

X = df[FEATURES]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# High-performance Gradient Boosting Classifier to push accuracy past 60%
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"New Test accuracy: {accuracy:.2%}")
print()
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Optimized model saved to classifier.pkl")