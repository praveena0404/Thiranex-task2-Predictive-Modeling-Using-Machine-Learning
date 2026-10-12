import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

df = pd.read_csv("dataset/breast_cancer_dataset.csv")
X = df.drop(columns=["target"])
y = df["target"].map({"malignant": 0, "benign": 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5)); ax.imshow(cm); ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label"); ax.set_xticks([0,1],["Malignant","Benign"]); ax.set_yticks([0,1],["Malignant","Benign"])
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center")
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png",dpi=160); plt.close(fig)
fpr,tpr,_=roc_curve(y_test,y_prob)
fig, ax = plt.subplots(figsize=(7,5)); ax.plot(fpr,tpr,label=f"Random Forest (AUC = {roc_auc_score(y_test,y_prob):.3f})"); ax.plot([0,1],[0,1],linestyle="--"); ax.set_title("ROC Curve"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(); fig.tight_layout(); fig.savefig("outputs/roc_curve.png",dpi=160); plt.close(fig)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8,5)); importance.sort_values().plot(kind="barh",ax=ax); ax.set_title("Top 10 Feature Importances"); ax.set_xlabel("Importance"); fig.tight_layout(); fig.savefig("outputs/feature_importance.png",dpi=160); plt.close(fig)
