
"""# logistic Regression"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

from sklearn.linear_model import LogisticRegression

logistic_pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("model",LogisticRegression(max_iter=1000,n_jobs=-1))
])
logistic_pipeline.fit(x_train,y_train)

y_pred=logistic_pipeline.predict(x_test)
y_prob = logistic_pipeline.predict_proba(x_test)[:,1]
y_prob

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred);
plt.figure(figsize=(6, 4));
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.xlabel("Predicted");
plt.ylabel("Actual");
plt.title("Logistic Regression - Confusion Matrix");
plt.show()

results = x_test.copy()
results["churn_probability"] = y_prob
results["risk_category"] = pd.cut(results["churn_probability"],bins=[0, 0.4, 0.7, 1],labels=["Low", "Medium", "High"],include_lowest=True)

"""# Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=10,min_samples_split=10,min_samples_leaf=5,random_state=42,class_weight="balanced"))
  ])

decision_tree_pipeline.fit(x_train, y_train)

y_pred_dt = decision_tree_pipeline.predict(x_test)
y_prob_dt = decision_tree_pipeline.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1 Score:", f1_score(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_dt))

print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt,annot=True,fmt="d",cmap="Blues",xticklabels=["Stayed", "Churned"],yticklabels=["Stayed", "Churned"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

dt_results = x_test.copy()

dt_results["actual_churn"] = y_test.values
dt_results["churn_probability"] = y_prob_dt
dt_results["predicted_churn"] = y_pred_dt

dt_results["risk_category"] = pd.cut(dt_results["churn_probability"],bins=[0, 0.4, 0.7, 1],labels=["Low", "Medium", "High"],include_lowest=True)
dt_results.head()

feature_names = decision_tree_pipeline.named_steps[
      "preprocessor"
      ].get_feature_names_out()
importances = decision_tree_pipeline.named_steps["model"].feature_importances_
feature_importance_dt = pd.DataFrame({
      "feature": feature_names,
      "importance": importances
}).sort_values("importance", ascending=False)

feature_importance_dt.head(20)

plt.figure(figsize=(10, 7))

sns.barplot(data=feature_importance_dt.head(15),x="importance",y="feature")

plt.title("Decision Tree - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

"""# random forest"""

from sklearn.ensemble import RandomForestClassifier

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
            n_estimators=10,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
          ))
  ])

random_forest_pipeline.fit(x_train, y_train)

y_pred_rf = random_forest_pipeline.predict(x_test)
y_prob_rf = random_forest_pipeline.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

rf_results = x_test.copy()

rf_results["actual_churn"] = y_test.values
rf_results["churn_probability"] = y_prob_rf
rf_results["predicted_churn"] = y_pred_rf

rf_results["risk_category"] = pd.cut(rf_results["churn_probability"],bins=[0, 0.4, 0.7, 1],labels=["Low", "Medium", "High"],)

rf_results.head()

feature_names_rf = random_forest_pipeline.named_steps[
      "preprocessor"
      ].get_feature_names_out()
importances_rf = random_forest_pipeline.named_steps[
      "model"
      ].feature_importances_
feature_importance_rf = pd.DataFrame({
      "feature": feature_names_rf,
          "importance": importances_rf
          }).sort_values("importance", ascending=False)
feature_importance_rf.head(20)

plt.figure(figsize=(10, 7))

sns.barplot(data=feature_importance_rf.head(15),x="importance",y="feature")

plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

"""# SVM"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

X_train_svm, _, y_train_svm, _ = train_test_split(x_train,y_train,train_size=100000,random_state=42)

svm_pipeline = Pipeline([
      ("preprocessor", preprocessor),
      ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
])
svm_pipeline.fit(X_train_svm, y_train_svm)

y_pred_svm = svm_pipeline.predict(x_test)
y_prob_svm = svm_pipeline.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm))

print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
    )

plt.xlabel("Predicted")
plt.title("SVM - Confusion Matrix")
plt.show()

svm_results = x_test.copy()

svm_results["actual_churn"] = y_test.values
svm_results["churn_probability"] = y_prob_svm
svm_results["predicted_churn"] = y_pred_svm

svm_results["risk_category"] = pd.cut(
    svm_results["churn_probability"],bins=[0, 0.4, 0.7, 1],labels=["Low", "Medium", "High"],include_lowest=True
)

"""# XGBoost"""

from xgboost import XGBClassifier

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
          n_estimators=200,
          max_depth=6,
          learning_rate=0.1,
          subsample=0.8,
          colsample_bytree=0.8,
          random_state=42,
          n_jobs=-1,
          eval_metric="logloss"
        ))
])
xgb_pipeline.fit(x_train, y_train)

y_pred_xgb = xgb_pipeline.predict(x_test)
y_prob_xgb = xgb_pipeline.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

print(classification_report(y_test, y_pred_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
  )

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost - Confusion Matrix")

feature_names_xgb = xgb_pipeline.named_steps[
      "preprocessor"
      ].get_feature_names_out()

importances_xgb = xgb_pipeline.named_steps["model"].feature_importances_
feature_importance_xgb = pd.DataFrame({
      "feature": feature_names_xgb,
      "importance": importances_xgb
}).sort_values("importance", ascending=False)

feature_importance_xgb.head(20)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance_xgb.head(15),
        x="importance",
        y="feature"
  )

plt.title("XGBoost - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

xgb_results = x_test.copy()

xgb_results["actual_churn"] = y_test.values
xgb_results["churn_probability"] = y_prob_xgb
xgb_results["predicted_churn"] = y_pred_xgb

xgb_results["risk_category"] = pd.cut(
    xgb_results["churn_probability"],
    bins=[0, 0.4, 0.7, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True
  )
xgb_results.head()