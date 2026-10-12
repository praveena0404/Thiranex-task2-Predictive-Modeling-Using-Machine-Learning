# Predictive Modeling Using Machine Learning

## Objective
Build a supervised machine learning model to predict outcomes from data and evaluate its performance.

## Algorithm
Random Forest Classifier

## Dataset
Breast Cancer Wisconsin Diagnostic Dataset. The target classes are malignant and benign. The complete CSV is included in the `dataset` folder.

## Workflow
Dataset → Preprocessing → Train/Test Split → Feature Scaling → Random Forest → Prediction → Evaluation → Visualization

## Evaluation Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix and ROC Curve.

## Results
Accuracy: 0.9561
Precision: 0.9589
Recall: 0.9722
F1-score: 0.9655
ROC-AUC: 0.9932

## Project Structure
```text
Predictive_Modeling_ML/
├── dataset/
│   └── breast_cancer_dataset.csv
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── predictive_model.py
├── model_results.txt
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python predictive_model.py
```

## Expected Outcome
Gain practical experience in supervised learning, classification, model training, testing, evaluation and visualization.

## Conclusion
The Random Forest model provides a complete predictive modeling workflow, with quantitative metrics and visual outputs to understand classification performance.
