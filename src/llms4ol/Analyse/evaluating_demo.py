from evaluate import EvaluationMetrics

y_true=["yes","yes","yes","yes","yes"]
y_pred=["yes","no","yes","no","yes"]
results = EvaluationMetrics.evaluate(actual=y_true, predicted=y_pred)
print("F1-score:", results['clf-report-dict']['macro avg']['f1-score'])