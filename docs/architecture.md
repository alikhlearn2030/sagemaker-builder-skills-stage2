# Architecture (Stage 2)

## Workflow
1) Load dataset (Titanic CSV)
2) Preprocess:
   - Missing values (Age median, Embarked mode)
   - One-hot encoding (Sex, Embarked)
3) Train (Local scikit-learn):
   - Logistic Regression
4) Artifacts:
   - Save model (joblib)
   - Package `model.tar.gz` (SageMaker-compatible)
5) Storage:
   - Upload data + artifacts to Amazon S3
6) Hosting experiments:
   - Create `SKLearnModel` object (for deployment / batch later)

## AWS Services used
- Amazon SageMaker Unified Studio
- Amazon S3
- (Optional later) SageMaker Hosting / Batch Transform / Tuning

