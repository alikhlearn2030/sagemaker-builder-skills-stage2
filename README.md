# SageMaker Builder Skills – Stage 2 (Titanic) | المرحلة 2 (تيتانيك)

## 🇸🇦 ماذا أنجزت؟
قمت ببناء تدفق عمل ML عملي داخل **Amazon SageMaker Unified Studio** يشمل:
- تحميل بيانات Titanic (CSV)
- تجهيز البيانات (تعويض القيم الناقصة + One-Hot Encoding)
- تدريب نموذج Logistic Regression محليًا (scikit-learn)
- حفظ النموذج كـ **Model Artifact** ورفعه إلى **Amazon S3**
- تجهيز Artifact متوافق مع SageMaker (`model.tar.gz`)
- إنشاء كائن `SKLearnModel` لتجارب الاستضافة والنشر

### مخرجات مهمة
- `src/train.py`: سكربت تدريب مناسب لـ SageMaker Training Jobs (Channels: train_x, train_y, test_x, test_y)
- `src/inference.py`: سكربت الاستدلال (Inference) لاستقبال CSV
- `model.tar.gz`: Model Artifact (تم رفعه إلى S3)

### ملاحظات/دروس مستفادة
- أسماء قنوات التدريب في SageMaker لا تقبل `/` (استخدم `train_x` بدل `train/x`).
- أخطاء 500 في Endpoint غالبًا سببها parsing في `input_fn` أو شكل البيانات (1D vs 2D).
- حذف الـ Endpoints بعد الاختبار مباشرة ضروري لتجنب أي تكلفة مستمرة.

### الخطوات التالية
- Batch Transform (بدون Endpoint دائم)
- Hyperparameter Tuning
- توثيق MLOps (Model Registry + Pipelines)

---

## 🇺🇸 English — What I built
An end-to-end ML workflow in **Amazon SageMaker Unified Studio**:
- Loaded the Titanic dataset (CSV)
- Preprocessed data (missing values + one-hot encoding)
- Trained a Logistic Regression model locally (scikit-learn)
- Saved model artifacts and uploaded them to **Amazon S3**
- Packaged a SageMaker-compatible artifact (`model.tar.gz`)
- Created a `SKLearnModel` object for hosting/deployment experiments

### Key artifacts
- `src/train.py`: training script for SageMaker training jobs (channels: train_x, train_y, test_x, test_y)
- `src/inference.py`: inference handler for CSV payloads

### Lessons learned
- Training channel names cannot contain `/` (use `train_x` not `train/x`).
- Endpoint 500 errors typically come from input parsing / payload shape mismatches.
- Always delete endpoints after testing to avoid ongoing charges.

### Next steps
- Batch Transform inference (no persistent endpoint)
- Hyperparameter tuning
- MLOps: Model Registry + Pipelines
