
---

`
```md
# Troubleshooting

## 1) Training channel name validation
**Error:** channelName failed regex when using `train/x`
**Fix:** use `train_x`, `train_y`, `test_x`, `test_y` (no `/`)

## 2) Training job "Failed" but logs show it reached model upload
Often caused by user script error (ExitCode != 0). Always check CloudWatch logs.

## 3) Endpoint returns 500 ModelError
Usually caused by:
- input_fn parsing (bytes vs str)
- incorrect payload shape (1D vs 2D)
Fix by ensuring CSV parsing returns 2D array and matches model features.

## 4) "Could not find model data at S3..."
Confirm the object exists in S3. Sometimes the job output path is created but `model.tar.gz` was never produced.
