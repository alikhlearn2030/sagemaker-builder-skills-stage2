# Cost Safety / إيقاف الفوترة

## Golden rules
- Endpoint = billing continues while it exists (even with zero requests)
- Notebook/Compute instance = billing continues while running

## After finishing a session
### In Studio
- Shut down kernels (Notebook)
- Stop compute / instance

### If you deployed an endpoint
- Delete endpoint immediately after testing

## Quick Python cleanup
```python
# If you have a predictor object:
predictor.delete_endpoint()
