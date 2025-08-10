
# Bumblebeat — Heart-Sound Stability (v2.1)

**Wellness feedback only — not diagnosis.** You vs You with:
- RMS loudness normalization
- Autocorr‑guided cycle segmentation
- Template whitening residual with baseline calibration
- Murmur band (150–600 Hz) delta
- Stricter, calibrated thresholds

## Files
- `bumblebeat_streamlit_app_v2_1.py`
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run bumblebeat_streamlit_app_v2_1.py
```

## Deploy on Streamlit Cloud
1. Push files to GitHub.
2. Streamlit → Community Cloud → New app → main file `bumblebeat_streamlit_app_v2_1.py`.
3. Deploy.
