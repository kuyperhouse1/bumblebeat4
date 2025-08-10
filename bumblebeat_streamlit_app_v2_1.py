
import streamlit as st
import numpy as np
import base64, io, wave, contextlib, json, math

st.set_page_config(page_title="Bumblebeat — Heart-Sound Stability (v2.1)", page_icon="🐝", layout="centered")
st.title("Bumblebeat — Heart-Sound Stability (v2.1)")
st.caption("Wellness feedback only — not diagnosis. You vs You with cycle segmentation + whitening + calibration.")

# ---------------- WAV decoding & loudness normalization ----------------
def parse_wav_bytes(data: bytes):
    bio = io.BytesIO(data)
    with contextlib.closing(wave.open(bio, 'rb')) as wf:
        ch = wf.getnchannels(); sr = wf.getframerate(); sw = wf.getsampwidth(); n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128; full = 127.0
    elif sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.int32); full = 32767.0
    elif sw == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1,3)
        b = (a[:,0].astype(np.int32) | (a[:,1].astype(np.int32)<<8) | (a[:,2].astype(np.int32)<<16))
        sign = (b & 0x800000) != 0; b = b - (1<<24)*sign; arr = b; full = (1<<23) - 1
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.int64); full = (1<<31) - 1
    else:
        return None, None
    if ch > 1: arr = arr.reshape(-1, ch).mean(axis=1)
    x = (arr / full).astype(np.float32)
    # RMS normalization to target ~ -20 dBFS (rms ~ 0.1)
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    target = 0.1
    gain = target / rms
    # Avoid clipping
    peak = float(np.max(np.abs(x)) + 1e-12)
    max_gain = 0.98 / peak
    gain = min(gain, max_gain)
    x = (x * gain).astype(np.float32)
    return x, int(sr)

# ---------------- FFT helpers ----------------
def bandpass_fft(x, sr, f1, f2):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    mask = (freqs >= f1) & (freqs <= f2)
    Xf = np.zeros_like(X); Xf[mask] = X[mask]
    y = np.fft.irfft(Xf, n=N).astype(np.float32)
    return y

def moving_avg(x, win):
    if win <= 1: return x.copy()
    k = int(win)
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, ker, mode='same')

# ---------------- Autocorr-guided cycle segmentation ----------------
def estimate_rr(env, sr):
    # Autocorrelation within plausible HR range 40-140 bpm
    min_bpm, max_bpm = 40, 140
    min_lag = int(sr * 60 / max_bpm)
    max_lag = int(sr * 60 / min_bpm)
    env_z = env - np.mean(env)
    ac = np.correlate(env_z, env_z, mode='full')[len(env_z)-1:]
    ac[:min_lag] = -np.inf
    if max_lag < len(ac):
        ac[max_lag:] = -np.inf
    lag = int(np.nanargmax(ac[min_lag:max_lag]) + min_lag)
    return max(lag, 1)

def segment_cycles(x, sr):
    # Emphasize 25–120 Hz for S1/S2, smoother env
    lf = bandpass_fft(x, sr, 25.0, 120.0)
    env = np.abs(lf)
    env = moving_avg(env, int(0.02*sr))  # 20 ms
    # Adaptive threshold and refractory based on autocorr RR
    rr = estimate_rr(env, sr)
    if rr <= 0: return []
    th = np.percentile(env, 70)
    peaks = []
    i = 1; N = len(env)
    min_gap = max(int(0.6*rr), int(0.25*sr))   # at least 0.25s and 60% of rr
    max_gap = int(1.6*rr)
    while i < N-1:
        if env[i] > th and env[i] >= env[i-1] and env[i] >= env[i+1]:
            if not peaks or (i - peaks[-1] > min_gap):
                peaks.append(i)
                i += min_gap
                continue
        i += 1
    # prune peaks that are too close/far via rr bounds
    pk = [peaks[0]] if peaks else []
    for p in peaks[1:]:
        if (p - pk[-1]) < min_gap:  # too close, keep the higher env
            if env[p] > env[pk[-1]]: pk[-1] = p
        elif (p - pk[-1]) <= max_gap:
            pk.append(p)
        else:
            pk.append(p)  # allow skips for arrhythmia; we'll resample anyway
    # build cycles
    cycles = []
    for a, b in zip(pk[:-1], pk[1:]):
        if b <= a: continue
        seg = x[a:b]
        L = 512
        idx = np.linspace(0, len(seg)-1, L)
        seg_r = np.interp(idx, np.arange(len(seg)), seg).astype(np.float32)
        # per-cycle energy normalize
        e = float(np.sqrt(np.mean(seg_r**2)) + 1e-9)
        seg_r = seg_r / e
        cycles.append(seg_r)
    return cycles

# ---------------- Template whitening ----------------
def build_template(cycles):
    if not cycles: return None
    A = np.stack(cycles, axis=0)
    template = np.median(A, axis=0).astype(np.float32)  # median is robust
    template = template / (np.linalg.norm(template) + 1e-9)
    return template

def residual_ratio(cycles, template):
    if template is None or not cycles: return None
    T = template / (np.linalg.norm(template) + 1e-9)
    ratios = []
    for c in cycles:
        coef = float(np.dot(c, T))
        proj = coef * T
        num = float(np.sum((c - proj)**2))
        den = float(np.sum(c**2) + 1e-12)
        ratios.append(num/den)
    return float(np.median(ratios)) if ratios else None

# ---------------- Murmur ratio and spectral embedding ----------------
def band_energy_ratio(x, sr, band_lo, band_hi, ref_lo=20.0, ref_hi=600.0):
    N = len(x)
    X = np.fft.rfft(x); P = np.abs(X)**2
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    ref = (freqs >= ref_lo) & (freqs <= ref_hi)
    band = (freqs >= band_lo) & (freqs <= band_hi)
    num = float(np.sum(P[band]) + 1e-12)
    den = float(np.sum(P[ref]) + 1e-12)
    return num / den

def murmur_ratio(x, sr):  # 150–600 Hz fraction
    return float(band_energy_ratio(x, sr, 150.0, 600.0, 20.0, 600.0))

def band_embed(x, sr, nfft=1024, hop=512, bands=16, fmin=20.0, fmax=600.0):
    if len(x) < nfft: x = np.pad(x, (0, nfft - len(x)))
    w = np.hanning(nfft).astype(np.float32)
    frames = max(1, (len(x)-nfft)//hop + 1)
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)
    edges = np.geomspace(fmin, fmax, bands+1)
    masks = [(freqs>=edges[i]) & (freqs<edges[i+1]) for i in range(bands)]
    E = np.zeros((frames, bands), dtype=np.float32)
    for i in range(frames):
        seg = x[i*hop:i*hop+nfft] * w
        P = (np.abs(np.fft.rfft(seg))**2).astype(np.float32)
        for b, m in enumerate(masks): E[i,b] = np.log1p(float(np.sum(P[m])))
    mu = E.mean(axis=0); sd = E.std(axis=0)
    feat = np.concatenate([mu, sd]).astype(np.float32)
    feat /= (np.linalg.norm(feat)+1e-9)
    return feat

def cosine(a,b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

# ---------------- Scoring ----------------
def combine_score(spec_sim, rr0, rr1, h0, h1):
    # residual delta calibrated to baseline
    rr_delta = max(0.0, (rr1 if rr1 is not None else 0.3) - (rr0 if rr0 is not None else 0.25)*1.05)
    r_sim = math.exp(-8.0 * rr_delta)  # small delta -> ~1
    h_delta = abs((h1 if h1 is not None else 0.0) - (h0 if h0 is not None else 0.0))
    h_sim = math.exp(-4.0 * h_delta)
    sim = 0.60*spec_sim + 0.30*r_sim + 0.10*h_sim
    return float(sim), rr_delta, h_delta

def status_from_score(score, calib=None):
    thr_similar = 0.95
    thr_little = 0.90
    if calib and ('mean' in calib) and ('std' in calib) and calib['std'] is not None:
        mu = calib['mean']; sd = calib['std']
        thr_similar = max(0.90, min(0.985, mu - 0.2*sd))
        thr_little  = max(0.85, min(0.97,  mu - 0.6*sd))
    if score >= thr_similar: return "Similar"
    if score >= thr_little: return "A little different (pending)"
    return "Different (needs confirm)"

# ---------------- Token helpers ----------------
def pack_token(obj: dict) -> str:
    return base64.b64encode(json.dumps(obj).encode('utf-8')).decode('ascii')

def unpack_token(tok: str) -> dict:
    try:
        return json.loads(base64.b64decode(tok.encode('ascii')).decode('utf-8'))
    except Exception:
        return {}

def load_vec(tokobj, key):
    return np.frombuffer(base64.b64decode(tokobj[key].encode("ascii")), dtype=np.float32)

# ---------------- UI ----------------
st.subheader("1) Create your baseline (3 WAVs)")
st.write("Same person, same chest spot, same stethoscope. We’ll build a cycle template, spectral profile, and baseline residual/murmur.")

bfiles = st.file_uploader("Baseline WAVs (3 files)", type=["wav"], accept_multiple_files=True, key="baseline")
colA, colB = st.columns(2)
with colA: build = st.button("Build baseline", type="primary")
with colB: clear = st.button("Clear baseline")

if "baseline_tok" not in st.session_state: st.session_state["baseline_tok"] = None
if "calib_scores" not in st.session_state: st.session_state["calib_scores"] = []

if clear:
    st.session_state["baseline_tok"] = None
    st.session_state["calib_scores"] = []
    st.success("Baseline cleared.")

def make_baseline(files):
    feats = []; all_cycles = []; murmur_list = []; rr_list = []
    for f in files:
        x, sr = parse_wav_bytes(f.read())
        if x is None: return None
        cycles = segment_cycles(x, sr)
        if len(cycles) >= 2:
            rr_list.append(residual_ratio(cycles, build_template(cycles)))
        all_cycles += (cycles if cycles else [])
        murmur_list.append(murmur_ratio(x, sr))
        feats.append(band_embed(x, sr))
    if not all_cycles: return None
    template = build_template(all_cycles)
    # Baseline residual measured against final template for consistency
    rr0 = residual_ratio(all_cycles, template)
    spec = np.mean(np.stack(feats, axis=0), axis=0)
    spec = spec / (np.linalg.norm(spec)+1e-9)
    murmur_med = float(np.median(np.array(murmur_list)))
    tok = pack_token({
        "template": base64.b64encode(template.astype(np.float32).tobytes()).decode("ascii"),
        "spec": base64.b64encode(spec.astype(np.float32).tobytes()).decode("ascii"),
        "rr0": rr0,
        "murmur": murmur_med,
        "calib": {"mean": None, "std": None, "n": 0}
    })
    return tok, rr0, murmur_med

if build:
    if not bfiles or len(bfiles) != 3:
        st.error("Please upload exactly 3 WAV files.")
    else:
        res = make_baseline(bfiles)
        if res is None:
            st.error("Could not build baseline (not enough clean cycles). Try re-recording in a quiet room, same spot.")
        else:
            tok, rr0, mh0 = res
            st.session_state["baseline_tok"] = tok
            st.success(f"Baseline ready. Baseline residual={rr0:.4f}, murmur={mh0:.4f}")

if st.session_state["baseline_tok"]:
    st.info("You can **download** your baseline token or paste an existing token below.")
    st.download_button("Download baseline token", data=st.session_state["baseline_tok"], file_name="baseline_token_v2_1.txt")
    tok_in = st.text_area("Paste an existing baseline token (optional)", value="", height=80, placeholder="Paste token here…")
    if tok_in.strip():
        st.session_state["baseline_tok"] = tok_in.strip()
        st.success("Baseline token replaced from pasted value.")

st.divider()
st.subheader("2) Weekly check-in (1 WAV)")

cfile = st.file_uploader("Check-in WAV", type=["wav"], key="checkin")
analyze = st.button("Analyze")

def analyze_once(tok, file):
    tokobj = unpack_token(tok)
    if not tokobj or "template" not in tokobj: return None
    x, sr = parse_wav_bytes(file.read())
    if x is None: return None
    cycles = segment_cycles(x, sr)
    T = load_vec(tokobj, "template")
    rr1 = residual_ratio(cycles, T) if cycles else None
    S0 = load_vec(tokobj, "spec"); S1 = band_embed(x, sr); ssim = cosine(S0, S1)
    h0 = float(tokobj.get("murmur", 0.0)); h1 = float(murmur_ratio(x, sr))
    rr0 = float(tokobj.get("rr0", 0.25))
    sim, rr_delta, h_delta = combine_score(ssim, rr0, rr1, h0, h1)
    calib = tokobj.get("calib", {"mean": None, "std": None, "n": 0})
    status = status_from_score(sim, calib=calib)
    return {
        "score": round(sim*100.0,1), "status": status,
        "rr0": rr0, "rr1": None if rr1 is None else round(rr1,4), "rr_delta": round(rr_delta,4),
        "spec_sim": round(ssim,4), "murmur0": round(h0,4), "murmur1": round(h1,4), "murmur_delta": round(h_delta,4),
        "tokobj": tokobj
    }

if analyze:
    tok = st.session_state.get("baseline_tok")
    if not tok:
        st.error("No baseline yet. Please build or paste a baseline token above.")
    elif not cfile:
        st.error("Please upload a check-in WAV.")
    else:
        res = analyze_once(tok, cfile)
        if not res:
            st.error("Analysis failed. Try another recording.")
        else:
            st.metric("Similarity score", f"{res['score']}")
            st.caption("Spectral + residual delta (whitened) + murmur delta")
            st.write(f"Status: **{res['status']}**")
            with st.expander("Details"):
                st.write(f"Residual baseline rr0: {res['rr0']:.4f}")
                st.write(f"Residual rr1 (this take): {res['rr1']}")
                st.write(f"Residual delta: {res['rr_delta']:.4f}")
                st.write(f"Spectral cosine similarity: {res['spec_sim']:.4f}")
                st.write(f"Murmur baseline: {res['murmur0']:.4f} → this take: {res['murmur1']:.4f} (Δ={res['murmur_delta']:.4f})")
            # Per-user calibration (last 7 scores)
            score01 = res['score']/100.0
            hist = st.session_state.get("calib_scores", [])
            hist = [score01] + hist
            hist = hist[:7]
            st.session_state["calib_scores"] = hist
            if len(hist) >= 5:
                mu = float(np.mean(hist)); sd = float(np.std(hist))
                res['tokobj']['calib'] = {"mean": mu, "std": sd, "n": len(hist)}
                new_tok = pack_token(res['tokobj'])
                st.download_button("Download updated baseline token (with calibration)", data=new_tok, file_name="baseline_token_v2_1_calibrated.txt")

st.divider()
st.caption("© Bumblebeat — wellness feedback, not diagnosis.")
