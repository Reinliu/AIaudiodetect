#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import gradio as gr

from scipy.signal import welch, medfilt, find_peaks

# =========================================================
# MUST MATCH train.py  :contentReference[oaicite:0]{index=0}
# =========================================================
BANDS = [(50, 300), (300, 1000), (1000, 3000), (3000, 8000)]
PEAK_TOPK = 3

def _band_peaks_features(freqs, psd_db, bands=BANDS, topk=PEAK_TOPK):
    k = 101 if 101 % 2 == 1 else 102
    base = medfilt(psd_db, k)
    prom = np.maximum(psd_db - base, 0.0)
    feats = []
    for (fmin, fmax) in bands:
        m = (freqs >= fmin) & (freqs <= fmax)
        if m.sum() < 8:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        band_prom = prom[m]
        band_freq = freqs[m]
        peaks, props = find_peaks(band_prom, prominence=0.3)
        if len(peaks) == 0:
            feats.extend([band_prom.mean(), 0.0, band_prom.std(), 0.0, 0.0])
            continue
        pvals = props.get("prominences", np.zeros_like(peaks, dtype=float))
        order = np.argsort(pvals)[-topk:]
        f_top = band_freq[peaks[order]]
        if len(f_top) >= 2:
            spacing = np.diff(np.sort(f_top))
            mean_spacing = float(np.mean(spacing))
        else:
            mean_spacing = 0.0
        feats.extend([
            float(band_prom.mean()),
            float(np.max(pvals)) if pvals.size else 0.0,
            float(band_prom.std()),
            float(len(peaks)) / float(m.sum()),
            mean_spacing
        ])
    return np.asarray(feats, dtype=np.float32)

def _avg_psd_db(y, sr, nperseg=4096, noverlap=3072):
    f, P = welch(y, fs=sr, nperseg=nperseg, noverlap=noverlap,
                 window="hann", scaling="spectrum")
    P = np.maximum(P, 1e-12)
    return f, 10.0*np.log10(P)

def _peak_feats(freqs, psd_db, med_ksize=101, topk=20):
    k = med_ksize if med_ksize % 2 == 1 else med_ksize + 1
    base = medfilt(psd_db, k)
    prom = np.maximum(psd_db - base, 0.0)
    is_peak = (prom > np.roll(prom, 1)) & (prom >= np.roll(prom, -1)) & (prom > 0.5)
    vals = prom[is_peak]
    if vals.size == 0:
        vals = np.array([0.0])
    top = np.sort(vals)[-topk:]
    return np.array([
        prom.mean(), prom.max(), prom.std(),
        top.mean() if top.size else 0.0,
        top.std() if top.size else 0.0,
        is_peak.mean()
    ], dtype=np.float32)

def _comb_feats(freqs, psd_db, fmin=100, fmax=8000):
    m = (freqs >= fmin) & (freqs <= fmax)
    if m.sum() < 32:
        return np.zeros(4, dtype=np.float32)
    s = (psd_db[m] - psd_db[m].mean()) / (psd_db[m].std() + 1e-8)
    ac = np.correlate(s, s, mode="full")
    ac = ac[ac.size//2:]
    ac[0] = 0
    lag = int(np.argmax(ac))
    top5 = np.sort(ac)[-5:]
    return np.array([ac.max(), lag, top5.mean(), top5.std()], dtype=np.float32)

def extract_fourier_features(y, sr):
    f, psd = _avg_psd_db(y, sr)
    base10 = np.concatenate([_peak_feats(f, psd), _comb_feats(f, psd)], axis=0)
    bandf = _band_peaks_features(f, psd)
    feat = np.concatenate([base10, bandf], axis=0)
    return feat.astype(np.float32)

FOURIER_DIM = 10 + 5 * len(BANDS)

# =========================================================
# audio utils
# =========================================================
def load_audio_any(path: str, target_sr: int):
    y, s = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != target_sr:
        y = torchaudio.functional.resample(torch.from_numpy(y).float(), s, target_sr).numpy()
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y, target_sr

def center_crop(y: np.ndarray, L: int):
    if len(y) < L:
        pad = L - len(y)
        y = np.pad(y, (pad//2, pad - pad//2))
    elif len(y) > L:
        st = (len(y) - L)//2
        y = y[st:st+L]
    return y

def make_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float):
    L = int(sr * win_s)
    H = int(sr * hop_s)
    if len(y) < L:
        return [center_crop(y, L)]
    xs = []
    for st in range(0, max(1, len(y) - L + 1), H):
        xs.append(y[st:st+L])
    return xs

def to_logmel_torch(y: np.ndarray, sr: int, n_mels: int=128,
                    n_fft: int=1024, hop: int=256, win: int=1024):
    wav = torch.from_numpy(y).float().unsqueeze(0)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, f_min=20, f_max=sr//2, power=2.0
    )(wav)
    logmel = torchaudio.transforms.AmplitudeToDB(stype="power")(spec)
    return logmel

# =========================================================
# Deezer-like CNN + Fourier fusion  (same spirit as we discussed)
# you can replace this block with the exact code in your train.py
# =========================================================
class DeezerSimpleCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=2, in_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(256, n_classes)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        emb = self.forward_features(x)
        return self.head(emb)

class DeezerCNNWithFourier(nn.Module):
    def __init__(self, n_mels=128, fourier_dim=FOURIER_DIM, n_classes=2):
        super().__init__()
        self.cnn = DeezerSimpleCNN(n_mels=n_mels, n_classes=n_classes)
        self.fuse = nn.Sequential(
            nn.Linear(256 + fourier_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, fvec):
        emb = self.cnn.forward_features(x)
        z = torch.cat([emb, fvec], dim=1)
        return self.fuse(z)

def get_model(name="cnns", n_mels=128, fourier_dim=None,
              use_fourier=False, fourier_only=False):
    if fourier_only:
        return nn.Sequential(
            nn.LayerNorm(fourier_dim),
            nn.Linear(fourier_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    if use_fourier:
        return DeezerCNNWithFourier(n_mels=n_mels, fourier_dim=fourier_dim or FOURIER_DIM)
    if name == "cnns":
        return DeezerSimpleCNN(n_mels=n_mels)
    # fallback resnet
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, 2)
    return m

# =========================================================
# Detector
# =========================================================
class Detector:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        # load state (pytorch 2.6 friendly)
        try:
            state = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "model" in state:
            model_sd = state["model"]
            saved_args = state.get("args", {})
        else:
            model_sd = state
            saved_args = {}

        # args from training
        self.sr = saved_args.get("sr", 32000)
        self.n_mels = saved_args.get("n_mels", 128)
        self.model_name = saved_args.get("model", "cnns")
        self.use_fourier = bool(saved_args.get("use_fourier", False))
        self.fourier_only = bool(saved_args.get("fourier_only", False))

        # device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---- infer fourier dim from checkpoint ----
        inferred_fourier_dim = None
        if self.use_fourier and not self.fourier_only:
            for k, v in model_sd.items():
                if k.startswith("fuse.0.weight") or k.startswith("fuse.0.linear.weight"):
                    in_feats = v.shape[1]     # e.g. 291
                    inferred_fourier_dim = in_feats - 256
                    break
        elif self.fourier_only:
            for k, v in model_sd.items():
                if "0.weight" in k or "0.linear.weight" in k:
                    inferred_fourier_dim = v.shape[1]
                    break

        if inferred_fourier_dim is None:
            inferred_fourier_dim = FOURIER_DIM

        self.fourier_dim = inferred_fourier_dim

        # build model with correct dim
        self.model = get_model(
            self.model_name,
            self.n_mels,
            fourier_dim=self.fourier_dim,
            use_fourier=self.use_fourier,
            fourier_only=self.fourier_only,
        ).to(self.device)

        self.model.load_state_dict(model_sd)
        self.model.eval()
        self.use_bf16 = (self.device.type == "cuda") and torch.cuda.is_bf16_supported()

        print(f"[app] loaded model. use_fourier={self.use_fourier}, fourier_only={self.fourier_only}, fourier_dim={self.fourier_dim}, sr={self.sr}")

    # ------------ internal: score a single clip ------------
    @torch.no_grad()
    def _score_clip(self, y: np.ndarray) -> float:
        # 1) logmel branch (if not fourier_only)
        if not self.fourier_only:
            logmel = to_logmel_torch(y, self.sr, self.n_mels)
            x = logmel.unsqueeze(0).to(self.device)   # (1,1,M,T)
        # 2) fourier branch (if needed)
        if self.use_fourier or self.fourier_only:
            fvec = extract_fourier_features(y, self.sr)   # maybe 30
            # pad/trim to checkpoint dim
            if fvec.shape[0] < self.fourier_dim:
                fvec = np.pad(fvec, (0, self.fourier_dim - fvec.shape[0]))
            elif fvec.shape[0] > self.fourier_dim:
                fvec = fvec[:self.fourier_dim]
            f = torch.from_numpy(fvec).float().unsqueeze(0).to(self.device)

        # 3) forward
        with torch.autocast(device_type=self.device.type,
                            dtype=(torch.bfloat16 if self.use_bf16 else torch.float16)):
            if self.fourier_only:
                logits = self.model(f)
            elif self.use_fourier:
                logits = self.model(x, f)
            else:
                logits = self.model(x)

        prob = torch.softmax(logits.float(), dim=1)[0, 1].item()
        return float(prob)

    # ------------ public: score a file ------------
    @torch.no_grad()
    def predict(self,
                filepath: str,
                mode: str = "center",
                clip_seconds: float = 10.0,
                win_seconds: float = 6.0,
                hop_seconds: float = 3.0,
                agg: str = "mean") -> dict:
        y, _ = load_audio_any(filepath, self.sr)
        L = int(self.sr * clip_seconds)

        if mode == "center":
            y0 = center_crop(y, L)
            scores = [self._score_clip(y0)]
        else:
            windows = make_windows(y, self.sr, win_seconds, hop_seconds)
            scores = [self._score_clip(w) for w in windows]

        if agg == "mean":
            agg_score = float(np.mean(scores))
        elif agg == "max":
            agg_score = float(np.max(scores))
        else:
            agg_score = float(np.median(scores))

        return {
            "file": os.path.basename(filepath),
            "sr": self.sr,
            "duration_sec": round(len(y)/self.sr, 3),
            "mode": mode,
            "window_scores": [float(s) for s in scores],
            "aggregate": agg_score,
            "label": "AI" if agg_score >= 0.5 else "REAL",
            "flags": {
                "use_fourier": self.use_fourier,
                "fourier_only": self.fourier_only,
                "fourier_dim": self.fourier_dim,
            }
        }


# =========================================================
# Gradio UI
# =========================================================
def build_interface(detector: Detector):
    with gr.Blocks(title="Audio Deepfake Detector") as demo:
        gr.Markdown(
            f"# 🔍 Audio Deepfake Detector\n"
            f"- sr: **{detector.sr}**\n"
            f"- Fourier fusion: **{detector.use_fourier}**\n"
            f"- Fourier only: **{detector.fourier_only}**\n"
            f"- bands: `{BANDS}`"
        )
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                         label="Upload / record WAV/MP3")
        with gr.Accordion("Inference options", open=False):
            mode = gr.Radio(["center", "sliding"], value="sliding", label="mode")
            clip_seconds = gr.Slider(3, 30, value=10, step=1, label="center-crop seconds")
            win_seconds  = gr.Slider(3, 15, value=10, step=1, label="sliding window seconds")
            hop_seconds  = gr.Slider(1, 10, value=3, step=1, label="sliding hop seconds")
            agg          = gr.Radio(["mean", "max", "median"], value="max", label="aggregate")
        btn = gr.Button("Detect", variant="primary")
        out_label = gr.Label(label="Prediction")
        out_num   = gr.Number(label="AI probability")
        out_json  = gr.JSON(label="Details")

        def _infer(path, mode, clip_seconds, win_seconds, hop_seconds, agg):
            if path is None:
                return {"REAL": 1.0, "AI": 0.0}, 0.0, {"error": "no audio file"}
            res = detector.predict(
                filepath=path,
                mode=mode,
                clip_seconds=float(clip_seconds),
                win_seconds=float(win_seconds),
                hop_seconds=float(hop_seconds),
                agg=agg
            )
            return {"AI": res["aggregate"], "REAL": 1.0 - res["aggregate"]}, res["aggregate"], res

        btn.click(_infer,
                  inputs=[audio, mode, clip_seconds, win_seconds, hop_seconds, agg],
                  outputs=[out_label, out_num, out_json])
    return demo

# =========================================================
# main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpt.pth")
    ap.add_argument("--server_name", type=str, default="0.0.0.0")
    ap.add_argument("--server_port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    det = Detector(args.ckpt)
    ui = build_interface(det)
    # ui.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
    ui.launch(share=True)
