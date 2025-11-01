#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, math, random, argparse, json, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =========================================================
# Config
# =========================================================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds per sample")
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", type=str, default="cnns", choices=["cnns","resnet"])
    ap.add_argument("--mixup", type=float, default=0.0, help="0=off, e.g. 0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="ckpt.pth")
    ap.add_argument("--use_encodec_fake", action="store_true",
                    help="on-the-fly EnCodec recon as fake if fake dir missing (need encodec)")
    # Fourier method switches
    ap.add_argument("--use_fourier", action="store_true",
                    help="use Fourier artifact features and fuse with CNN")
    ap.add_argument("--fourier_only", action="store_true",
                    help="only use Fourier linear head (ignore CNN)")
    return ap.parse_args()

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# =========================================================
# Audio utils
# =========================================================
def load_audio(path, sr=32000, tgt_len=None):
    y, s = sf.read(path, always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if s != sr:
        y = torchaudio.functional.resample(torch.from_numpy(y).float(), s, sr).numpy()
    if tgt_len is not None:
        if len(y) < tgt_len:
            pad = tgt_len - len(y)
            y = np.pad(y, (pad//2, pad - pad//2))
        elif len(y) > tgt_len:
            st = (len(y)-tgt_len)//2
            y = y[st:st+tgt_len]
    return y.astype(np.float32)

def random_gain(y):
    g = 10**(np.random.uniform(-3, 3)/20)
    return (y*g).astype(np.float32)

# ---- Band-wise peak features ----
from scipy.signal import find_peaks

# 音乐常用频段：
BANDS = [(50, 300), (300, 1000), (1000, 3000), (3000, 8000)]  # Hz
PEAK_TOPK = 3  # 每个频段统计 top-k 峰值（按 prominence 排序）

def _band_peaks_features(freqs, psd_db, bands=BANDS, topk=PEAK_TOPK):
    # 先做基线扣除得到 prominence（与上面 _peak_feats 同口径）
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

        # 找峰：要求最小 prominence，避免把噪声当峰
        # prominence 阈值可按数据调，先给个温和值 0.3 dB
        peaks, props = find_peaks(band_prom, prominence=0.3)
        if len(peaks) == 0:
            feats.extend([band_prom.mean(), 0.0, band_prom.std(), 0.0, 0.0])
            continue

        pvals = props.get("prominences", np.zeros_like(peaks, dtype=float))
        order = np.argsort(pvals)[-topk:]
        p_top = pvals[order]
        f_top = band_freq[peaks[order]]

        # 峰的“间距”可反映梳状结构的周期性
        if len(f_top) >= 2:
            spacing = np.diff(np.sort(f_top))
            mean_spacing = float(np.mean(spacing))
        else:
            mean_spacing = 0.0

        feats.extend([
            float(band_prom.mean()),
            float(np.max(pvals)) if pvals.size else 0.0,
            float(band_prom.std()),
            float(len(peaks)) / float(m.sum()),  # 峰密度（归一化）
            mean_spacing
        ])
    return np.asarray(feats, dtype=np.float32)

def to_logmel(y, sr, n_mels=128):
    wav = torch.from_numpy(y).float().unsqueeze(0)  # (1, T)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=n_mels, f_min=20, f_max=sr//2, power=2.0
    )(wav)  # (1,n_mels,frames)
    logmel = torchaudio.transforms.AmplitudeToDB(stype="power")(spec)  # (1, n_mels, F)
    return logmel  # torch tensor

# -------- EnCodec reconstruct (robust, GPU safe) ----------
FRAME_HOP = 320  # align to frame hop to avoid tail mismatch
def _pad_to_multiple(x: torch.Tensor, m: int):
    T = x.shape[-1]
    r = T % m
    if r == 0: return x, 0
    pad = m - r
    return F.pad(x, (0, pad)), pad

def encodec_reconstruct(y, sr):
    try:
        from encodec import EncodecModel
    except Exception as e:
        raise RuntimeError("pip install encodec (or install from source) needed for --use_encodec_fake") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.encodec_model_24khz().to(device).eval()

    x = torch.from_numpy(y).float().unsqueeze(0).to(device)  # (1,T)
    to24 = torchaudio.transforms.Resample(sr, model.sample_rate).to(device)
    x24 = to24(x).unsqueeze(1)  # (1,1,T')
    x24, pad = _pad_to_multiple(x24, FRAME_HOP)

    with torch.no_grad():
        frames = model.encode(x24)      # expects (B,C,T)
        y24 = model.decode(frames)      # (B,C,T)
        if y24.dim() == 3: y24 = y24[:,0,:]
        if pad>0: y24 = y24[:, :-pad]

    back = torchaudio.transforms.Resample(model.sample_rate, sr).to(device)
    yhat = back(y24).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # center crop/pad to original length
    if len(yhat) < len(y):
        yhat = np.pad(yhat, (0, len(y)-len(yhat)))
    else:
        yhat = yhat[:len(y)]
    return yhat

# =========================================================
# Fourier artifact features (paper-inspired)
# =========================================================
# Welch PSD + baseline subtraction (median filter) + comb periodicity
from scipy.signal import welch, medfilt

# 加在 to_logmel 后（或 Dataset 的 __getitem__ 中）：
def spec_augment(logmel, max_t_mask=20, max_f_mask=16, p=0.5):
    # logmel: (1, n_mels, T)
    if np.random.rand() < p:
        _, m, t = logmel.shape
        # 频带遮挡
        f = np.random.randint(0, max_f_mask+1)
        f0 = np.random.randint(0, max(1, m - f + 1))
        logmel[:, f0:f0+f, :] = logmel.min()
        # 时间遮挡
        tt = np.random.randint(0, max_t_mask+1)
        t0 = np.random.randint(0, max(1, t - tt + 1))
        logmel[:, :, t0:t0+tt] = logmel.min()
    return logmel

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
    if vals.size == 0: vals = np.array([0.0])
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
    f, psd = _avg_psd_db(y, sr)                       # Welch PSD → dB
    base10 = np.concatenate([_peak_feats(f, psd),     # 6 维
                             _comb_feats(f, psd)],    # 4 维
                            axis=0)                   # => 10 维
    bandf = _band_peaks_features(f, psd)              # 每段 5 维 × 段数
    feat = np.concatenate([base10, bandf], axis=0)
    return feat.astype(np.float32)

FOURIER_DIM = 10 + 5 * len(BANDS)


# =========================================================
# Dataset
# =========================================================
class TwoClassAudio(Dataset):
    def __init__(self, root_split, sr, duration, n_mels,
                 use_encodec_fake=False, use_fourier=False, fourier_only=False):
        self.sr = sr
        self.dur = duration
        self.tgt_len = int(sr*duration)
        self.n_mels = n_mels
        self.use_encodec_fake = use_encodec_fake
        self.use_fourier = use_fourier
        self.fourier_only = fourier_only

        self.real = []
        self.fake = []
        for ext in ("wav","flac","mp3","ogg","m4a"):
            self.real += glob.glob(os.path.join(root_split, "real", f"**/*.{ext}"), recursive=True)
            self.fake += glob.glob(os.path.join(root_split, "fake", f"**/*.{ext}"), recursive=True)

        if len(self.real)==0:
            raise RuntimeError(f"No real audio found under {root_split}/real")
        if len(self.fake)==0 and not self.use_encodec_fake:
            raise RuntimeError(f"No fake audio found under {root_split}/fake; or enable --use_encodec_fake")

        self.paths = [(p,0) for p in self.real] + ([(p,1) for p in self.fake] if len(self.fake)>0 else [])
        if len(self.fake)==0 and self.use_encodec_fake:
            # on-the-fly recon as fake
            self.paths += [(p,1) for p in random.sample(self.real, min(len(self.real), 10000))]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        y = load_audio(path, sr=self.sr, tgt_len=self.tgt_len)
        # data aug
        if random.random()<0.5: y = random_gain(y)
        if label==1 and (len(self.fake)==0) and self.use_encodec_fake:
            y = encodec_reconstruct(y, self.sr)

        # features
        logmel = to_logmel(y, self.sr, self.n_mels)
        if getattr(self, "split", "train") == "train":
            logmel = spec_augment(logmel)

        if self.use_fourier or self.fourier_only:
            fvec = extract_fourier_features(y, self.sr)  # (D,)
            return (logmel, torch.from_numpy(fvec).float()), torch.tensor(label, dtype=torch.long)
        else:
            return logmel, torch.tensor(label, dtype=torch.long)

# =========================================================
# Models
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):  # x: (B,1,M,T)
        h = self.net(x)
        h = h.squeeze(-1).squeeze(-1)
        return self.fc(h)

class CNNWithFourierHead(nn.Module):
    def __init__(self, n_mels=128, fourier_dim=FOURIER_DIM, n_classes=2):
        super().__init__()
        # CNN backbone -> 256-d embedding
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + fourier_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, fvec):
        h = self.backbone(x)            # (B,256,1,1)
        h = h.squeeze(-1).squeeze(-1)   # (B,256)
        z = torch.cat([h, fvec], dim=1) # (B,256+D)
        return self.head(z)

def get_model(name="cnns", n_mels=128, fourier_dim=None, use_fourier=False, fourier_only=False):
    if fourier_only:
        # Only Fourier linear head
        return nn.Sequential(
            nn.LayerNorm(fourier_dim),
            nn.Linear(fourier_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    if use_fourier:
        return CNNWithFourierHead(n_mels=n_mels, fourier_dim=fourier_dim or FOURIER_DIM)
    if name=="cnns":
        return SmallCNN(n_mels=n_mels)
    # ResNet18 (1ch)
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, 2)
    return m

# =========================================================
# Mixup (optional)
# =========================================================
def mixup_data(x, y, alpha=0.2):
    if alpha<=0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    mixed_x = lam*x + (1-lam)*x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)

# =========================================================
# Train / Eval
# =========================================================
def run_epoch(model, loader, opt, scaler, device, args, mix_alpha=0.0, train=True):
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    all_y, all_p = [], []
    total_loss = 0.0
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()

    for xb, yb in tqdm(loader, disable=not train):
        # xb may be Tensor or (logmel, fourier_vec)
        if isinstance(xb, (list, tuple)):
            x_logmel, x_fourier = xb
            x_logmel = x_logmel.to(device)
            x_fourier = x_fourier.to(device)
            has_fourier = True
        else:
            x_logmel = xb.to(device)
            x_fourier = None
            has_fourier = False
        yb = yb.to(device)

        if train:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
                if args.fourier_only:
                    # model expects Fourier vector only
                    logits = model(x_fourier)
                elif has_fourier and args.use_fourier:
                    logits = model(x_logmel, x_fourier)
                else:
                    if mix_alpha>0:
                        x2, y_a, y_b2, lam = mixup_data(x_logmel, yb, alpha=mix_alpha)
                        logits = model(x2)
                        loss = mixup_criterion(criterion, logits, y_a, y_b2, lam)
                    else:
                        logits = model(x_logmel)
                        loss = criterion(logits, yb)
                if not (mix_alpha>0 and not (has_fourier or args.fourier_only)):
                    # when not using mixup path above
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
                if args.fourier_only:
                    logits = model(x_fourier)
                elif has_fourier and args.use_fourier:
                    logits = model(x_logmel, x_fourier)
                else:
                    logits = model(x_logmel)
                loss = criterion(logits, yb)

        # metrics: cast to float32 before numpy
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_p.append(probs.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
        total_loss += loss.item()*x_logmel.size(0)

    all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
    pred = (all_p>=0.5).astype(int)
    acc = accuracy_score(all_y, pred)
    f1 = f1_score(all_y, pred)
    try:
        auc = roc_auc_score(all_y, all_p)
    except Exception:
        auc = float("nan")
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, f1, auc

def main():
    args = get_args()
    if args.fourier_only and args.use_fourier:
        print("[warn] both --use_fourier and --fourier_only set; using fourier_only")
        args.use_fourier = False
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    train_ds = TwoClassAudio(
        os.path.join(args.data_root, "train"),
        args.sr, args.duration, args.n_mels,
        use_encodec_fake=args.use_encodec_fake,
        use_fourier=args.use_fourier, fourier_only=args.fourier_only
    )
    val_ds   = TwoClassAudio(
        os.path.join(args.data_root, "val"),
        args.sr, args.duration, args.n_mels,
        use_encodec_fake=False,
        use_fourier=args.use_fourier, fourier_only=args.fourier_only
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Model
    fourier_dim = FOURIER_DIM if (args.use_fourier or args.fourier_only) else None
    model = get_model(args.model, args.n_mels,
                      fourier_dim=fourier_dim,
                      use_fourier=args.use_fourier,
                      fourier_only=args.fourier_only).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_auc, best_state = -1.0, None
    history = []

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = run_epoch(model, train_dl, opt, scaler, device, args, mix_alpha=args.mixup, train=True)
        va = run_epoch(model, val_dl,   opt, scaler, device, args, mix_alpha=0.0,       train=False)
        print(f"train: loss={tr[0]:.4f} acc={tr[1]:.4f} f1={tr[2]:.4f} auc={tr[3]:.4f}")
        print(f"valid: loss={va[0]:.4f} acc={va[1]:.4f} f1={va[2]:.4f} auc={va[3]:.4f}")

        history.append({"epoch":epoch,"train":{"loss":tr[0],"acc":tr[1],"f1":tr[2],"auc":tr[3]},
                                   "val":{"loss":va[0],"acc":va[1],"f1":va[2],"auc":va[3]}})

        if not math.isnan(va[3]) and va[3] > best_auc:
            best_auc = va[3]
            best_state = { "model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": history[-1] }
            torch.save(best_state, args.out)
            print(f"[save] best ckpt @ {args.out} (AUC={best_auc:.4f})")

    with open(os.path.splitext(args.out)[0] + "_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
