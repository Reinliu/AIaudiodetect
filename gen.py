#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, numpy as np
import torch, torchaudio, soundfile as sf
from tqdm import tqdm
from encodec import EncodecModel

# EnCodec 24kHz 每帧步长通常是 320/480 这种因子；为了稳妥，这里取 320 做对齐
FRAME_HOP = 320

def pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    """在最后一维 T 上 pad 到 m 的倍数，返回新张量与 pad 的长度。"""
    T = x.shape[-1]
    rem = T % m
    if rem == 0:
        return x, 0
    pad = m - rem
    return torch.nn.functional.pad(x, (0, pad)), pad

def recon_one(wav_np: np.ndarray, sr: int, model: EncodecModel) -> np.ndarray:
    """
    wav_np: (T,) float32, 建议幅度在 [-1, 1]
    sr:     原始采样率（我们最终也输出这个 sr）
    model:  已 .to(device).eval() 的 EncodecModel
    返回:   (T,) float32
    """
    device = next(model.parameters()).device

    # numpy -> torch (B=1, T)
    x = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)  # (1, T)
    x = torch.clamp(x, -1.0, 1.0)

    # 重采样到 24k（与模型一致），注意把 Resample 放到同一设备
    to24 = torchaudio.transforms.Resample(sr, model.sample_rate).to(device)
    x24 = to24(x)  # (1, T')
    # 加通道维 C=1 → (B, C, T)
    x24 = x24.unsqueeze(1)  # (1, 1, T')

    # 对齐到帧步长（避免最后一帧不完整）
    x24, pad = pad_to_multiple(x24, FRAME_HOP)

    with torch.no_grad():
        frames = model.encode(x24)        # 期望 (B, C, T)
        y24 = model.decode(frames)        # (B, C, T)
    # 去掉通道维
    if y24.dim() == 3:
        y24 = y24[:, 0, :]                # (B, T)
    # 去掉可能的 pad
    if pad > 0:
        y24 = y24[:, :-pad]

    # 重采样回原训练采样率 sr（同一设备）
    back = torchaudio.transforms.Resample(model.sample_rate, sr).to(device)
    y = back(y24)                         # (B, T)
    y = y.squeeze(0).detach().cpu().numpy().astype("float32")
    # 振幅裁剪
    y = np.clip(y, -1.0, 1.0)
    return y

def gen_split(split="train", sr=32000, in_dir=None, out_dir=None):
    in_dir  = in_dir  or f"data/{split}/real"
    out_dir = out_dir or f"data/{split}/fake"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.encodec_model_24khz().to(device).eval()

    files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    for rp in tqdm(files, desc=f"{split} encodec recon"):
        fp = os.path.join(out_dir, os.path.basename(rp))
        if os.path.exists(fp):
            continue
        wav, s = sf.read(rp)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # 保证输入与目标 sr 一致（训练时也用这个 sr）
        if s != sr:
            wav = torchaudio.functional.resample(torch.from_numpy(wav).float(), s, sr).numpy()
        # 调幅到 [-1,1]
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / (np.max(np.abs(wav)) + 1e-9)

        y = recon_one(wav.astype("float32"), sr, model)

        # 与原长度对齐（可选）
        if len(y) < len(wav):
            y = np.pad(y, (0, len(wav) - len(y)))
        else:
            y = y[:len(wav)]

        sf.write(fp, y, sr)

    print(f"[ok] wrote fake wavs to {out_dir}")

if __name__ == "__main__":
    # 生成 train/val 的 fake
    gen_split("train", sr=32000)
    gen_split("val",   sr=32000)
