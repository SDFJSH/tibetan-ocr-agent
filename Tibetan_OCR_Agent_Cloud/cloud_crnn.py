"""CRNN + CTC: faithful copy of user's original train_crnn.py inference structure.
Only trusted self-produced PyTorch checkpoints; never accepts arbitrary uploaded checkpoints.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

class Charset:
    def __init__(self, chars):
        self.blank_id = 0
        self.chars = list(chars)
        self.char2id = {c: i + 1 for i, c in enumerate(self.chars)}
        self.id2char = {i + 1: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"])

    @property
    def num_classes(self):
        return len(self.chars) + 1

    def decode_ctc(self, ids):
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != self.blank_id and i != prev:
                out.append(self.id2char.get(i, ""))
            prev = i
        return "".join(out)

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(256, hidden_size, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)

def resize_keep_ratio(img, img_height=48, max_width=1024):
    w, h = img.size
    new_w = max(8, int(round(w * img_height / max(1, h))))
    new_w = min(new_w, max_width)
    return img.resize((new_w, img_height), Image.Resampling.BILINEAR), new_w

def pil_to_tensor(img):
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

class CachedCRNN:
    """Load once per Streamlit process, use same original decoding and padding."""
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CRNN权重不存在: {path}。请先从AutoDL下载并放到 models/best.pt")
        torch.set_num_threads(min(2, torch.get_num_threads()))
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("请使用支持weights_only的PyTorch版本") from exc
        if not isinstance(ckpt, dict) or not {'model', 'charset'}.issubset(ckpt):
            raise ValueError("检查点必须包含 model 与 charset 字段")
        self.charset = Charset.from_dict(ckpt['charset'])
        self.model = CRNN(self.charset.num_classes).cpu()
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.model.eval()
        args = ckpt.get('args', {})
        args = args if isinstance(args, dict) else {}
        self.height = int(args.get('img_height', 48))
        self.max_width = int(args.get('max_width', 1024))

    @torch.inference_mode()
    def recognize(self, image_path):
        with Image.open(image_path) as image:
            # Like user's original OCRDataset, convert RGB. Do not auto-rotate.
            image = image.convert("RGB")
            resized, true_width = resize_keep_ratio(image, self.height, self.max_width)
            tensor = pil_to_tensor(resized)
        padded_width = ((true_width + 3) // 4) * 4
        batch = torch.zeros(1, 3, self.height, padded_width, dtype=tensor.dtype)
        batch[0, :, :, :true_width] = tensor
        logits = self.model(batch)
        idx = logits.argmax(dim=-1)[:, 0]
        nsteps = max(1, true_width // 4)
        output = self.charset.decode_ctc(idx[:nsteps].cpu().tolist())
        return {'text': output, 'confidence': None, 'demo_only': False,
                'model': 'CRNN', 'image_width_after_resize': true_width, 'ctc_steps': nsteps}
