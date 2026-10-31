"""Small, model-independent utilities for the competition UI.

No OCR model weights, API calls or user data are persisted by this module.
"""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np


def normalize(text: str) -> str:
    """Apply only NFC; preserve Tibetan tsheg, punctuation and whitespace."""
    return unicodedata.normalize('NFC', str(text))


def char_edit_counts(prediction: str, reference: str, max_chars: int = 1800) -> dict:
    """Levenshtein distance with one optimal S/D/I alignment; cost O(mn).

    `reference` must be a human-supplied transcription. A zero-length reference
    cannot provide a meaningful CER denominator.
    """
    pred = normalize(prediction)
    ref = normalize(reference)
    if not ref:
        raise ValueError('参考真值不能为空，无法计算CER')
    if len(pred) > max_chars or len(ref) > max_chars:
        raise ValueError(f'单行评估文本超过{max_chars}字符限制；请拆分后评估')
    # Tuple: (edit distance, substitution count, deletion count, insertion count)
    # Deletion means a reference character was missing from the prediction.
    prior = [(j, 0, j, 0) for j in range(len(ref) + 1)]
    for i, x in enumerate(pred, 1):
        row = [(i, 0, 0, i)]
        for j, y in enumerate(ref, 1):
            match = prior[j - 1]
            if x != y:
                match = (match[0] + 1, match[1] + 1, match[2], match[3])
            missing = (row[-1][0] + 1, row[-1][1], row[-1][2] + 1, row[-1][3])
            added = (prior[j][0] + 1, prior[j][1], prior[j][2], prior[j][3] + 1)
            # Prefer match/substitution in ties; breakdown can be alignment-dependent.
            row.append(min((match, missing, added), key=lambda t: t[0]))
        prior = row
    distance, subs, dels, ins = prior[-1]
    return {
        'reference_chars': len(ref), 'predicted_chars': len(pred),
        'edit_distance': distance, 'substitutions': subs,
        'deletions': dels, 'insertions': ins,
        'cer': distance / len(ref),
    }


def enhance_preview_bytes(source: bytes) -> bytes:
    """Same CLAHE parameters as tib_agent.image_tools.enhance_image."""
    image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码输入图像')
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    improved = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    okay, result = cv2.imencode('.png', improved)
    if not okay:
        raise ValueError('无法生成增强预览')
    return result.tobytes()


def mask_secrets(payload):
    """Redact config/private token fields from diagnostic JSON views."""
    if isinstance(payload, dict):
        return {k: ('[已隐藏]' if any(word in k.lower() for word in
                  ('api_key', 'password', 'authorization', 'token', 'secret'))
                  else mask_secrets(v)) for k, v in payload.items()}
    if isinstance(payload, list):
        return [mask_secrets(v) for v in payload]
    if isinstance(payload, str):
        return re.sub(r'\bsk-[A-Za-z0-9_-]{8,}', '[已隐藏密钥]', payload)
    return payload


def make_review_archive(response: dict, corrections: dict) -> bytes:
    """Exports only user review and recognized text, never uploaded images."""
    records = []
    for item in response.get('records', []):
        file = item['file']
        correction = corrections.get(file, {})
        records.append({
            'file': file,
            'crnn_original': item.get('original_ocr_text', item.get('recognized_text', '')),
            'crnn_final': item.get('recognized_text', ''),
            'enhanced_candidate': item.get('enhanced_ocr_text'),
            'human_edited_text': correction.get('text'),
            'confirmed_by_human': bool(correction.get('confirmed', False)),
            'final_export_text': correction.get('text', item.get('recognized_text', '')),
            'needs_human_review': item.get('needs_human_review', False),
        })
    meta = {
        'project': '藏智录·藏文OCR智能体',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'planner_source': response.get('planner', {}).get('source', 'unknown'),
        'note': '人工确认字段仅在用户主动勾选时为true；未提供真值时不计算CER。',
        'records': records,
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('reviewed_results.json', json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr('reviewed_text.txt', '\n\n'.join(
            f"[{r['file']}]\n{r['final_export_text']}" for r in records))
        txt = io.StringIO()
        writer = csv.writer(txt)
        writer.writerow(['file','crnn_original','crnn_final','human_edited_text','confirmed_by_human','final_export_text'])
        for r in records:
            writer.writerow([r['file'], r['crnn_original'], r['crnn_final'],
                             r['human_edited_text'] or '', r['confirmed_by_human'], r['final_export_text']])
        zf.writestr('reviewed_summary.csv', '\ufeff' + txt.getvalue())
    return out.getvalue()


def metrics_csv(rows: list[dict]) -> bytes:
    """Export evaluation rows as UTF-8 BOM CSV, readable in Excel."""
    text = io.StringIO()
    names = ['file', 'prediction_source', 'reference_chars', 'predicted_chars',
             'substitutions', 'deletions', 'insertions', 'edit_distance', 'cer_percent']
    writer = csv.DictWriter(text, fieldnames=names, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return ('\ufeff' + text.getvalue()).encode('utf-8')
