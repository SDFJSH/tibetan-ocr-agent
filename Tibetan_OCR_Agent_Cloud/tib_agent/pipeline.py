import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from .settings import Settings
from .image_tools import analyze_image, enhance_image
from .ocr_bridge import OCRBridge
from .planner import plan_task, decide_review_after_tools


class OCRPipeline:
    def __init__(self, settings: Optional[Settings] = None, ocr=None):
        self.settings = settings or Settings()
        self.ocr = ocr or OCRBridge(self.settings)

    def process_one(self, image_path: str, review_mode: str = 'auto',
                    working_dir: Optional[Path] = None, task: str = '',
                    allow_qwen_feedback: bool = False) -> dict:
        input_path = Path(image_path)
        if input_path.stat().st_size > self.settings.max_image_mb * 1024 * 1024:
            raise ValueError(f'文件超过{self.settings.max_image_mb}MB大小限制：{input_path.name}')
        quality = analyze_image(str(input_path),
                                self.settings.sharpness_threshold,
                                self.settings.contrast_threshold)
        trace = [{'tool': 'analyze_image', 'detail': quality['flags'] or ['未发现明显问题']}]
        first = self.ocr.recognize(str(input_path))
        trace.append({'tool': 'crnn_ocr', 'detail': '首次识别完成',
                      'demo_only': first.get('demo_only', False)})
        should_review = review_mode == 'force' or (review_mode == 'auto' and quality['needs_enhancement'])
        review_decision_source = 'rule_fallback'
        if review_mode == 'auto' and allow_qwen_feedback:
            decision = decide_review_after_tools(task, quality, first['text'], self.settings)
            should_review = decision['action'] == 'enhance_retry'
            review_decision_source = decision['source']
            trace.append({'tool': 'agent_review_decision',
                          'detail': f"{decision['action']} · {decision['source']}"})
            if decision.get('warning'):
                trace.append({'tool': 'agent_fallback', 'detail': decision['warning']})
        alternative = None
        if should_review:
            root = working_dir or input_path.parent
            enhanced_path = str(root / (uuid.uuid4().hex[:10] + '_enhanced.png'))
            enhance_image(str(input_path), enhanced_path)
            trace.append({'tool': 'enhance_image', 'detail': '生成CLAHE增强候选图像'})
            alternative = self.ocr.recognize(enhanced_path)
            trace.append({'tool': 'crnn_ocr', 'detail': '增强后再次识别'})
        final_text = first['text']
        needs_human_review = False
        review_reasons = []
        if alternative is not None:
            if first['text'] != alternative['text']:
                needs_human_review = True
                review_reasons.append('原图与增强图的OCR结果不一致，需人工核验')
                # 仅在外部确实提供经验证集校准的可信度时自动改用第二结果
                c1, c2 = first.get('confidence'), alternative.get('confidence')
                if (self.settings.ocr_confidence_calibrated and
                    c1 is not None and c2 is not None and c2 > c1 + 0.05):
                    final_text = alternative['text']
                    review_reasons.append('采用校准可信度更高的候选，但仍保留人工复核提示')
                    trace.append({'tool': 'candidate_selection', 'detail': '使用经校准的可信度选择增强候选'})
                else:
                    trace.append({'tool': 'candidate_selection', 'detail': '保留原图结果并标记差异'})
            else:
                review_reasons.append('两次OCR结果一致；一致不等于一定正确')
        if quality['flags'] and alternative is None:
            review_reasons.append('图像存在质量风险，本次未执行重识别')
        if first.get('demo_only'):
            review_reasons.append('演示模式假数据：不能用于实验或参赛展示的模型性能')
        return {
            'file': input_path.name,
            'recognized_text': final_text,
            'original_ocr_text': first['text'],
            'enhanced_ocr_text': alternative['text'] if alternative else None,
            'original_confidence': first.get('confidence'),
            'enhanced_confidence': alternative.get('confidence') if alternative else None,
            'confidence_calibrated': self.settings.ocr_confidence_calibrated,
            'quality': quality,
            'review_performed': alternative is not None,
            'review_decision_source': review_decision_source,
            'needs_human_review': needs_human_review,
            'review_reasons': review_reasons,
            'trace': trace,
            'demo_only': bool(first.get('demo_only')),
        }

    def process_many(self, paths: list[str], task: str = '', export: bool = True) -> dict:
        if not paths:
            raise ValueError('请至少上传一张藏文图像')
        if len(paths) > 100:
            raise ValueError('单批最多100张图片')
        plan = plan_task(task, self.settings)
        idx = plan.get('target_index')
        selected = paths
        if idx is not None:
            if idx > len(paths):
                raise ValueError(f'用户指定第{idx}张，但只上传了{len(paths)}张')
            selected = [paths[idx - 1]]
        job_dir = self.settings.output_dir / ('job_' + uuid.uuid4().hex[:12])
        job_dir.mkdir(parents=True, exist_ok=False)
        records = []
        for file_path in selected:
            try:
                record = self.process_one(
                    str(file_path), plan['review_mode'], job_dir,
                    task=task,
                    allow_qwen_feedback=(plan['source'] == 'qwen_api' and len(selected) <= 8))
            except Exception as exc:
                record = {'file': Path(file_path).name, 'error': str(exc), 'trace': [],
                          'needs_human_review': True, 'recognized_text': ''}
            records.append(record)
        response = {
            'project': '藏智录·藏文OCR智能体',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'planner': plan,
            'records': records,
            'agent_runtime': 'online_qwen' if plan['source'] == 'qwen_api' else 'rule_fallback',
            'ocr_test_cer': None,  # 不把历史3.8%的CER当成当前图片的结果
            'note': '本任务没有参考真值，不能计算CER。识别差异不等于纠错成功。',
        }
        archive_path = None
        if export:
            json_file = job_dir / 'results.json'
            json_file.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding='utf-8')
            txt_file = job_dir / 'recognized.txt'
            txt_file.write_text('\n\n'.join(f"[{rec['file']}]\n{rec['recognized_text']}" for rec in records),
                                encoding='utf-8')
            archive_path = str(job_dir / 'results.zip')
            with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(json_file, arcname='results.json')
                if plan['export_format'] != 'json':
                    zf.write(txt_file, arcname='recognized.txt')
        response['archive_path'] = archive_path
        return response
