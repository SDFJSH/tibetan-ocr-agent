"""把用户已有的CRNN单图推理脚本当成一个CLI工具调用。

配置 OCR_COMMAND，例如：
OCR_COMMAND="python /path/to/infer.py --image {image} --weights {weights}"

原脚本最终最好打印单行：OCR_RESULT_JSON={"text":"བོད་ཡིག་","confidence":0.91}
否则，若所有 stdout 只有识别文字，也可直接读取纯文本。
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path
from .settings import Settings


class OCRBridge:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _parse_output(stdout: str) -> dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError('推理进程未输出识别结果')
        marked = [line.split('OCR_RESULT_JSON=', 1)[1] for line in lines
                  if line.startswith('OCR_RESULT_JSON=')]
        if marked:
            payload = json.loads(marked[-1])
        else:
            try:
                payload = json.loads(stdout.strip())
            except json.JSONDecodeError:
                # 仅在stdout没有模型日志、且只有一行时允许纯文本
                if len(lines) != 1:
                    raise RuntimeError('OCR stdout包含多行日志；请在推理脚本末尾打印 OCR_RESULT_JSON=...')
                payload = {'text': lines[0]}
        if isinstance(payload, str):
            payload = {'text': payload}
        if not isinstance(payload, dict) or not isinstance(payload.get('text'), str):
            raise RuntimeError('OCR输出必须包含字符串字段 text')
        conf = payload.get('confidence')
        if conf is not None:
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = None
            if conf is not None and not 0 <= conf <= 1:
                conf = None
        return {'text': payload['text'], 'confidence': conf}

    def recognize(self, image_path: str) -> dict:
        if self.settings.demo_mock_ocr:
            # 只为验证软件交互流程，绝非真实OCR结果
            return {
                'text': 'བོད་ཡིག་', 'confidence': None,
                'demo_only': True,
                'warning': '演示桩数据！未运行真实CRNN模型，不得用于比赛结果或实验统计。',
            }
        if not self.settings.ocr_command:
            raise RuntimeError('尚未配置OCR_COMMAND：请先把现有CRNN推理脚本接入，参见README。')
        command = shlex.split(self.settings.ocr_command)
        command = [p.replace('{image}', str(image_path))
                     .replace('{weights}', self.settings.ocr_weights)
                     .replace('{python}', sys.executable) for p in command]
        # 安全边界：绝不使用shell=True，也不允许大模型生成执行命令。
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=self.settings.ocr_timeout_sec, check=False,
                                  encoding='utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            raise RuntimeError(f'OCR运行超时（>{self.settings.ocr_timeout_sec}秒）')
        except OSError as exc:
            raise RuntimeError(f'无法执行OCR_COMMAND：{exc}') from exc
        if proc.returncode != 0:
            raise RuntimeError(f'OCR推理失败（退出码{proc.returncode}）：{proc.stderr[-700:]}')
        result = self._parse_output(proc.stdout)
        result['demo_only'] = False
        return result
