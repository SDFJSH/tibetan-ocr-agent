"""Production-minded Qwen planning. API uses text only, never image or OCR content."""
import json
import re
from urllib.parse import urlsplit
import requests
from .settings import Settings

VALID_MODES = {'auto', 'force', 'none'}
VALID_EXPORT = {'txt_json', 'json'}

def rule_plan(text):
    text = (text or '').strip()
    if any(x in text for x in ('不要复核', '不需要复核', '无需复核', '直接识别')):
        mode = 'none'
    elif any(x in text for x in ('重新识别', '重新检查', '强制复核', '二次识别')):
        mode = 'force'
    else:
        mode = 'auto'
    matched = re.search(r'第\s*(\d+)\s*张', text)
    export = 'json' if '只导出json' in text.lower().replace(' ', '') else 'txt_json'
    return {'review_mode': mode, 'target_index': int(matched.group(1)) if matched else None,
            'export_format': export, 'source': 'rule_fallback'}

def _endpoint_allowed(url):
    u = urlsplit(url)
    # Endpoint is specified only by the app operator, but still reject SSRF or HTTP.
    host = (u.hostname or '').lower()
    return (u.scheme == 'https' and
            (host == 'dashscope.aliyuncs.com' or
             host.endswith('.maas.aliyuncs.com')) and
            u.path.endswith('/chat/completions') and not u.username and not u.password)

def qwen_endpoint_status(settings):
    if not settings.enable_qwen:
        return '未启用千问（规则模式）'
    if not settings.qwen_api_url or not settings.qwen_api_key:
        return '千问凭据未配齐（规则回退）'
    if not _endpoint_allowed(settings.qwen_api_url):
        return '千问接口地址无效（规则回退）'
    return '千问API已配置，实际调用结果以任务日志为准'

def _parse_json(s):
    if not isinstance(s, str):
        raise ValueError('模型返回了非文本内容')
    decoder = json.JSONDecoder()
    objects = []
    for hit in re.finditer(r'\{', s):
        try:
            obj, _ = decoder.raw_decode(s[hit.start():])
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if not objects:
        raise ValueError('模型未输出有效JSON')
    return objects[-1]

def _ask(settings, system, user):
    if not (settings.enable_qwen and settings.qwen_api_key and _endpoint_allowed(settings.qwen_api_url)):
        raise RuntimeError('千问API未配置完整或地址不被允许')
    payload = {
        'model': settings.qwen_model,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': user[:2000]}],
        'temperature': 0,
        'max_tokens': 180,
    }
    response = requests.post(settings.qwen_api_url, json=payload,
        headers={'Authorization': 'Bearer ' + settings.qwen_api_key}, timeout=settings.qwen_timeout_sec)
    response.raise_for_status()
    body = response.json()
    return _parse_json(body['choices'][0]['message']['content'])

def plan_task(task, settings):
    fallback = rule_plan(task)
    system = ('你是藏智录的工具规划器，只处理用户任务，不进行OCR或翻译，不得编造藏文。'
              '只能返回JSON：{"review_mode":"auto|force|none",'
              '"target_index":null或正整数,"export_format":"txt_json|json"}。'
              'auto=按图像质量判断复核，force=强制增强再识别，none=仅识别。'
              '不要额外字段和解释。')
    try:
        res = _ask(settings, system, task or '识别所有图片并导出结果')
        mode, idx, ex = res.get('review_mode'), res.get('target_index'), res.get('export_format')
        if mode not in VALID_MODES or ex not in VALID_EXPORT:
            raise ValueError('非法处理模式或导出类型')
        if idx is not None and (type(idx) is not int or idx < 1 or idx > 5):
            raise ValueError('非法目标索引')
        # Natural-language safety: explicit user constraints have precedence.
        if any(x in (task or '') for x in ('不要复核', '不需要复核', '无需复核', '直接识别')):
            mode = 'none'
        if any(x in (task or '') for x in ('强制复核', '二次识别')):
            mode = 'force'
        return {'review_mode': mode, 'target_index': idx, 'export_format': ex, 'source': 'qwen_api'}
    except Exception as exc:
        return {**fallback, 'warning': 'API不可用，已自动回退规则模式：' + str(exc)[:130]}

def decide_review_after_tools(task, quality, ocr_text, settings):
    fallback = {'action': 'enhance_retry' if quality.get('needs_enhancement') else 'skip',
                'source': 'rule_fallback'}
    system = ('你是藏文OCR的工具调度员。CRNN已经识别过原图。你只阅读图像质量信息，'
              '不能推测或改写原始藏文，也不能自己OCR。'
              '只返回JSON：{"action":"enhance_retry"}或{"action":"skip"}。')
    observed = {'instruction': (task or '')[:450], 'quality': quality,
                'crnn_length': len(ocr_text), 'ocr_text_not_sent': True}
    try:
        data = _ask(settings, system, json.dumps(observed, ensure_ascii=False))
        action = data.get('action')
        if action not in ('enhance_retry', 'skip'):
            raise ValueError('动作不在白名单')
        return {'action': action, 'source': 'qwen_api'}
    except Exception as exc:
        return {**fallback, 'warning': '复核API不可用，回退质量规则：' + str(exc)[:110]}
