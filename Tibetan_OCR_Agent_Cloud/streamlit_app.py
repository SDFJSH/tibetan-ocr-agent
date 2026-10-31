"""藏智录 · 评审云端版。CRNN CPU-only, Qwen API for task planning only.

No large local Qwen model, no knowledge base, no automatic translation.
Uploaded images and temporary output files are removed after each completed request.
"""
import hmac
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="藏智录 | 藏文OCR智能体", page_icon="🔎", layout="wide")

# Read server-side secrets without writing or logging them.
def secret(name, default=""):
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

os.environ['QWEN_API_KEY'] = secret('QWEN_API_KEY', os.getenv('QWEN_API_KEY', ''))
os.environ['QWEN_API_URL'] = secret('QWEN_API_URL', os.getenv('QWEN_API_URL', ''))
os.environ['QWEN_MODEL'] = secret('QWEN_MODEL', os.getenv('QWEN_MODEL', 'qwen3.5-flash'))
os.environ['ENABLE_QWEN'] = '1' if os.environ['QWEN_API_KEY'] and os.environ['QWEN_API_URL'] else '0'

from cloud_crnn import CachedCRNN
from tib_agent.settings import Settings
from tib_agent.pipeline import OCRPipeline
from tib_agent.planner import qwen_endpoint_status

CSS = """
<style>
.block-container {max-width:1280px; padding-top:1.3rem;}
.z-hero {padding:27px 34px; border-radius:19px; background:linear-gradient(120deg,#122942,#194760 65%,#0c777f); color:white; margin-bottom:20px;}
.z-eyebrow {font-size:.78rem;letter-spacing:.16em;color:#a0dce0;font-weight:700;}
.z-hero h1 {color:white;font-size:2rem;margin:.3rem 0 .4rem;}
.z-hero p {color:#dcf3f5;margin-bottom:0;font-size:.93rem;}
.z-chip {border-radius:999px;border:1px solid #bcdde2;background:#edf7f9;color:#125775;display:inline-block;padding:5px 12px;margin:2px 5px 2px 0;font-size:.81rem;}
.z-note {font-size:.83rem;color:#5e7384;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Password is independent of cloud hosting account and API key.
expected = secret('REVIEW_PASSWORD', os.getenv('REVIEW_PASSWORD', ''))
if not expected:
    st.error('尚未配置评审密码。请在Streamlit Cloud的Secrets中设置 REVIEW_PASSWORD。')
    st.stop()
if not st.session_state.get('authorized', False):
    st.markdown('<div class="z-hero"><div class="z-eyebrow">TIBETAN DOCUMENT INTELLIGENCE</div><h1>藏智录 · 智能评审入口</h1><p>藏文OCR识别 · 动态复核 · 可追溯任务执行</p></div>', unsafe_allow_html=True)
    user = st.text_input('评审账号', value='reviewer')
    password = st.text_input('评审密码', type='password')
    if st.button('进入智能体', type='primary'):
        if hmac.compare_digest(user.strip(), 'reviewer') and hmac.compare_digest(password, expected):
            st.session_state['authorized'] = True
            st.rerun()
        else:
            st.error('账号或密码不正确')
    st.stop()

st.markdown('''<div class="z-hero">
<div class="z-eyebrow">TIBETAN OCR · AI AGENT</div>
<h1>藏智录 · 藏文 OCR 智能处理平台</h1>
<p>CRNN专业识别 · 图像质量分析 · 动态复核 · 千问API任务规划 · Unicode文本导出</p>
</div>''', unsafe_allow_html=True)

settings = Settings()
weight_path = Path(__file__).parent / 'models' / 'best.pt'
agent_status = qwen_endpoint_status(settings)
st.markdown(
    '<span class="z-chip">CRNN + CTC</span>'
    '<span class="z-chip">CPU推理</span>'
    '<span class="z-chip">不翻译 · 无知识数据库</span>'
    '<span class="z-chip">云端评审版</span>', unsafe_allow_html=True)

m1,m2,m3 = st.columns(3)
m1.metric('OCR识别', '真实CRNN' if weight_path.exists() else '缺少权重')
m2.metric('Agent规划', '千问API已配置' if settings.enable_qwen else '规则回退')
m3.metric('历史测试集CER', '3.8%', help='用户此前实验结果，非当前图片实时准确率')
st.caption('模型说明：千问云API负责规划与工具选择；所有藏文文字均由CRNN产生。')

@st.cache_resource(show_spinner='首次加载CRNN权重中…')
def load_crnn(path: str):
    return CachedCRNN(path)

left, right = st.columns([0.95, 1.05], gap='large')
with left:
    st.subheader('01  上传与任务配置')
    images = st.file_uploader('上传藏文文本行图片（可多选）', type=['png', 'jpg', 'jpeg', 'webp'],
                              accept_multiple_files=True,
                              help='第一版主要支持单行藏文图片；一次最多5张、每张8MB。')
    if images:
        if len(images) > 5:
            st.warning('本评审版每次最多处理5张，请减少上传数量。')
        else:
            for im in images:
                try:
                    im.seek(0)
                    with Image.open(im) as opened:
                        opened.verify()
                    im.seek(0)
                    st.image(im, caption=im.name, use_container_width=True)
                except (UnidentifiedImageError, OSError, ValueError):
                    st.warning('无法预览: ' + im.name)
    task = st.text_area('自然语言任务', value='识别这些藏文图片，如果图像质量较差则尝试增强与复核，最后导出TXT和JSON。', height=110)
    preset = st.selectbox('快捷任务', ['按文字指令处理', '只识别，不复核', '强制复核', '自动复核并导出'])
    if preset == '只识别，不复核':
        task = '直接识别所有图片，不要复核，并导出TXT和JSON。'
    elif preset == '强制复核':
        task = '识别全部图片，强制复核，导出TXT和JSON。'
    elif preset == '自动复核并导出':
        task = '识别全部图片，按图像质量决定是否增强复核，导出TXT和JSON。'
    run = st.button('开始处理 →', type='primary', use_container_width=True,
                    disabled=not images or len(images) > 5 or not weight_path.exists())
    st.markdown('<div class="z-note">系统默认保留CRNN首次识别结果。若增强前后有差异，仅提示人工核对，不自动宣称纠错成功。</div>', unsafe_allow_html=True)

with right:
    st.subheader('02  识别结果与执行分析')
    if 'result' not in st.session_state:
        st.info('上传藏文图像并点击“开始处理”，此处将展示识别结果、质量分析与Agent执行记录。')
    else:
        response = st.session_state['result']
        records = response['records']
        a,b,c = st.columns(3)
        a.metric('处理文件数', len(records))
        b.metric('成功识别', sum('error' not in r for r in records))
        c.metric('待人工核对', sum(r.get('needs_human_review', False) for r in records))
        if response['planner']['source'] == 'qwen_api':
            st.success('本次任务由真实千问API完成任务规划。')
        else:
            st.warning('本次使用规则回退规划；不能视为千问已参与决策。')
        if response['planner'].get('warning'):
            st.info(response['planner']['warning'])
        t1,t2,t3,t4 = st.tabs(['识别文本', '图像质量', '执行步骤', '原始JSON'])
        with t1:
            for r in records:
                st.markdown('**' + r['file'] + '**')
                if r.get('error'):
                    st.error(r['error'])
                else:
                    st.code(r['recognized_text'], language=None)
                    if r.get('enhanced_ocr_text') is not None:
                        with st.expander('查看增强图的OCR候选'):
                            st.code(r['enhanced_ocr_text'], language=None)
        with t2:
            for r in records:
                if 'quality' in r:
                    with st.expander(r['file'], expanded=len(records)==1):
                        st.json(r['quality'])
                        st.write('需要人工复核：', r.get('needs_human_review', False))
                        st.write('复核原因：', '；'.join(r.get('review_reasons', [])) or '无')
        with t3:
            st.json({'planner': response['planner'],
                     'tool_calls': [{ 'file': r['file'], 'trace': r.get('trace', []) } for r in records]},
                    expanded=False)
        with t4:
            st.json({k:v for k,v in response.items() if k != 'archive_path'}, expanded=False)
        if st.session_state.get('archive_bytes'):
            st.download_button('下载识别结果ZIP（TXT/JSON）',
                data=st.session_state['archive_bytes'],
                file_name='tibetan_ocr_results.zip', mime='application/zip',
                use_container_width=True)

if run:
    try:
        if st.session_state.get('jobs_run', 0) >= 20:
            raise ValueError('本次浏览器会话已达到20次处理上限，请联系作品提供方。')
        st.session_state['jobs_run'] = st.session_state.get('jobs_run', 0) + 1
        if len(images) > 5:
            raise ValueError('一次最多5张')
        with st.spinner('执行图像分析、CRNN识别和智能体任务规划…'):
            model = load_crnn(str(weight_path))
            with tempfile.TemporaryDirectory(prefix='tibagent_') as temp_dir:
                temp_root = Path(temp_dir)
                items = []
                for i, upload in enumerate(images):
                    if upload.size > 8 * 1024 * 1024:
                        raise ValueError(f'{upload.name}超过8MB上限')
                    suffix = Path(upload.name).suffix.lower()
                    if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
                        raise ValueError('仅支持PNG/JPG/WebP图片')
                    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(upload.name).stem)[:50] or 'image'
                    target = temp_root / f'{i+1:02d}_{safe_name}{suffix}'
                    target.write_bytes(upload.getvalue())
                    # PIL verification also rejects most malformed files.
                    with Image.open(target) as check:
                        check.verify()
                    items.append(str(target))
                settings_job = replace(settings, output_dir=temp_root / 'outputs')
                result = OCRPipeline(settings=settings_job, ocr=model).process_many(items, task=task, export=True)
                zip_bytes = Path(result['archive_path']).read_bytes()
                result.pop('archive_path', None)
        st.session_state['result'] = result
        st.session_state['archive_bytes'] = zip_bytes
        st.rerun()
    except Exception as exc:
        st.error('处理失败：' + str(exc)[:400])

st.divider()
st.caption('本系统为评审研究原型。CRNN历史CER=3.8%不代表当前图片准确率；质量阈值待进一步验证。'
           '请勿上传敏感材料或未经授权的个人信息。上传文件仅用于本次服务端处理；'
           '第三方云平台自身的日志与数据保留政策以其服务条款为准。')
