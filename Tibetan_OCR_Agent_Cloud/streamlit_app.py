"""藏智录 v1.2 · Streamlit contest workspace.

CRNN is the only Tibetan OCR engine. Qwen API plans tasks and optionally
makes review/tool decisions based on quality metadata; it never OCRs images.
No knowledge database or translation. OCR results are never silently changed.
"""
from __future__ import annotations

import hmac
import io
import json
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title='藏智录 | 藏文OCR智能体', page_icon='🔎', layout='wide', initial_sidebar_state='expanded')


def secret(name, default=''):
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


# Keep the v1.1 fix: reread Secrets on each rerun rather than frozen defaults.
os.environ['QWEN_API_KEY'] = secret('QWEN_API_KEY', os.getenv('QWEN_API_KEY', ''))
os.environ['QWEN_API_URL'] = secret('QWEN_API_URL', os.getenv('QWEN_API_URL', ''))
os.environ['QWEN_MODEL'] = secret('QWEN_MODEL', os.getenv('QWEN_MODEL', 'qwen3.5-flash'))
os.environ['ENABLE_QWEN'] = '1' if os.environ['QWEN_API_KEY'] and os.environ['QWEN_API_URL'] else '0'

from cloud_crnn import CachedCRNN
from tib_agent.settings import Settings
from tib_agent.pipeline import OCRPipeline
from tib_agent.planner import qwen_endpoint_status, _endpoint_allowed
from tib_agent.competition_tools import (
    char_edit_counts, enhance_preview_bytes, make_review_archive,
    mask_secrets, metrics_csv,
)

STYLE = '''<style>
.stApp {background: #f4f7fa;}
.block-container {max-width:1480px; padding-top:1.45rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {background: #edf4f8; border-right: 1px solid #d9e6ef;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height:1.6;}
.hero {border-radius:20px; padding:28px 34px; background:linear-gradient(115deg,#112b47 5%,#154c65 65%,#107c80); color:white; margin:0 0 20px;}
.hero-eyebrow {color:#a8e0e4;font-size:.75rem;letter-spacing:.16em;font-weight:750;}
.hero h1 {color:#fff !important;font-size:2.05rem !important;margin:.45rem 0 .5rem;line-height:1.4;}
.hero p {color:#e6f5f5; font-size:.94rem; margin:0;}
.chip {display:inline-block;border:1px solid #b7d6df;background:#edf7f9;color:#155a72;padding:5px 12px;border-radius:999px;font-size:.8rem;margin:0 7px 8px 0;}
.panel {background:#fff;border:1px solid #dce7ef;border-radius:16px;padding:18px 20px;margin-bottom:14px;}
.panel h3 {font-size:1.1rem;color:#173d59;margin:0 0 9px;}
.panel p {color:#4d6a80;font-size:.91rem;margin:0;}
.subtle {color:#61788a;font-size:.82rem;}
.step {background:#fff; border:1px solid #d9e5ec; border-left:4px solid #16828a;border-radius:12px;padding:13px 17px;margin-bottom:10px;}
.step-name {font-weight:700;color:#183b54;font-size:.92rem;}
.step-sub {font-size:.82rem;color:#547287;margin-top:4px;overflow-wrap:anywhere;}
body, div, p, span {overflow-wrap:break-word;}
[data-testid="stMetric"] {background:#fff;border-radius:14px;border:1px solid #deebf1;padding:13px 16px;}
[data-testid="stMetricLabel"] {color:#60788d;}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#137d8b,#125a80);border:0;border-radius:9px;}
</style>'''
st.markdown(STYLE, unsafe_allow_html=True)

# Simple independent reviewer authentication. No provider/admin credentials.
expected = secret('REVIEW_PASSWORD', os.getenv('REVIEW_PASSWORD', ''))
if not expected:
    st.error('缺少评审密码：请在Streamlit Secrets中设置 REVIEW_PASSWORD。')
    st.stop()
if not st.session_state.get('authorized', False):
    st.markdown('''<section class="hero"><div class="hero-eyebrow">TIBETAN OCR · REVIEW ACCESS</div>
      <h1>藏智录 · 评审入口</h1><p>藏文识别 · 任务规划 · 智能复核 · 人工校对</p></section>''', unsafe_allow_html=True)
    who = st.text_input('评审账号', value='reviewer')
    passwd = st.text_input('评审密码', type='password')
    if st.button('进入藏智录', type='primary'):
        if hmac.compare_digest(who.strip(), 'reviewer') and hmac.compare_digest(passwd, expected):
            st.session_state['authorized'] = True
            st.rerun()
        else:
            st.error('账号或密码不正确')
    st.stop()

# Create settings only from fresh secrets; preserves confirmed Qwen API access.
settings = Settings(
    enable_qwen=bool(secret('QWEN_API_KEY') and secret('QWEN_API_URL')),
    qwen_api_key=secret('QWEN_API_KEY'),
    qwen_api_url=secret('QWEN_API_URL'),
    qwen_model=secret('QWEN_MODEL', 'qwen3.5-flash'),
)
weight_path = Path(__file__).parent / 'models' / 'best.pt'

@st.cache_resource(show_spinner='正在首次加载CRNN权重…')
def load_crnn(path: str):
    return CachedCRNN(path)

for state_name, default in (
    ('result', None), ('archive_bytes', None), ('image_cache', {}),
    ('corrections', {}), ('evaluations', []), ('jobs_run', 0),
):
    if state_name not in st.session_state:
        st.session_state[state_name] = default

response = st.session_state['result']
records = response.get('records', []) if response else []
successful = [r for r in records if not r.get('error')]

NAV = ['平台总览', '智能OCR工作台', '图像增强与复核', '人工校对中心', 'OCR性能评测', 'Agent执行记录', '批量导出']
with st.sidebar:
    st.markdown('### 🔎 藏智录')
    st.caption('TIBETAN DOCUMENT INTELLIGENCE')
    page = st.radio('功能导航', NAV, label_visibility='collapsed', key='page_nav')
    st.divider()
    st.markdown('**当前服务**')
    st.caption('CRNN + CTC：' + ('已配置' if weight_path.exists() else '缺少权重'))
    st.caption('千问API：' + ('已配置，需以执行记录验证' if settings.enable_qwen else '未配置'))
    if response:
        st.caption('最近任务：' + ('真实千问规划' if response.get('planner', {}).get('source') == 'qwen_api' else '规则回退'))
    st.divider()
    st.caption('应用研究原型 · 不提供藏文自动翻译')
    if st.button('退出评审账号'):
        st.session_state.clear()
        st.rerun()


def hero(title: str, subtitle: str, eyebrow: str = 'TIBETAN DOCUMENT INTELLIGENCE'):
    st.markdown(f'<section class="hero"><div class="hero-eyebrow">{eyebrow}</div>'
                f'<h1>{title}</h1><p>{subtitle}</p></section>', unsafe_allow_html=True)


def get_image_bytes(file):
    return st.session_state['image_cache'].get(file)


def choose_record(label='选择需要查看的图片', key='pick_record'):
    available = [r for r in records if not r.get('error')]
    if not available:
        st.info('请先进入「智能OCR工作台」，上传图像并运行一次识别。')
        return None
    chosen = st.selectbox(label, [r['file'] for r in available], key=key)
    return next(r for r in available if r['file'] == chosen)


def run_job(uploaded, instruction):
    if st.session_state['jobs_run'] >= 20:
        raise ValueError('本次浏览器会话已达到20次处理上限')
    if not uploaded or len(uploaded) > 5:
        raise ValueError('每次需上传1—5张图片')
    st.session_state['jobs_run'] += 1
    model = load_crnn(str(weight_path))
    with tempfile.TemporaryDirectory(prefix='zangzhilu_') as temp:
        root = Path(temp)
        paths, image_map = [], {}
        for index, upload in enumerate(uploaded, 1):
            if upload.size > 8 * 1024 * 1024:
                raise ValueError(f'{upload.name}超过8MB限制')
            suffix = Path(upload.name).suffix.lower()
            if suffix not in ('.png', '.jpg', '.jpeg', '.webp'):
                raise ValueError('仅支持PNG、JPG或WebP文件')
            safe = re.sub('[^A-Za-z0-9_.-]', '_', Path(upload.name).stem)[:45] or 'line'
            path = root / f'{index:02d}_{safe}{suffix}'
            data = upload.getvalue()
            path.write_bytes(data)
            with Image.open(path) as img:
                img.verify()
            image_map[path.name] = data
            paths.append(str(path))
        task_settings = replace(settings, output_dir=root / 'outputs')
        result = OCRPipeline(settings=task_settings, ocr=model).process_many(paths, task=instruction, export=True)
        archive = Path(result['archive_path']).read_bytes()
        result.pop('archive_path', None)
        # Keep only processed images in volatile session state, not filesystem.
        st.session_state['image_cache'] = {r['file']: image_map[r['file']]
                                            for r in result['records'] if r['file'] in image_map}
    st.session_state['result'] = result
    st.session_state['archive_bytes'] = archive
    st.session_state['corrections'] = {}
    st.session_state['evaluations'] = []
    # A new batch must not reuse an old batch's editable text/reference fields.
    for old_key in list(st.session_state.keys()):
        if old_key.startswith(('human_text_', 'human_confirm_', 'reference_')):
            del st.session_state[old_key]
    return result


def diagnostics():
    with st.expander('千问连接诊断（不展示API Key）'):
        st.json({
            '密钥已读取': bool(settings.qwen_api_key),
            '接口地址已读取': bool(settings.qwen_api_url),
            'API地址安全校验通过': _endpoint_allowed(settings.qwen_api_url),
            '千问已启用': settings.enable_qwen,
            '模型': settings.qwen_model,
            '接口状态': qwen_endpoint_status(settings),
        })
        st.caption('配置正确不等于已经调用成功。只有单次任务 planner.source=qwen_api 才可认为本次规划来自千问。')


if page == '平台总览':
    hero('藏智录 · 藏文智能数字化处理平台',
         '以专业CRNN识别为基础，结合千问任务规划、动态复核及人工校对，形成可信的藏文图像处理流程。')
    st.markdown('<span class="chip">CRNN + CTC</span><span class="chip">千问API任务规划</span>'
                '<span class="chip">单行文本识别</span><span class="chip">可追溯复核</span>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric('OCR引擎', 'CRNN' if weight_path.exists() else '未就绪')
    cols[1].metric('Agent接口', '已配置' if settings.enable_qwen else '规则模式')
    cols[2].metric('本次处理文件', len(records))
    cols[3].metric('历史测试CER', '3.8%', help='既有测试集CER，不是当前图片实时精度')
    st.subheader('主要能力')
    features = [
        ('01  智能OCR识别', '支持一次上传1—5张藏文文本行图片，CRNN生成Unicode识别结果。'),
        ('02  图像增强复核', '展示原图与CLAHE增强图、两次OCR候选及差异提醒。'),
        ('03  人工校对中心', '逐图编辑文本、标记人工确认并导出修订稿。'),
        ('04  真实性能评测', '提供人工参考真值后计算CER、替换/插入/删除错误。'),
        ('05  Agent执行可视化', '展示真实千问规划来源、工具调用、复核决策与失败回退。'),
        ('06  批量结果导出', '下载原始CRNN结果或人工校对后的TXT、JSON、CSV。'),
    ]
    for start in range(0, len(features), 2):
        a,b = st.columns(2)
        for col, (name, detail) in zip((a,b), features[start:start+2]):
            with col:
                st.markdown(f'<div class="panel"><h3>{name}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    st.info('请从左侧「智能OCR工作台」开始。千问只负责工具规划，不负责生成或修正藏文OCR文本。')
    diagnostics()

elif page == '智能OCR工作台':
    hero('智能藏文OCR工作台', '上传图像，输入指令，由千问选择处理模式，CRNN执行识别与复核。', 'INTELLIGENT OCR WORKSPACE')
    diagnostics()
    left, right = st.columns([.96, 1.04], gap='large')
    with left:
        st.subheader('01  上传与任务配置')
        uploads = st.file_uploader('选择藏文文本行图像（多选）', type=['png','jpg','jpeg','webp'],
                                   accept_multiple_files=True, key='ocr_uploader',
                                   help='最多5张，每张8MB。复杂整页排版尚未专门训练与验证。')
        if uploads:
            if len(uploads) > 5:
                st.warning('每次最多上传5张')
            else:
                with st.expander('查看待处理图像', expanded=len(uploads) <= 2):
                    for upload in uploads:
                        try:
                            st.image(upload.getvalue(), caption=upload.name, width=340)
                        except Exception:
                            st.warning('无法预览：' + upload.name)
        instruction = st.text_area('自然语言任务', value='识别所有图片，按图像质量决定是否增强复核，导出TXT和JSON。', height=110)
        mode = st.selectbox('快捷任务', ['按自然语言指令处理', '直接识别不复核', '强制复核', '质量驱动自动复核'])
        if mode == '直接识别不复核':
            instruction = '直接识别所有图片，不要复核，导出TXT和JSON。'
        elif mode == '强制复核':
            instruction = '识别所有图片，强制复核，导出TXT和JSON。'
        elif mode == '质量驱动自动复核':
            instruction = '识别所有图片，按图像质量自动决定是否复核，导出TXT和JSON。'
        clicked = st.button('开始处理 →', type='primary', use_container_width=True,
                            disabled=not uploads or len(uploads)>5 or not weight_path.exists())
        st.caption('默认保留CRNN首轮结果；增强结果只是候选，不自动称作纠错成功。')
        if clicked:
            try:
                with st.spinner('执行CRNN识别与Agent规划…'):
                    run_job(uploads, instruction)
                st.success('本次处理已完成。')
            except Exception as e:
                st.error('任务失败：' + str(e)[:280])
    with right:
        st.subheader('02  识别结果与执行分析')
        response = st.session_state.get('result')
        if not response:
            st.info('上传图片后点击「开始处理」，即可查看真实OCR输出及工具执行结果。')
        else:
            current = response['records']
            c1,c2,c3 = st.columns(3)
            c1.metric('处理文件', len(current))
            c2.metric('识别成功', sum(not r.get('error') for r in current))
            c3.metric('待人工校对', sum(r.get('needs_human_review', False) for r in current))
            plan = response.get('planner', {})
            if plan.get('source') == 'qwen_api':
                st.success('本次任务由真实千问API进行初始任务规划。')
            else:
                st.warning('本次为规则回退，不能宣称千问参与本次规划。')
            if plan.get('warning'):
                st.info(plan['warning'])
            tab1,tab2,tab3 = st.tabs(['识别文字', '质量信息', '原始JSON'])
            with tab1:
                for rec in current:
                    st.markdown('**' + rec['file'] + '**')
                    if rec.get('error'):
                        st.error(rec['error'])
                    else:
                        st.code(rec['recognized_text'], language=None)
                        if rec.get('enhanced_ocr_text') is not None:
                            st.caption('本图存在复核OCR候选；请进入「图像增强与复核」比较。')
            with tab2:
                for rec in current:
                    with st.expander(rec['file'], expanded=len(current)==1):
                        st.json(rec.get('quality', {}))
                        st.caption('；'.join(rec.get('review_reasons', [])) or '无复核原因')
            with tab3:
                st.json(mask_secrets(response), expanded=False)
            if st.session_state.get('archive_bytes'):
                st.download_button('下载原始结果ZIP (TXT/JSON)', st.session_state['archive_bytes'],
                                   file_name='zangzhilu_original_results.zip', mime='application/zip',
                                   use_container_width=True)

elif page == '图像增强与复核':
    hero('图像质量与动态复核', '原图与增强图并排对照，CRNN两次识别候选相互比较，不凭差异自动替换。', 'QUALITY & REVIEW')
    record = choose_record(key='quality_pick')
    if record:
        quality = record.get('quality', {})
        a,b,c = st.columns(3)
        a.metric('图像尺寸', f"{quality.get('width', '-')} × {quality.get('height', '-')}")
        b.metric('Laplacian方差', quality.get('sharpness_laplacian_var', '-'))
        c.metric('对比度标准差', quality.get('contrast_std', '-'))
        st.caption('质量指标采用启发式阈值；不是OCR置信度，也不代表识别准确率。')
        if quality.get('flags'):
            st.warning('图像质量标记：' + '、'.join(quality['flags']))
        else:
            st.success('未检测到当前规则覆盖的明显图像质量风险。')
        original = get_image_bytes(record['file'])
        if original:
            p1,p2 = st.columns(2)
            with p1:
                st.markdown('**原始图像**')
                st.image(original, use_container_width=True)
            with p2:
                st.markdown('**CLAHE增强预览图像**')
                try:
                    st.image(enhance_preview_bytes(original), use_container_width=True)
                except Exception as exc:
                    st.error('增强图预览失败：' + str(exc))
                if not record.get('review_performed'):
                    st.caption('仅供画面对照，本次任务并未执行增强图CRNN重识别。')
        st.subheader('识别候选对照')
        x,y = st.columns(2)
        with x:
            st.markdown('**原图CRNN首次识别**')
            st.code(record.get('original_ocr_text', ''), language=None)
        with y:
            st.markdown('**增强图CRNN重识别**')
            if record.get('enhanced_ocr_text') is None:
                st.info('本次任务未触发重识别，暂无第二个OCR候选。')
            else:
                st.code(record['enhanced_ocr_text'], language=None)
        alt = record.get('enhanced_ocr_text')
        if alt is not None:
            if alt != record.get('original_ocr_text'):
                st.warning('两次OCR候选不同，系统已保留原始结果，请人工核对。')
            else:
                st.info('两次候选一致；一致并不能证明文字一定正确。')
        st.write('复核模式：', '已执行增强后重识别' if record.get('review_performed') else '未执行增强后重识别')
        st.write('复核决策来源：', '真实千问API' if record.get('review_decision_source') == 'qwen_api' else '规则/用户模式')
        st.caption('需要人工核对：' + ('是' if record.get('needs_human_review') else '否'))
        st.caption('；'.join(record.get('review_reasons', [])) or '未触发额外复核原因。')

elif page == '人工校对中心':
    hero('人工校对与修订导出', '对照原图逐行修改OCR结果；原始模型输出和人工修改稿独立保存。', 'HUMAN-IN-THE-LOOP')
    if not successful:
        st.info('先在「智能OCR工作台」完成一次真实识别。')
    else:
        st.caption('“已人工确认”必须由校对者主动勾选，不能自动判定。编辑不会修改CRNN模型或训练数据。')
        with st.form('manual_review_form'):
            for i,record in enumerate(successful):
                name = record['file']
                with st.expander(f'{i+1:02d}  {name}', expanded=i==0):
                    col_img,col_text = st.columns([.92,1.08])
                    with col_img:
                        if get_image_bytes(name):
                            st.image(get_image_bytes(name), caption='原始图像', use_container_width=True)
                        st.caption('原始CRNN结果')
                        st.code(record.get('original_ocr_text', ''), language=None)
                    with col_text:
                        field = 'human_text_' + str(i)
                        if field not in st.session_state:
                            st.session_state[field] = st.session_state['corrections'].get(name, {}).get(
                                'text', record.get('recognized_text', ''))
                        st.text_area('可编辑Unicode文本', key=field, height=125)
                        st.checkbox('已由人工对照原图确认', key='human_confirm_'+str(i))
            saved = st.form_submit_button('保存本批人工校对', type='primary')
        if saved:
            st.session_state['corrections'] = {
                rec['file']: {'text': st.session_state['human_text_'+str(i)],
                              'confirmed': st.session_state['human_confirm_'+str(i)]}
                for i,rec in enumerate(successful)
            }
            st.success('人工校对内容已保存在当前浏览器会话，可以导出。')
        if st.session_state['corrections']:
            confirmed = sum(r.get('confirmed', False) for r in st.session_state['corrections'].values())
            st.metric('已人工确认的文本行', f'{confirmed}/{len(successful)}')
            st.download_button('下载人工校对版本（TXT / JSON / CSV）',
                               data=make_review_archive(st.session_state['result'], st.session_state['corrections']),
                               file_name='zangzhilu_human_review.zip', mime='application/zip',
                               use_container_width=True)
        st.info('校对结果只保存在浏览器会话内；刷新部署或更换设备可能失去状态，请及时导出。')

elif page == 'OCR性能评测':
    hero('OCR性能评估', '只有输入人工参考真值后才能计算本批CER；默认采用NFC规范化，保留藏文标点与音节分隔符。', 'EVALUATION & METRICS')
    if not successful:
        st.info('先在「智能OCR工作台」识别图片，再输入独立核对的参考文字。')
    else:
        st.info('历史CER 3.8%来自此前测试集。这里的评测只基于本次填写的参考真值，不自动继承历史数值。')
        with st.form('eval_form'):
            eval_source = st.radio('比较对象', ['CRNN原始输出', '人工校对后文本'], horizontal=True)
            for index,record in enumerate(successful):
                name=record['file']
                st.markdown('**' + name + '**')
                st.caption('模型原始输出：' + record.get('original_ocr_text', ''))
                st.text_area('人工参考真值（没有真值可留空）', key='reference_'+str(index), height=76)
            calculate = st.form_submit_button('计算本批CER', type='primary')
        if calculate:
            rows=[]
            for index,record in enumerate(successful):
                reference = st.session_state.get('reference_'+str(index), '').strip()
                if not reference:
                    continue
                if eval_source == '人工校对后文本':
                    correction = st.session_state['corrections'].get(record['file'])
                    if correction is None:
                        st.warning(f"{record['file']} 尚未保存人工校对稿，已跳过该样本")
                        continue
                    predicted = correction['text']
                else:
                    predicted = record.get('original_ocr_text', '')
                try:
                    m = char_edit_counts(predicted, reference)
                except ValueError as err:
                    st.warning(record['file'] + '：' + str(err))
                    continue
                rows.append({'file':record['file'], 'prediction_source':eval_source,
                             'reference_chars':m['reference_chars'], 'predicted_chars':m['predicted_chars'],
                             'substitutions':m['substitutions'], 'deletions':m['deletions'],
                             'insertions':m['insertions'], 'edit_distance':m['edit_distance'],
                             'cer_percent':round(100*m['cer'], 4)})
            st.session_state['evaluations'] = rows
        rows = st.session_state.get('evaluations', [])
        if rows:
            total_errors=sum(r['edit_distance'] for r in rows)
            total_ref=sum(r['reference_chars'] for r in rows)
            c1,c2,c3=st.columns(3)
            c1.metric('本批加权CER', f'{100*total_errors/total_ref:.2f}%')
            c2.metric('已评估样本', len(rows))
            c3.metric('编辑错误总数', total_errors)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption('CER = 所有参考字符对应的最小编辑错误总数 / 全部参考字符数。S/D/I拆分可能受最优对齐方式影响。')
            st.download_button('下载本批评估CSV', metrics_csv(rows), file_name='zangzhilu_cer_report.csv',
                               mime='text/csv', use_container_width=True)
        else:
            st.caption('尚无本批CER结果。请输入至少一条经过人工核对的参考真值并点击计算。')

elif page == 'Agent执行记录':
    hero('Agent工具调用与执行轨迹', '真实记录任务规划、CRNN识别、图像分析、增强复核，以及API离线时的规则回退。', 'TRACEABILITY & TOOL ROUTING')
    if not response:
        st.info('请先识别图片，执行记录会自动生成。')
    else:
        plan=response.get('planner', {})
        source=plan.get('source', 'unknown')
        if source == 'qwen_api':
            st.success('本次初始任务规划来源：真实千问API。')
        else:
            st.warning('本次初始任务规划来源：规则回退，并非千问。')
        cols=st.columns(3)
        cols[0].metric('复核策略', {'auto':'按质量判断','force':'强制复核','none':'不复核'}.get(plan.get('review_mode'),'未知'))
        cols[1].metric('选择图片', '全部' if plan.get('target_index') is None else '第'+str(plan['target_index'])+'张')
        cols[2].metric('导出模式', plan.get('export_format','-'))
        if plan.get('warning'):
            st.warning('规划器反馈：'+str(plan['warning']))
        st.subheader('实际执行时间线')
        for rec in records:
            with st.expander('📄 '+rec['file'],expanded=len(records)==1):
                if rec.get('error'):
                    st.error('图像处理失败：'+rec['error'])
                trace=rec.get('trace', [])
                if not trace:
                    st.caption('没有可展示的工具执行记录。')
                for i,entry in enumerate(trace,1):
                    label={'analyze_image':'图像质量分析','crnn_ocr':'CRNN文字识别',
                           'agent_review_decision':'Agent复核决策',
                           'agent_fallback':'API失败回退','enhance_image':'图像增强',
                           'candidate_selection':'OCR候选选择'}.get(entry.get('tool'),entry.get('tool','工具'))
                    st.markdown(f'**{i:02d}. {label}**')
                    detail=entry.get('detail','')
                    if isinstance(detail, (dict,list)):
                        st.json(mask_secrets(detail),expanded=False)
                    else:
                        st.caption(str(mask_secrets(detail)))
                if rec.get('review_performed'):
                    st.caption('复核决策来源：'+('千问API' if rec.get('review_decision_source')=='qwen_api' else '规则/强制指令'))
        with st.expander('查看原始执行JSON'):
            st.json(mask_secrets(response),expanded=False)
        st.caption('任务日志呈现实际已调用的工具，不将规划中未执行的操作冒充为已完成。')

elif page == '批量导出':
    hero('批量结果归档', '分别获取OCR原始结果、人工校对记录和真实参考真值评测结果。', 'EXPORT & ARCHIVE')
    if not response:
        st.info('暂无任务记录。请先在智能OCR工作台执行识别。')
    else:
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="panel"><h3>原始识别结果</h3><p>保留CRNN模型原始输出、工具执行记录及任务规划来源。</p></div>',unsafe_allow_html=True)
            if st.session_state.get('archive_bytes'):
                st.download_button('下载原始TXT/JSON压缩包',st.session_state['archive_bytes'],
                                   file_name='zangzhilu_original.zip',mime='application/zip',use_container_width=True)
        with c2:
            st.markdown('<div class="panel"><h3>人工校对记录</h3><p>人工修改版与原始OCR分开标记，提供TXT/JSON/CSV。</p></div>',unsafe_allow_html=True)
            if st.session_state.get('corrections'):
                st.download_button('下载人工校对压缩包',
                                   make_review_archive(response,st.session_state['corrections']),
                                   file_name='zangzhilu_reviewed.zip',mime='application/zip',use_container_width=True)
            else:
                st.info('尚未保存校对记录。')
        if st.session_state.get('evaluations'):
            st.download_button('下载CER评估CSV',metrics_csv(st.session_state['evaluations']),
                               file_name='zangzhilu_cer.csv',mime='text/csv',use_container_width=True)
        st.caption('导出包不包含原始上传图像和API密钥。为保护数据，不提供跨用户共享的历史文件数据库。')

st.divider()
st.caption('参赛研究原型 · 不含机构、学生或指导教师身份信息 · 主要支持藏文单行图像。'
           ' 历史测试CER为3.8%，不等于当前图像准确率。不要上传未授权或敏感文档。'
           ' 第三方平台的数据保留政策以运营方条款为准。')
