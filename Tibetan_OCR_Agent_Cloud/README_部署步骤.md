# 藏智录：Streamlit免费云 + CRNN + 千问API

**这是移植版v1，尚未使用用户真实 best.pt 在Streamlit正式验证。**

## 1. 需要的文件

- 本包里的 `streamlit_app.py`、`cloud_crnn.py`、`tib_agent/`、`requirements.txt`、`.streamlit/config.toml`
- 你自己训练的 `/root/autodl-tmp/runs/crnn_final/best.pt`（本包不包含）
- 阿里云百炼API Key和同地域兼容接口地址

**不用上传**：8.3GB千问权重、LoRA文件、百万张图片、真实用户数据、AutoDL的原始训练环境。

## 2. 先在AutoDL确认云端推理代码与原有CRNN一致

下载原有权重到包内 `models/best.pt`，或在AutoDL直接运行：

```bash
cd /path/to/本包
python verify_crnn.py --weights /root/autodl-tmp/runs/crnn_final/best.pt --image /root/autodl-tmp/runs/0001.png
```

如果输出与此前 `scripts/crnn_infer.py` 输出不同，**不要上线**，先核对预处理、字符表及网络定义。由于当前未拥有你的权重，无法替你完成这个验证。

## 3. 创建GitHub仓库

- 使用独立项目名，例如 `tibetan-ocr-agent-review`，仓库不要包含学校、参赛人姓名或指导教师信息。
- 可以建立私有GitHub仓库，授权Streamlit从私有仓库部署，但为了让评委访问，应在Streamlit App中配置公开网页访问或可用的评审访问方式。
- 上传解压后的全部代码、`.streamlit/config.toml` 和 `models/best.pt`。
- 不上传 `.streamlit/secrets.toml`、`.env` 或任何API Key。
- GitHub单文件超过100MB不能直接普通上传，应使用Git LFS或另行安排权重分发，不要把权重放在不可靠的公网下载链接。

## 4. 千问API（代替本地Qwen3-VL-4B）

用阿里云百炼创建API Key；确定region和WorkspaceId对应。
在Streamlit的 **Advanced settings → Secrets** 配置以下内容（不要写进GitHub）：

```toml
REVIEW_PASSWORD = "你设置的独立评委密码"
QWEN_API_KEY = "你的百炼密钥"
QWEN_API_URL = "https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen3.5-flash"
```

这次使用的是**阿里云托管qwen3.5-flash**，不是之前本地的原始Qwen3-VL-4B。为了准确，不应在参赛材料中写“云端运行了本人本地Qwen3-VL-4B”，可写“基于千问API的任务规划”。

注意：Qwen API目前只接收文字任务指令与图像质量数值/识别字符数量，不上传藏文图片，也不让Qwen生成藏文识别文字。API断开时网页明确显示规则回退模式。

## 5. 部署Streamlit

1. 访问 https://share.streamlit.io/，登录并连接GitHub。
2. 选择 **Create app → Yup, I have an app**。
3. 选择仓库和分支；**Main file path** 填 `streamlit_app.py`。
4. Advanced settings中选择可用的Python 3.11，粘贴Secrets。
5. 部署成功后获得 `https://你的子域名.streamlit.app/`。
6. 使用手机流量/其他网络打开此网址，以账号 `reviewer` 和你设定的密码登录。
7. 上传单行藏文图片，查看是否为真实CRNN；输入多个自然语言任务，查看原始JSON中的 `planner.source` 是否为 `qwen_api`。

## 6. 免费服务风险及前置测试

- Community Cloud现行公开资源限制约2 vCPU、最高约2.7GB RAM。CPU运行PyTorch可能因内存超限失败；若发生OOM，不能宣称已部署成功。此时可尝试精简Torch依赖或申请教育公益应用资源，若仍不适合需换平台。
- 评审期间免费平台可能休眠或冷启动，需在提交前和评审期间检查，不能保证24小时运行。
- 不要把CRNN测试集3.8% CER写成云端实时准确率。
- 评审账号只能访问网页，不应提供服务器root账号或API Key。
- 公开网站产生的Qwen API费用由你承担；建议设置用量预算并限额。本版限制单批最多5张图片，避免大量调用。

## 7. 比赛匿名要求

平台作品名、PPT正文、视频、网页展示、图片文件名、备注等均不要出现学校、学生和指导教师信息；提交文件名如按官方要求需学校/队长姓名，则仅在允许的文件名字段体现，并严格保持作品内容匿名。

## 8. 本包保留了哪些旧版功能

- 你的CRNN架构、CTC贪心解码、图像预处理与原图优先策略
- 图像质量评估（Laplacian方差、对比度）和CLAHE增强候选
- 按用户指令的Qwen任务规划、质量反馈后复核决策
- 差异标记、人工核对提示、ZIP导出TXT/JSON

旧Gradio美化版UI不直接搬过去，本包改用Streamlit构建类似的两栏布局，这是为了适配免费云托管。
