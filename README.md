# 智能RAG问答系统 (RAGFlow + Qwen)

这是一个基于 **RAGFlow** 检索服务和 **通义千问 (Qwen)** 大语言模型构建的高级检索增强生成 (RAG) 问答系统。项目采用 FastAPI 框架，并集成了微信公众号服务，提供从知识检索、智能回答到多渠道部署的全方位解决方案。

系统不仅实现了标准的RAG流程，还包含了诸多增强功能，如 **智能检索策略**、**高级引用溯源**、**动态提示词管理** 和 **微信异步应答**，旨在提供更精准、可靠且用户体验更佳的智能问答服务。

## ✨ 核心功能

- **双引擎驱动**:
    - **检索端**: 深度集成 [RAGFlow](https://github.com/infiniflow/ragflow)，实现高效、精准的知识文档检索。
    - **生成端**: 对接阿里云通义千问 (Qwen) 大模型，提供高质量、符合逻辑的自然语言回答。

- **🚀 智能检索策略**:
    - 内置 `SmartRetrievalStrategy` 模块，可根据用户问题的类型（如人名查询、定义解释、方法询问等）自动优化检索参数（如 `top_k`, `similarity_threshold`），从而为不同类型的问题匹配最相关的知识片段。

- **🔗 高级引用溯源与验证**:
    - **内容解析与链接**: 能够从RAGFlow返回的复杂文本中精准分离正文与 `微信链接`、`原文链接`。
    - **智能链接匹配**: 通过 `citation_matcher` 模块，为无链接的知识片段智能匹配最相关的引用来源。
    - **引用验证**: `citation_validator` 模块确保最终答案中引用的来源与内容高度相关，提升答案的可信度。

- **👤 自动人物链接**:
    - `name_linker` 模块能够自动识别回答中的特定人物姓名，并附上预设的个人主页链接，方便用户进行深度探索。

- **💬 微信公众号集成**:
    - 提供独立的 `wechat_service.py`，可作为微信公众号后台服务。
    - 采用 **异步处理 + 客服消息** 机制，完美解决微信5秒超时限制，确保复杂查询也能可靠响应。

- **⚙️ 灵活的配置与管理**:
    - **动态提示词管理**: 通过 `prompt_templates.json` 和管理API (`/prompts/*`)，轻松创建、更新和管理用于不同场景的提示词模板。
    - **全面的API接口**: 提供包括健康检查 (`/health`)、数据集查询 (`/datasets`)、调试工具 (`/debug/*`) 在内的丰富API。
    - **Web界面**: 内置一个基于 `FastAPI` 的简单前端页面，用于快速演示和功能测试。

## 🛠️ 项目结构

```
fuwu/
├── main.py                 # 🚀 FastAPI主应用，包含所有API路由和核心逻辑
├── wechat_service.py       # 💬 微信公众号后台服务
├── 启动脚本.py             # 📦 推荐的服务启动脚本
│
├── ragflow_client.py       # 🔍 RAGFlow客户端，封装检索API
├── qwen_client.py          # 🤖 Qwen大模型客户端，封装生成API
│
├── config.py               # ⚙️ 配置文件加载模块
├── requirements.txt        # 📋 Python依赖列表
│
├── prompt_templates.json   # 📝 提示词模板定义文件
├── cleaned_links.txt       # 📚 （示例）引用链接数据源
├── name_output.txt         # 👤 （示例）人物链接数据源
│
├── citation_matcher.py     # 🔗 智能引用匹配器
├── citation_validator.py   # ✔️ 引用验证器
├── enhanced_content_parser.py # 📄 增强内容解析器
├── name_linker.py          # � 自动人物链接器
│
└── templates/
    └── index.html          # 🎨 Web前端界面
```

## 🚀 快速开始

### 1. 环境准备

确保您已安装 Python 3.8+。

```bash
# 进入项目目录
cd fuwu

# 安装所有依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

在 `fuwu` 目录下，根据 `config.py` 的要求创建您的环境变量配置文件（如 `.env` 文件），并填入以下核心配置：

```env
# RAGFlow API 配置
RAGFLOW_URL="http://<your-ragflow-host>/api/v1/retrieval"
RAGFLOW_API_KEY="<your-ragflow-api-key>"

# Qwen API 配置
QWEN_URL="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_API_KEY="<your-qwen-api-key>"
QWEN_MODEL="qwen-plus"

# 默认知识库ID (可设置多个，用逗号分隔)
DEFAULT_DATASET_IDS="<your-dataset-id>"

# 微信公众号配置 (如果使用wechat_service.py)
WECHAT_TOKEN="<your-wechat-token>"
WECHAT_APPID="<your-wechat-appid>"
WECHAT_APPSECRET="<your-wechat-appsecret>"
```
*提示：请查阅 `config.py` 文件以了解所有可配置的参数。*

### 3. 启动服务

**启动主问答服务:**
```bash
python main.py
```
或者使用 `uvicorn` 以获得更佳性能：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
服务启动后，您可以访问：
- **Web界面**: `http://localhost:8000`
- **API文档 (Swagger UI)**: `http://localhost:8000/docs`

**启动微信后台服务 (如果需要):**
```bash
python wechat_service.py
```
*请确保已正确配置微信相关的环境变量，并将您的公网服务器地址配置到微信公众号后台。*

## 🎯 API 核心接口

### 1. 智能问答

- **Endpoint**: `POST /qa`
- **功能**: 执行一次完整的RAG问答流程。
- **请求体**:
  ```json
  {
    "question": "介绍一下大模型中的RAG技术？",
    "dataset_ids": ["your_dataset_id"],
    "prompt_template": "default"
  }
  ```

### 2. 调试接口

- **RAGFlow调试**: `POST /debug/ragflow`
  - 用于独立测试RAGFlow的检索效果，返回原始的检索文档和分析数据。
- **Qwen调试**: `POST /debug/qwen`
  - 用于独立测试Qwen大模型的生成能力。

### 3. 管理接口

- **健康检查**: `GET /health`
  - 检查主服务、RAGFlow及Qwen服务的连通性。
- **提示词管理**: `GET /prompts`, `POST /prompts`, ...
  - 用于查询、创建、更新和删除提示词模板。

## 🔍 故障排查

- **连接失败**:
    - 运行 `GET /health` 接口，检查各组件的健康状态。
    - 确认 `RAGFLOW_URL` 和 `QWEN_URL` 配置正确且网络可达。
- **认证失败**:
    - 检查 `RAGFLOW_API_KEY` 和 `QWEN_API_KEY` 是否正确且有效。
- **检索不到文档**:
    - 确认 `DEFAULT_DATASET_IDS` 是否正确。
    - 使用 `/debug/ragflow` 接口测试检索过程，检查相似度分数和返回结果。
- **微信服务无响应**:
    - 确认 `wechat_service.py` 正在运行。
    - 检查微信公众号后台的URL和Token配置是否正确。
    - 确认服务器IP已加入微信公众号的IP白名单。
