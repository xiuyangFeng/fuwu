# RAGFlow + Qwen 智能问答系统 v2.1

> 🚀 一个功能强大的增强检索生成（RAG）问答系统。它深度整合了 RAGFlow 的高效检索能力和通义千问（Qwen）大模型的强大语言能力，并在此基础上构建了**智能检索策略**、**高级引用溯源**、**自动人物链接**和**动态配置管理**等一系列增强功能，为用户提供更精准、更智能、可追溯的问答体验。

## ✨ 系统核心特性

-   **🧠 智能检索策略**: 系统能够自动分析用户问题的类型（如人名查询、方法询问、定义查询等），并动态调整检索参数（相似度阈值、Top-K、向量权重），以获取最相关的上下文，显著提升答案的精准度。
-   **🔗 高级引用溯源与验证**:
    -   **内容-链接映射**: 通过 `cleaned_links.txt` 构建内容与引用链接的索引。
    -   **精准匹配**: 利用关键词、文本相似度、实体识别等多种策略，为无链接的文本内容智能匹配最相关的引用链接。
    -   **引用验证**: 验证回答中引用的链接与内容是否匹配，确保答案的可信度和可追溯性。
    -   **专用提示词模板**: 内置 `enhanced_citation` 模板，引导大模型生成格式规范、内容准确的参考文献。
-   **👤 自动人物链接**:
    -   **智能识别**: 自动从问题和回答中识别预定义的人物姓名（基于 `name_output.txt`）。
    -   **链接追加**: 在回答末尾自动附加相关人物的个人主页链接，方便用户进行深度探索。
-   **⚡ 实时流式响应**: 通过 `/qa/stream` 接口提供 Server-Sent Events (SSE) 支持，实现AI回答的实时流式输出，显著提升用户交互体验。
-   **⚙️ 动态配置管理**:
    -   **热重载**: 支持通过 API (`/config/reload`) 动态重新加载 `.env` 配置文件，无需重启服务。
    -   **动态更新**: 支持通过 API (`/config/datasets`) 动态更新知识库ID，并可选择持久化到 `.env` 文件。
-   **📝 灵活的提示词管理**: 通过 `prompt_templates.json` 和管理API (`/prompts/*`)，轻松创建、更新和管理用于不同场景的提示词模板。
-   **🛠️ 强大的调试工具**: 提供独立的 RAGFlow 和 Qwen 调试接口 (`/debug/ragflow`, `/debug/qwen`)，方便开发者对检索和生成环节进行精细化分析。
-   **🌐 现代化Web界面**: 提供一个简洁易用的前端界面，用于快速测试和演示系统功能。

## 🛠️ 项目结构

```
服务端2/
├── main.py                 # 🚀 FastAPI主应用，包含所有API路由和核心逻辑
├── config.py               # ⚙️ 动态配置管理模块，从.env加载并支持热重载
├── ragflow_client.py       # 🔍 RAGFlow客户端，封装检索API调用
├── qwen_client.py          # 🤖 Qwen大模型客户端，封装聊天API调用（支持流式）
├── citation_matcher.py     # 🔗 智能引用匹配器
├── citation_validator.py   # ✔️ 引用验证器
├── enhanced_content_parser.py # 📄 增强内容解析器
├── name_linker.py          # 👤 自动人物链接器
├── 启动脚本.py             # 📦 推荐的启动脚本，包含环境检查
├── requirements.txt        # 📋 Python依赖列表
├── prompt_templates.json   # 📝 自定义提示词模板
├── cleaned_links.txt       # 📚 引用链接数据源
├── name_output.txt         # 👤 人物姓名与主页链接数据源
├── .env                    # 🔐 环境变量配置文件
├── templates/
│   └── index.html          # 🎨 Web前端界面
└── README.md               # 📖 本文档
```

## 🚀 快速开始

### 1. 环境准备

确保你已安装 Python 3.8+。

```bash
# 进入项目目录
cd 服务端2

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

在项目根目录下创建一个名为 `.env` 的文件，并填入以下内容。请将`your_..._key_here`替换为你的真实密钥。

```env
# === 必需配置 ===
# RAGFlow API配置
RAGFLOW_URL=http://localhost:80/api/v1/retrieval
RAGFLOW_API_KEY=your_ragflow_api_key_here

# Qwen API配置 (阿里云通义千问)
QWEN_API_KEY=your_qwen_api_key_here
QWEN_MODEL=qwen-plus

# 默认使用的数据集ID (可设置多个，用逗号分隔)
DEFAULT_DATASET_IDS=your_dataset_id_here

# === 可选配置 ===
# 服务器配置
HOST=0.0.0.0
PORT=8001
DEBUG=true
LOG_LEVEL=INFO

# 检索参数
DEFAULT_SIMILARITY_THRESHOLD=0.1
DEFAULT_TOP_K=15
DEFAULT_VECTOR_WEIGHT=0.7
MAX_CHUNK_LENGTH=1200

# API服务超时设置 (秒)
RAGFLOW_TIMEOUT=30
QWEN_TIMEOUT=120
QWEN_STREAM_TIMEOUT=300 # 流式输出专用超时
```

### 3. 启动服务

推荐使用增强的启动脚本，它会自动检查环境和配置。

```bash
python 启动脚本.py
```

服务启动后，你可以访问：
-   **Web界面**: `http://localhost:8001`
-   **API文档 (Swagger UI)**: `http://localhost:8001/docs`

## 🎯 API 接口指南

### 1. 智能问答 (标准/流式)

-   **标准模式**: `POST /qa`
-   **流式模式**: `POST /qa/stream`

这两个接口执行完整的 RAG 流程。流式接口通过 SSE 协议实时返回数据块。

**请求示例**:
```json
{
    "question": "介绍一下张三的研究方向？",
    "dataset_ids": ["your_dataset_id"],
    "prompt_template": "enhanced_citation",
    "top_k": 10
}
```

**响应示例 (标准模式)**:
```json
{
    "answer": "张三主要研究深度学习... \n\n---\n\n👤 相关人物主页 (1个)\n\n1️⃣ 张三 - [点击访问张三的个人主页](http://example.com/zhangsan)\n",
    "question": "介绍一下张三的研究方向？",
    "context_used": true,
    "ragflow_docs_count": 10,
    "ragflow_status": "normal",
    "prompt_template_used": "enhanced_citation",
    "citations": [
        {
            "source": "来源描述...",
            "link": "https://s.caixuan.cc/xxxx"
        }
    ],
    "performance_stats": { ... }
}
```

### 2. 智能检索分析

**Endpoint**: `POST /analyze_question`

该接口用于展示智能检索策略。它会分析问题类型并返回优化后的检索参数。

**请求示例**:
```json
{
    "question": "什么是Transformer模型？"
}
```

**响应示例**:
```json
{
    "question_type": "definition_query",
    "original_params": { ... },
    "optimized_params": {
        "similarity_threshold": 0.08,
        "vector_similarity_weight": 0.6,
        "top_k": 12
    },
    "optimization_description": "定义查询：平衡关键词与语义权重"
}
```

### 3. 配置与状态管理

-   `GET /config`: 获取当前所有配置。
-   `POST /config/reload`: 重新加载 `.env` 文件。
-   `GET /config/datasets`: 获取当前数据集配置。
-   `PUT /config/datasets`: 更新数据集ID（支持持久化）。
-   `GET /stats`: 获取系统统计信息。
-   `GET /health`: 检查所有组件的健康状态。

### 4. 提示词管理

-   `GET /prompts`: 列出所有提示词模板。
-   `POST /prompts`: 创建新模板。
-   `PUT /prompts/{prompt_name}`: 更新指定模板。
-   `DELETE /prompts/{prompt_name}`: 删除指定模板。

## 🔍 故障排查

-   **连接失败**:
    -   检查 `.env` 文件中的 `RAGFLOW_URL` 是否正确。
    -   确认 RAGFlow 服务正在运行且网络可达。
    -   运行 `GET /health` 接口检查各组件健康状态。
-   **认证失败**:
    -   检查 `RAGFLOW_API_KEY` 和 `QWEN_API_KEY` 是否正确且未过期。
-   **检索不到文档**:
    -   确认 `DEFAULT_DATASET_IDS` 是否正确。
    -   使用 `/debug/ragflow` 接口测试检索过程，检查相似度分数。
    -   尝试在请求中动态调整 `similarity_threshold` 的值。
-   **中文乱码**:
    -   本系统已针对UTF-8进行优化。如仍有问题，请确保你的终端或客户端使用UTF-8编码。
