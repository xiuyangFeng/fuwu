import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from ragflow_client import ragflow_client, query_ragflow
from qwen_client import qwen_client, query_qwen
from config import config
from enhanced_content_parser import enhanced_parser, parse_content_and_link
from citation_matcher import citation_matcher
from citation_validator import citation_validator, validate_response_citations
from name_linker import name_linker
import os
import sys

# 设置UTF-8编码环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.utf8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.utf8')
        except:
            pass

# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent.absolute()

def setup_logging():
    """配置增强版日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)

    # 创建支持UTF-8的StreamHandler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))

    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[handler],
        force=True  # 强制重新配置
    )

    # 为第三方库设置合适的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    return logging.getLogger(__name__)

logger = setup_logging()

# 初始化智能引用匹配器
def initialize_citation_system():
    """初始化智能引用匹配系统"""
    try:
        logger.info("正在初始化智能引用匹配系统...")
        if config.ENABLE_CITATION_MATCHER:
            citation_matcher.load_links_data()
            logger.info("智能引用匹配系统初始化完成")
        else:
            logger.info("智能引用匹配系统已禁用")
    except Exception as e:
        logger.warning(f"智能引用匹配系统初始化失败，将使用基础模式: {e}")

def initialize_name_linker_system():
    """初始化人物姓名链接系统"""
    try:
        logger.info("正在初始化人物姓名链接系统...")
        if config.ENABLE_NAME_LINKER:
            success = name_linker.load_name_data()
            if success:
                stats = name_linker.get_statistics()
                logger.info(f"人物姓名链接系统初始化完成 - 加载了 {stats['total_persons']} 个人物，其中 {stats['persons_with_links']} 个有主页链接")
            else:
                logger.warning("人物姓名链接系统初始化失败，该功能将不可用")
        else:
            logger.info("人物姓名链接系统已禁用")
    except Exception as e:
        logger.warning(f"人物姓名链接系统初始化失败: {e}")

# 应用启动时初始化
initialize_citation_system()
initialize_name_linker_system()

# 智能检索策略
class SmartRetrievalStrategy:
    """智能检索策略，根据问题类型优化检索参数"""

    def __init__(self):
        # 定义不同问题类型的关键词
        self.question_patterns = {
            'name_query': ['是谁', '谁是', '什么人', '人名', '姓名'],
            'action_query': ['做了什么', '在做', '研究', '工作', '从事', '负责', '参与', '完成', '实现', '发现'],
            'definition_query': ['是什么', '什么是', '定义', '概念', '含义'],
            'method_query': ['怎么做', '如何', '方法', '步骤', '流程', '过程'],
            'comparison_query': ['区别', '不同', '比较', '优缺点', '差异']
        }

    def analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        question_lower = question.lower()

        # 计算每种类型的匹配分数
        type_scores = {}
        for qtype, keywords in self.question_patterns.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                type_scores[qtype] = score

        # 返回得分最高的类型，如果没有匹配则返回默认类型
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return 'general'

    def get_optimized_params(self, question: str, base_params: dict) -> dict:
        """根据问题类型获取优化的检索参数"""
        question_type = self.analyze_question_type(question)

        # 复制基础参数
        params = base_params.copy()

        if question_type == 'name_query':
            # 人名查询：更依赖关键词匹配
            params['similarity_threshold'] = max(params.get('similarity_threshold', 0.1), 0.1)
            params['vector_similarity_weight'] = min(params.get('vector_similarity_weight', 0.7), 0.5)
            params['top_k'] = 5

        elif question_type == 'action_query':
            # 行为/工作查询：更依赖语义理解
            params['similarity_threshold'] = config.ADAPTIVE_SIMILARITY_THRESHOLD
            params['vector_similarity_weight'] = 0.8
            params['top_k'] = 5

        elif question_type == 'definition_query':
            # 定义查询：平衡关键词和语义
            params['similarity_threshold'] = 0.08
            params['vector_similarity_weight'] = 0.6
            params['top_k'] = 5

        elif question_type == 'method_query':
            # 方法查询：需要更多上下文
            params['similarity_threshold'] = 0.06
            params['vector_similarity_weight'] = 0.7
            params['top_k'] = 7

        logger.info(f"问题类型: {question_type}, 优化参数: 阈值={params['similarity_threshold']}, "
                   f"向量权重={params['vector_similarity_weight']}, TopK={params['top_k']}")

        return params

# 初始化智能检索策略
smart_retrieval = SmartRetrievalStrategy()

# 文档内容解析函数 - 核心功能实现
def parse_content_and_link(content: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    从RAGFlow返回的content字符串中解析出正文文本、微信链接和原文链接。

    设计原则：
    - 文本内容是主要部分。
    - "微信链接:" 和 "原文链接:" 是明确的键值对分隔符。
    - 链接总是位于文本内容的末尾。

    Args:
        content (str): RAGFlow返回的文档内容，期望格式为
                       "[正文内容]... 微信链接: [URL1] 原文链接: [URL2]"

    Returns:
        tuple: (正文文本, 微信链接URL或None, 原文链接URL或None)
    """
    import re

    if not content or not isinstance(content, str):
        return content or "", None, None

    text_content = content
    wechat_link = None
    source_link = None

    # 使用正则表达式匹配 "微信链接:" 和 "原文链接:"
    # 使用 re.IGNORECASE 忽略大小写，提高鲁棒性
    wechat_match = re.search(r'微信链接\s*[:：]\s*(https?://[^\s]+)', content, re.IGNORECASE)
    source_match = re.search(r'原文链接\s*[:：]\s*(https?://[^\s]+)', content, re.IGNORECASE)

    # 提取链接并从文本中移除它们，以获得纯净的正文
    if wechat_match:
        wechat_link = wechat_match.group(1).strip()
        # 更新正文内容为匹配链接之前的所有部分
        text_content = content[:wechat_match.start()].strip()

    if source_match:
        source_link = source_match.group(1).strip()
        # 如果微信链接也存在，确保正文只截取到两个链接中靠前的一个
        if wechat_match:
            text_content = content[:min(wechat_match.start(), source_match.start())].strip()
        else:
            text_content = content[:source_match.start()].strip()

    logger.debug(f"解析文档 - 文本长度: {len(text_content)}, 微信链接: {wechat_link}, 原文链接: {source_link}")

    # 如果一个链接都未匹配到，则退回旧的解析逻辑作为兼容
    if not wechat_link and not source_link:
        legacy_text, legacy_link = legacy_parse_single_link(content)
        return legacy_text, legacy_link, None # 将旧链接放在微信链接的位置

    return text_content, wechat_link, source_link

def legacy_parse_single_link(content: str) -> tuple[str, Optional[str]]:
    """保留旧的解析逻辑用于兼容"""
    import re
    pattern = r'(.+?)\s*链接\s*[：:]\s*(https?://[^\s]+)(?:\s|$)'
    match = re.search(pattern, content.strip())
    if match:
        text_content = match.group(1).strip()
        link_url = match.group(2).strip()
        return text_content, link_url
    return content.strip(), None

# 增强引用溯源工具函数
def extract_citations_from_documents(documents: List[Dict]) -> List[Dict]:
    """从文档列表中提取并去重来源信息 - 增强版"""
    citations = []
    seen_citations = set()  # 用于去重

    # 首先使用增强解析器处理文档
    try:
        if config.ENABLE_ENHANCED_PARSER:
            enhanced_documents = enhanced_parser.batch_parse_documents(documents, fast_mode=True)
            logger.info(f"使用增强解析器处理了 {len(enhanced_documents)} 个文档（快速模式）")
        else:
            enhanced_documents = documents
            logger.info(f"已跳过增强内容解析，直接使用原始文档")
    except Exception as e:
        logger.warning(f"增强解析器处理失败，回退到原始文档: {e}")
        enhanced_documents = documents

    for doc in enhanced_documents:
        metadata = doc.get('metadata', {})

        # 提取来源信息 - 优先使用增强解析的结果
        source = metadata.get('source', '').strip()

        # 多层级链接提取策略
        link = None

        # 1. 优先使用智能匹配的链接
        if metadata.get('matched_link'):
            link = metadata.get('matched_link')
            if not source and metadata.get('matched_source'):
                source = metadata.get('matched_source')


        # 2. 使用解析出的原文链接
        elif metadata.get('source_link'):
            link = metadata.get('source_link')

        # 3. 使用微信链接作为备选
        elif metadata.get('wechat_link'):
            link = metadata.get('wechat_link')

        # 4. 回退到原始metadata中的链接
        elif metadata.get('link'):
            link = metadata.get('link')

        # 其他字段
        document_path = metadata.get('document_path', '').strip()
        page_number = metadata.get('page_number', '').strip()
        section = metadata.get('section', '').strip()

        # 如果没有source，尝试从其他字段获取
        if not source:
            source = doc.get('document_name', '').strip()

        # 如果仍然没有source，尝试从内容中提取
        if not source and doc.get('content'):
            content = doc.get('content', '')[:100]  # 取前100字符作为简要描述
            source = f"文档摘要: {content}..." if len(content) >= 100 else content

        # 只有当source不为空时才添加引用
        if source:
            # 创建唯一标识符用于去重（基于source和link）
            citation_key = f"{source}|{link or ''}"

            if citation_key not in seen_citations:
                seen_citations.add(citation_key)

                citation = {"source": source}

                # 只添加有值的字段
                if link:
                    citation["link"] = link
                if document_path:
                    citation["document_path"] = document_path
                if page_number:
                    citation["page_number"] = page_number
                if section:
                    citation["section"] = section

                # 添加调试信息（仅在DEBUG模式下）
                if config.DEBUG and metadata.get('parsing_metadata'):
                    citation["_debug_parsing"] = metadata.get('parsing_metadata')

                citations.append(citation)

    logger.info(f"从 {len(documents)} 个文档中提取到 {len(citations)} 个唯一引用来源（增强模式）")
    return citations

def build_prompt_with_citations(question: str, documents: List[Dict], prompt_template: str) -> str:
    """构建包含引用信息的提示词（内部使用，不在最终回答中显示）"""
    context = prompt_template + "\n\n"
    context += "相关资料：\n"

    for idx, doc in enumerate(documents):
        content = doc.get('content', '').strip()
        if content:
            # 限制每个文档的长度
            if len(content) > config.MAX_CHUNK_LENGTH:
                content = content[:config.MAX_CHUNK_LENGTH] + "..."

            # 为内部处理添加来源信息标记（但不在最终回答中显示）
            metadata = doc.get('metadata', {})
            source_info = metadata.get('source', doc.get('document_name', ''))

            context += f"{idx+1}. {content}\n"
            # 添加隐藏的来源标记供模型内部参考
            if source_info:
                context += f"   [内部引用: {source_info}]\n"
            context += "\n"

    context += f"问题：{question}\n\n请基于上述资料回答问题，如果资料中没有明确答案，请基于你的专业知识补充回答。不需要在回答中提及相似度、评分、引用标记或技术参数等信息。"
    return context

# 提示词管理系统
class PromptManager:
    """提示词模板管理类"""

    def __init__(self):
        self.prompt_file = BASE_DIR / "prompt_templates.json"
        self.default_templates = {
            "default": {
                "name": "default",
                "template": "你是一个专业的智能助手。请基于提供的资料回答用户问题，回答要准确、清晰、有条理。不要在回答中提及相似度、评分、检索结果数量等技术细节。",
                "description": "默认提示词模板",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "detailed": {
                "name": "detailed",
                "template": "你是一个专业的智能助手。请基于提供的资料详细回答用户问题，如果资料中信息不完整，请结合你的专业知识进行补充。回答要全面、深入，结构清晰。请避免在回答中提及技术参数或检索细节。",
                "description": "详细回答模板",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "concise": {
                "name": "concise",
                "template": "你是一个专业的智能助手。请基于提供的资料简洁明了地回答用户问题，抓住要点，避冗余。不要提及技术细节或检索信息。",
                "description": "简洁回答模板",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "analytical": {
                "name": "analytical",
                "template": "你是一个专业的智能助手。请基于提供的资料进行深入分析并回答用户问题，包括原因分析、逻辑推理和实用建议。回答要有分析性，逻辑清晰。请专注于内容分析，不要提及技术参数。",
                "description": "分析型回答模板",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "citation_focused": {
                "name": "citation_focused",
                "template": """你是一个严谨、专业的AI研究助手。你的任务是基于我提供的、带有引用链接的"相关资料"来回答问题。

**任务指令**:
1.  请仔细阅读我提供的每一份"相关资料"。每份资料都包含 [正文] 以及可选的 [微信链接] 和 [原文链接]。
2.  基于这些资料的 [正文] 内容，准确、清晰、有条理地回答用户提出的"问题"。
3.  在你的回答内容**完全结束**后，你 **必须** 另起一个新段落，创建一个标题为"**引用来源**"的部分。
4.  在"引用来源"标题下，你 **必须只列出** 你在组织答案时**实际使用并参考过**的那些资料所对应的链接。
5.  **输出格式要求**：
    *   为每个来源创建一个无序列表项 (使用 `-`)。
    *   如果[微信链接]存在，格式化为 `- [微信文章](微信链接_URL)`。
    *   如果[原文链接]存在，格式化为 `- [原文地址](原文链接_URL)`。
    *   如果一篇资料同时提供了两个链接，请都列出来。

**严格约束**:
- 在你的主要回答正文中，**绝对不能**出现任何 URL 链接或提及"根据资料X"等字样。
- "引用来源"部分**只能包含**来自"相关资料"中提供的链接，严禁创造或推断任何不存在的链接。
- 如果提供的所有资料都与问题无关，导致你无法基于它们回答，那么你的回答中**不应包含**"引用来源"部分。

---

现在，请根据以下信息回答问题：""",
                "description": "专门用于引导LLM进行引用链接生成的提示词模板，支持微信和原文双链接精准溯源",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "enhanced_citation": {
                "name": "enhanced_citation",
                "template": """你是一个严谨、专业的AI研究助手。你需要基于提供的学术资料来回答问题，并确保引用的准确性。

**核心任务**:
1. **内容理解**: 仔细阅读每份"相关资料"，理解其核心观点和结论
2. **准确回答**: 基于资料内容提供专业、准确的回答
3. **精确引用**: 只引用你实际参考过的资料对应的链接

**回答结构**:
1. 首先提供完整、专业的回答内容
2. 回答结束后，另起段落添加"**参考文献**"部分
3. 在参考文献中列出实际使用的资料链接

**引用原则**:
- 只引用实际参考的资料
- 链接与内容必须完全匹配
- 优先使用原文链接，其次是微信链接
- 如果资料来源明确但缺少链接，说明"来源：[资料描述]"

**格式要求**:
- 引用格式：`- [资料标题或描述](链接URL)`
- 如无链接：`- 来源：[资料描述]`

现在请基于以下资料回答问题：""",
                "description": "增强版引用模板，专注于确保链接与内容的精确匹配，集成智能链接匹配功能",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
        self.load_templates()

    def load_templates(self):
        """加载提示词模板"""
        try:
            if self.prompt_file.exists():
                with open(self.prompt_file, 'r', encoding='utf-8') as f:
                    loaded_templates = json.load(f)
                    # 合并默认模板和加载的模板
                    self.templates = {**self.default_templates, **loaded_templates}
            else:
                self.templates = self.default_templates.copy()
                self.save_templates()
        except Exception as e:
            logger.warning(f"加载提示词模板失败，使用默认模板: {e}")
            self.templates = self.default_templates.copy()

    def save_templates(self):
        """保存提示词模板到文件"""
        try:
            with open(self.prompt_file, 'w', encoding='utf-8') as f:
                # 只保存非默认模板
                custom_templates = {k: v for k, v in self.templates.items()
                                  if k not in self.default_templates}
                json.dump(custom_templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存提示词模板失败: {e}")

    def get_template(self, name: str) -> str:
        """获取提示词模板"""
        if name in self.templates:
            return self.templates[name]["template"]
        else:
            # 如果是自定义提示词（不在预设模板中），直接返回
            return name

    def list_templates(self) -> List[Dict]:
        """获取所有提示词模板列表"""
        return list(self.templates.values())

    def add_template(self, name: str, template: str, description: str = "") -> bool:
        """添加新的提示词模板"""
        try:
            self.templates[name] = {
                "name": name,
                "template": template,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self.save_templates()
            return True
        except Exception as e:
            logger.error(f"添加提示词模板失败: {e}")
            return False

    def update_template(self, name: str, template: str = None, description: str = None) -> bool:
        """更新提示词模板"""
        try:
            if name in self.templates:
                if template is not None:
                    self.templates[name]["template"] = template
                if description is not None:
                    self.templates[name]["description"] = description
                self.templates[name]["updated_at"] = datetime.now().isoformat()
                self.save_templates()
                return True
            return False
        except Exception as e:
            logger.error(f"更新提示词模板失败: {e}")
            return False

    def delete_template(self, name: str) -> bool:
        """删除提示词模板（不能删除默认模板）"""
        try:
            if name in self.default_templates:
                return False  # 不能删除默认模板
            if name in self.templates:
                del self.templates[name]
                self.save_templates()
                return True
            return False
        except Exception as e:
            logger.error(f"删除提示词模板失败: {e}")
            return False

# 初始化提示词管理器
prompt_manager = PromptManager()

app = FastAPI(
    title="RAGFlow + Qwen 智能问答系统 v2.0",
    description="增强版基于RAGFlow检索和Qwen大模型的智能问答API，支持文档切片可视化和深度调试",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 挂载静态文件目录
templates_dir = BASE_DIR / "templates"
if templates_dir.exists():
    app.mount("/static", StaticFiles(directory=str(templates_dir)), name="static")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理未捕获的异常"""
    logger.error(f"未处理的异常 [{request.method} {request.url}]: {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "服务遇到了意外错误，请稍后重试",
            "timestamp": datetime.now().isoformat(),
            "request_id": f"{int(time.time() * 1000)}",
            "debug_info": str(exc) if config.DEBUG else None
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.warning(f"请求验证失败 [{request.method} {request.url}]: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求参数验证失败",
            "detail": exc.errors(),
            "message": "请检查请求格式和参数类型",
            "timestamp": datetime.now().isoformat()
        }
    )

# 数据模型定义
class QARequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    dataset_ids: List[str] = Field(default_factory=lambda: config.DEFAULT_DATASET_IDS, description="数据集ID列表")
    document_ids: List[str] = Field(default_factory=lambda: config.DEFAULT_DOCUMENT_IDS, description="文档ID列表")
    prompt_template: str = Field(default="enhanced_citation", description="提示词模板名称或自定义提示词")
    similarity_threshold: float = Field(default=None, ge=0.0, le=1.0, description="相似度阈值")
    vector_similarity_weight: float = Field(default=None, ge=0.0, le=1.0, description="向量相似度权重")
    top_k: int = Field(default=None, ge=1, le=100, description="返回文档数量")
    page_size: int = Field(default=None, ge=1, le=100, description="分页大小")
    keyword: bool = Field(default=True, description="是否启用关键词搜索")
    highlight: bool = Field(default=True, description="是否启用高亮显示")
    rerank_id: str = Field(default="", description="重排序模型ID")

    # Qwen参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Qwen温度参数")
    max_tokens: int = Field(default=2000, ge=1, le=4000, description="Qwen最大token数")

class QAStreamRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    dataset_ids: List[str] = Field(default_factory=lambda: config.DEFAULT_DATASET_IDS, description="数据集ID列表")
    document_ids: List[str] = Field(default_factory=lambda: config.DEFAULT_DOCUMENT_IDS, description="文档ID列表")
    prompt_template: str = Field(default="default", description="提示词模板名称或自定义提示词")
    similarity_threshold: float = Field(default=None, ge=0.0, le=1.0, description="相似度阈值")
    vector_similarity_weight: float = Field(default=None, ge=0.0, le=1.0, description="向量相似度权重")
    top_k: int = Field(default=None, ge=1, le=100, description="返回文档数量")
    page_size: int = Field(default=None, ge=1, le=100, description="分页大小")
    keyword: bool = Field(default=True, description="是否启用关键词搜索")
    highlight: bool = Field(default=True, description="是否启用高亮显示")
    rerank_id: str = Field(default="", description="重排序模型ID")

    # Qwen参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Qwen温度参数")
    max_tokens: int = Field(default=2000, ge=1, le=4000, description="Qwen最大token数")

class Citation(BaseModel):
    """引用溯源信息模型"""
    source: str = Field(..., description="文档来源名称")
    link: Optional[str] = Field(default=None, description="文档链接")
    document_path: Optional[str] = Field(default=None, description="文档路径")
    page_number: Optional[str] = Field(default=None, description="页码")
    section: Optional[str] = Field(default=None, description="章节或标题")

class QAResponse(BaseModel):
    answer: str = Field(..., description="AI回答")
    question: str = Field(..., description="用户问题")
    context_used: bool = Field(..., description="是否使用了检索到的上下文")
    ragflow_docs_count: int = Field(..., description="检索到的文档数量")
    ragflow_status: str = Field(default="normal", description="RAGflow服务状态")
    prompt_template_used: str = Field(default="", description="使用的提示词模板")
    performance_stats: Dict = Field(default={}, description="性能统计信息")
    debug_info: Optional[Dict] = Field(default=None, description="调试信息")
    citations: List[Citation] = Field(default=[], description="引用溯源信息")

class PromptTemplate(BaseModel):
    name: str = Field(..., description="提示词模板名称")
    template: str = Field(..., description="提示词模板内容")
    description: str = Field(default="", description="提示词描述")
    created_at: Optional[str] = Field(default=None, description="创建时间")
    updated_at: Optional[str] = Field(default=None, description="更新时间")

class PromptTemplateCreate(BaseModel):
    name: str = Field(..., description="提示词模板名称", min_length=1, max_length=50)
    template: str = Field(..., description="提示词模板内容", min_length=1, max_length=1000)
    description: str = Field(default="", description="提示词描述", max_length=200)

class PromptTemplateUpdate(BaseModel):
    template: Optional[str] = Field(None, description="提示词模板内容", min_length=1, max_length=1000)
    description: Optional[str] = Field(None, description="提示词描述", max_length=200)

class DatasetInfo(BaseModel):
    dataset_id: str
    name: Optional[str] = None
    document_count: Optional[int] = None
    status: str
    last_updated: Optional[str] = None

# API路由定义
@app.get("/")
async def root():
    """根路径，返回Web界面"""
    index_file = BASE_DIR / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "RAGFlow + Qwen 智能问答系统 v2.0",
        "version": "2.0.0",
        "features": [
            "文档切片可视化",
            "深度调试功能",
            "性能监控",
            "多数据集支持"
        ],
        "docs": "/docs",
        "error": f"模板文件不存在: {index_file}",
        "base_dir": str(BASE_DIR)
    }

@app.get("/health")
async def health_check():
    """增强版健康检查接口"""
    start_time = time.time()

    health_status = {
        "status": "healthy",
        "service": "ragflow-qwen-qa-v2",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "uptime": time.time() - start_time,
        "components": {}
    }

    # 检查配置
    config_status = {"status": "ok", "issues": []}
    if not config.RAGFLOW_API_KEY:
        config_status["issues"].append("RAGflow API密钥未配置")
    if not config.QWEN_API_KEY:
        config_status["issues"].append("Qwen API密钥未配置")
    if not config.DEFAULT_DATASET_IDS:
        config_status["issues"].append("默认数据集ID未配置")

    if config_status["issues"]:
        config_status["status"] = "warning"
    health_status["components"]["config"] = config_status

    # 检查RAGFlow连接（异步）
    try:
        ragflow_health = await ragflow_client.health_check()
        health_status["components"]["ragflow"] = ragflow_health
    except Exception as e:
        health_status["components"]["ragflow"] = {
            "status": "error",
            "message": f"RAGFlow健康检查失败: {str(e)}"
        }

    # 检查Qwen连接（异步）
    try:
        qwen_health = await qwen_client.health_check()
        health_status["components"]["qwen"] = qwen_health
    except Exception as e:
        health_status["components"]["qwen"] = {
            "status": "error",
            "message": f"Qwen健康检查失败: {str(e)}"
        }

    # 计算总体健康状态
    component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
    if "error" in component_statuses:
        health_status["status"] = "unhealthy"
    elif "warning" in component_statuses or "unhealthy" in component_statuses:
        health_status["status"] = "degraded"

    health_status["response_time"] = time.time() - start_time

    return health_status

@app.get("/datasets")
async def list_datasets():
    """获取数据集列表和状态"""
    datasets = []

    for dataset_id in config.DEFAULT_DATASET_IDS:
        try:
            # 尝试获取数据集信息
            dataset_info = await ragflow_client.get_dataset_info(dataset_id)

            datasets.append({
                "dataset_id": dataset_id,
                "name": dataset_info.get("name", f"数据集-{dataset_id[:8]}"),
                "status": "active" if "error" not in dataset_info else "error",
                "error": dataset_info.get("error"),
                "document_count": dataset_info.get("document_count"),
                "last_updated": dataset_info.get("last_updated")
            })
        except Exception as e:
            datasets.append({
                "dataset_id": dataset_id,
                "name": f"数据集-{dataset_id[:8]}",
                "status": "unknown",
                "error": str(e)
            })

    return {
        "datasets": datasets,
        "total": len(datasets),
        "active": len([d for d in datasets if d["status"] == "active"])
    }

@app.get("/datasets/{dataset_id}/test")
async def test_dataset(dataset_id: str = FastAPIPath(..., description="数据集ID")):
    """测试特定数据集的检索功能"""
    try:
        result = await ragflow_client.retrieve(
            question="测试检索",
            dataset_ids=[dataset_id],
            top_k=3,
            similarity_threshold=0.1
        )

        return {
            "dataset_id": dataset_id,
            "status": "success",
            "documents_found": len(result.get("documents", [])),
            "response_time": result.get("response_time", 0),
            "sample_documents": result.get("processed_documents", [])[:2]  # 返回前2个文档作为示例
        }
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "status": "error",
            "error": str(e)
        }

# 提示词管理API
@app.get("/prompts", response_model=List[PromptTemplate])
async def list_prompts():
    """获取所有提示词模板列表"""
    return prompt_manager.list_templates()

@app.get("/prompts/{prompt_name}", response_model=PromptTemplate)
async def get_prompt(prompt_name: str = FastAPIPath(..., description="提示词模板名称")):
    """获取指定的提示词模板"""
    if prompt_name not in prompt_manager.templates:
        raise HTTPException(status_code=404, detail=f"提示词模板 '{prompt_name}' 不存在")
    return prompt_manager.templates[prompt_name]

@app.post("/prompts", response_model=Dict[str, str])
async def create_prompt(prompt_data: PromptTemplateCreate):
    """创建新的提示词模板"""
    if prompt_data.name in prompt_manager.templates:
        raise HTTPException(status_code=400, detail=f"提示词模板 '{prompt_data.name}' 已存在")

    success = prompt_manager.add_template(
        name=prompt_data.name,
        template=prompt_data.template,
        description=prompt_data.description
    )

    if success:
        return {"message": f"提示词模板 '{prompt_data.name}' 创建成功"}
    else:
        raise HTTPException(status_code=500, detail="创建提示词模板失败")

@app.put("/prompts/{prompt_name}", response_model=Dict[str, str])
async def update_prompt(
    prompt_name: str = FastAPIPath(..., description="提示词模板名称"),
    prompt_data: PromptTemplateUpdate = None
):
    """更新提示词模板"""
    if prompt_name not in prompt_manager.templates:
        raise HTTPException(status_code=404, detail=f"提示词模板 '{prompt_name}' 不存在")

    if prompt_name in prompt_manager.default_templates:
        raise HTTPException(status_code=400, detail=f"不能修改默认提示词模板 '{prompt_name}'")

    success = prompt_manager.update_template(
        name=prompt_name,
        template=prompt_data.template,
        description=prompt_data.description
    )

    if success:
        return {"message": f"提示词模板 '{prompt_name}' 更新成功"}
    else:
        raise HTTPException(status_code=500, detail="更新提示词模板失败")

@app.delete("/prompts/{prompt_name}", response_model=Dict[str, str])
async def delete_prompt(prompt_name: str = FastAPIPath(..., description="提示词模板名称")):
    """删除提示词模板"""
    if prompt_name not in prompt_manager.templates:
        raise HTTPException(status_code=404, detail=f"提示词模板 '{prompt_name}' 不存在")

    success = prompt_manager.delete_template(prompt_name)

    if success:
        return {"message": f"提示词模板 '{prompt_name}' 删除成功"}
    else:
        raise HTTPException(status_code=400, detail=f"不能删除默认提示词模板 '{prompt_name}' 或删除失败")

@app.post("/debug/ragflow")
async def debug_ragflow(request: QARequest):
    """增强版RAGFlow调试接口"""
    logger.info(f"RAGFlow调试请求 - 问题: {request.question[:50]}...")

    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "request_params": request.dict(),
        "config_info": {
            "ragflow_url": config.RAGFLOW_URL,
            "has_api_key": bool(config.RAGFLOW_API_KEY),
            "timeout": config.RAGFLOW_TIMEOUT,
            "max_retries": config.RAGFLOW_MAX_RETRIES
        }
    }

    try:
        result = await ragflow_client.retrieve(
            question=request.question,
            dataset_ids=request.dataset_ids,
            document_ids=request.document_ids,
            similarity_threshold=request.similarity_threshold,
            vector_similarity_weight=request.vector_similarity_weight,
            top_k=request.top_k,
            page_size=request.page_size,
            keyword=request.keyword,
            highlight=request.highlight,
            rerank_id=request.rerank_id
        )

        debug_info.update({
            "status": "success",
            "ragflow_response": result,
            "analysis": {
                "documents_retrieved": len(result.get("documents", [])),
                "avg_score": result.get("retrieval_stats", {}).get("avg_score", 0),
                "score_distribution": result.get("retrieval_stats", {}).get("score_range", {}),
                "performance": {
                    "response_time": result.get("response_time", 0),
                    "request_size": len(json.dumps(request.dict())),
                    "response_size": len(json.dumps(result))
                }
            }
        })

    except Exception as e:
        debug_info.update({
            "status": "error",
            "error_message": str(e),
            "error_type": type(e).__name__
        })

    return debug_info

class QwenDebugRequest(BaseModel):
    prompt: str = Field(..., description="测试提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=100, ge=1, le=1000, description="最大token数")

@app.post("/debug/qwen")
async def debug_qwen(request: QwenDebugRequest):
    """Qwen调试接口"""
    logger.info(f"Qwen调试请求 - 提示词长度: {len(request.prompt)}")

    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "request_params": {
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        },
        "config_info": {
            "qwen_url": config.QWEN_URL,
            "qwen_model": config.QWEN_MODEL,
            "has_api_key": bool(config.QWEN_API_KEY),
            "timeout": config.QWEN_TIMEOUT
        }
    }

    try:
        result = await qwen_client.chat(
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        debug_info.update({
            "status": "success",
            "qwen_response": result,
            "analysis": {
                "response_length": len(result.get("content", "")),
                "tokens_used": result.get("usage", {}),
                "performance": {
                    "response_time": result.get("response_time", 0),
                    "model": result.get("model", ""),
                    "finish_reason": result.get("finish_reason", "")
                }
            }
        })

    except Exception as e:
        debug_info.update({
            "status": "error",
            "error_message": str(e),
            "error_type": type(e).__name__
        })

    return debug_info

@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    """增强版智能问答接口"""
    start_time = time.time()
    logger.info(f"问答请求 - 问题: {request.question[:50]}...")

    performance_stats = {
        "start_time": start_time,
        "ragflow_time": 0,
        "qwen_time": 0,
        "total_time": 0
    }

    debug_info = {} if config.DEBUG else None

    # 1. 智能优化检索参数
    base_params = {
        'similarity_threshold': request.similarity_threshold,
        'vector_similarity_weight': request.vector_similarity_weight,
        'top_k': request.top_k
    }
    optimized_params = smart_retrieval.get_optimized_params(request.question, base_params)

    # 应用优化后的参数
    final_similarity_threshold = optimized_params['similarity_threshold']
    final_vector_weight = optimized_params['vector_similarity_weight']
    final_top_k = optimized_params['top_k']

    # 2. 调用RAGflow获取知识碎片
    ragflow_result = {"documents": []}
    ragflow_error = None
    ragflow_start = time.time()

    try:
        logger.info(f"调用RAGflow检索接口（智能优化）- 阈值:{final_similarity_threshold}, TopK:{final_top_k}")
        ragflow_result = await ragflow_client.retrieve(
            question=request.question,
            dataset_ids=request.dataset_ids,
            document_ids=request.document_ids,
            similarity_threshold=final_similarity_threshold if final_similarity_threshold is not None else 0.05,
            vector_similarity_weight=final_vector_weight if final_vector_weight is not None else 0.6 ,
            top_k=final_top_k if final_top_k is not None else 5,
            page_size=request.page_size,
            keyword=request.keyword,
            highlight=request.highlight,
            rerank_id=request.rerank_id
        )
        performance_stats["ragflow_time"] = time.time() - ragflow_start
        logger.info(f"RAGflow返回 {len(ragflow_result.get('documents', []))} 个文档")

        if debug_info is not None:
            debug_info["ragflow_result"] = ragflow_result

    except Exception as e:
        ragflow_error = str(e)
        performance_stats["ragflow_time"] = time.time() - ragflow_start
        logger.warning(f"RAGflow请求失败，将使用Qwen直接回答: {e}")

        if debug_info is not None:
            debug_info["ragflow_error"] = ragflow_error

    # 2. 构建上下文并优化回答策略
    documents = ragflow_result.get("documents", [])
    context_used = False

    # 过滤低质量文档（使用优化后的相似度阈值）
    quality_documents = []
    if documents:
    # 确定用于比较的阈值，如果为 None，则使用配置文件中的默认值
        comparison_threshold = final_similarity_threshold if final_similarity_threshold is not None else 0.05

        for doc in documents:
            score = doc.get('score', 0)
            if score >= comparison_threshold:
                quality_documents.append(doc)

    # 获取提示词模板
    prompt_template = prompt_manager.get_template(request.prompt_template)

    # 检查是否使用citation_focused或enhanced_citation模板，如果是则采用增强解析逻辑
    if request.prompt_template in ["citation_focused", "enhanced_citation"] and quality_documents:
        # 使用增强解析器处理文档，获得更准确的链接匹配
        try:
            enhanced_documents = enhanced_parser.batch_parse_documents(quality_documents, fast_mode=True)
            parsed_documents = []

            for doc in enhanced_documents:
                content = doc.get('content', '').strip()
                metadata = doc.get('metadata', {})

                if content:
                    # 限制文本内容长度
                    if len(content) > config.MAX_CHUNK_LENGTH:
                        content = content[:config.MAX_CHUNK_LENGTH] + "..."

                    # 从metadata中获取链接信息，优先使用智能匹配的结果
                    wechat_link = metadata.get('wechat_link')
                    source_link = metadata.get('source_link') or metadata.get('matched_link')

                    parsed_documents.append({
                        'text': content,
                        'wechat_link': wechat_link,
                        'source_link': source_link,
                        'parsing_metadata': metadata.get('parsing_metadata', {})
                    })

            logger.info(f"增强解析处理了 {len(enhanced_documents)} 个文档，获得 {len(parsed_documents)} 个有效解析结果")

        except Exception as e:
            logger.warning(f"增强解析失败，回退到标准解析: {e}")
            # 回退到原有解析逻辑
            parsed_documents = []
            for doc in quality_documents:
                content = doc.get('content', '').strip()
                if content:
                    text_content, wechat_link, source_link = parse_content_and_link(content)
                    if text_content:
                        if len(text_content) > config.MAX_CHUNK_LENGTH:
                            text_content = text_content[:config.MAX_CHUNK_LENGTH] + "..."
                        parsed_documents.append({
                            'text': text_content,
                            'wechat_link': wechat_link,
                            'source_link': source_link
                        })

        if parsed_documents:
            # 使用增强的引用格式构建上下文
            context = prompt_template + "\n\n"
            context += "相关资料：\n"
            for idx, parsed_doc in enumerate(parsed_documents):
                context += f"{idx+1}. [正文]: {parsed_doc['text']}\n"
                if parsed_doc['wechat_link']:
                    context += f"   [微信链接]: {parsed_doc['wechat_link']}\n"
                if parsed_doc['source_link']:
                    context += f"   [原文链接]: {parsed_doc['source_link']}\n"
                context += "\n"

            context += f"问题：{request.question}\n"
            print(context)
            context_used = True
            logger.info(f"使用{request.prompt_template}模板，解析出 {len(parsed_documents)} 个带链接的文档")
        else:
            # 容错处理：没有任何文档能解析出有效的正文内容，回退到原有逻辑
            logger.warning(f"citation_focused模板下未能解析出任何有效的正文内容，回退到原有上下文构建逻辑")
            context = prompt_template + "\n\n"
            context += "相关资料：\n"
            for idx, doc in enumerate(quality_documents):
                content = doc.get('content', '').strip()
                if content:
                    if len(content) > config.MAX_CHUNK_LENGTH:
                        content = content[:config.MAX_CHUNK_LENGTH] + "..."
                    context += f"{idx+1}. {content}\n\n"
            context += f"问题：{request.question}\n\n请基于上述资料回答问题，如果资料中没有明确答案，请基于你的专业知识补充回答。不需要提及相似度、评分或技术参数等信息。"
            context_used = True
    elif quality_documents:
        # 原有的上下文构建逻辑（非citation_focused模板）
        context = prompt_template + "\n\n"
        # 优化后的上下文构建 - 移除相似度信息，使用更自然的格式
        context += "相关资料：\n"
        for idx, doc in enumerate(quality_documents):
            content = doc.get('content', '').strip()
            if content:
                # 限制每个文档的长度
                if len(content) > config.MAX_CHUNK_LENGTH:
                    content = content[:config.MAX_CHUNK_LENGTH] + "..."
                # 移除相似度信息，使用更自然的格式
                context += f"{idx+1}. {content}\n\n"
        context += f"问题：{request.question}\n\n请基于上述资料回答问题，如果资料中没有明确答案，请基于你的专业知识补充回答。不需要提及相似度、评分或技术参数等信息。"
        context_used = True
        logger.info(f"使用了 {len(quality_documents)} 个高质量文档作为上下文，提示词模板: {request.prompt_template}")
    else:
        # 没有检索到相关文档或文档质量过低时的处理策略
        if ragflow_error:
            context = f"""请基于你的知识回答以下问题：{request.question}

注意：由于文档检索服务暂时不可用，我将基于我的训练知识为您提供答案。"""
            logger.info("RAGflow不可用，使用Qwen直接回答模式")
        elif documents:
            # 有文档但质量不高
            context = f"""请基于你的知识详细回答以下问题：{request.question}

注意：我在知识库中检索到了 {len(documents)} 个文档，但相似度较低（可能不够相关），因此我将主要基于我的训练知识为您提供答案。如果您的问题比较专业或特定，建议您：
1. 尝试换个方式描述问题
2. 提供更多上下文信息
3. 确认知识库中是否包含相关资料"""
            logger.info(f"检索到 {len(documents)} 个文档，但质量不高，使用Qwen知识回答")
        else:
            # 完全没有文档
            context = f"""请基于你的知识详细回答以下问题：{request.question}

注意：我在当前知识库中没有找到与您问题直接相关的资料，因此我将基于我的训练知识为您提供答案。如果您需要更专业或最新的信息，建议您：
1. 检查问题的关键词是否准确
2. 确认知识库中是否包含相关文档
3. 考虑咨询专业人士或查阅最新资料"""
            logger.warning(f"数据集 {request.dataset_ids} 中未检索到任何文档，使用Qwen知识回答")

    # 3. 调用Qwen大模型
    qwen_start = time.time()
    try:
        logger.info("调用Qwen大模型...")
        # 在调试模式下输出完整的上下文内容
        if config.DEBUG:
            logger.debug(f"完整上下文内容:\n{context}")
            logger.debug(f"上下文长度: {len(context)} 字符")
        
        qwen_result = await qwen_client.chat(
            messages=[{"role": "user", "content": context}],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        answer = qwen_result["content"]
        performance_stats["qwen_time"] = time.time() - qwen_start
        logger.info("Qwen大模型调用成功")

        # ==================================================
        # 人物主页链接后处理逻辑 - 新增功能
        # ==================================================

        try:
            # 在问题和答案中搜索提及的人物
            mentioned_in_question = name_linker.find_mentioned_names(request.question)
            mentioned_in_answer = name_linker.find_mentioned_names(answer)

            # 合并结果并去重，优先保留有链接的人物
            all_mentioned = {}
            all_mentioned.update(mentioned_in_question)
            all_mentioned.update(mentioned_in_answer)

            # 只保留有有效链接的人物
            valid_mentioned = {name: url for name, url in all_mentioned.items() if url}

            if valid_mentioned:
                # 生成人物链接的Markdown格式
                person_links_md = name_linker.format_person_links_markdown(valid_mentioned)

                # 将人物链接附加到答案末尾
                if person_links_md:
                    answer += person_links_md
                    logger.info(f"为答案附加了 {len(valid_mentioned)} 个人物主页链接: {list(valid_mentioned.keys())}")

        except Exception as e:
            logger.error(f"处理人物主页链接时出错: {e}")
        # ==================================================
        # 人物链接处理逻辑结束
        # ==================================================

        if debug_info is not None:
            debug_info["qwen_result"] = qwen_result

    except Exception as e:
        performance_stats["qwen_time"] = time.time() - qwen_start
        logger.error(f"Qwen请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"Qwen请求失败: {str(e)}")

    # 4. 处理结果
    performance_stats["total_time"] = time.time() - start_time

    # 确定RAGflow状态和文档处理结果
    if ragflow_error:
        ragflow_status = "failed"
    elif not documents:
        ragflow_status = "no_docs"
    elif not quality_documents:
        ragflow_status = "low_quality"
    else:
        ragflow_status = "normal"

    # 添加性能统计
    if debug_info is not None:
        debug_info["performance_stats"] = performance_stats
        debug_info["context_info"] = {
            "context_length": len(context),
            "context_used": context_used,
            "documents_count": len(documents)
        }

            # 添加详细的统计信息（多知识库增强+智能检索信息）
        if debug_info is not None:
            debug_info["document_stats"] = {
                "total_retrieved": len(documents),
                "quality_filtered": len(quality_documents),
                "similarity_threshold": final_similarity_threshold,
                "context_strategy": "knowledge_base" if context_used else "ai_knowledge",
                "multi_kb_analysis": ragflow_result.get("retrieval_stats", {}).get("dataset_coverage", {}),
                "datasets_queried": len(request.dataset_ids),
                "datasets_with_results": len(ragflow_result.get("retrieval_stats", {}).get("dataset_coverage", {}).get("datasets_with_results", []))
            }
            # 添加智能检索信息
            debug_info["smart_retrieval"] = {
                "question_type": smart_retrieval.analyze_question_type(request.question),
                "original_params": base_params,
                "optimized_params": optimized_params,
                "optimization_applied": True
            }

    # 提取引用溯源信息
    citations = []
    citation_validation_result = None

    if context_used and quality_documents:
        try:
            citations_data = extract_citations_from_documents(quality_documents)
            citations = [Citation(**citation) for citation in citations_data]
            logger.info(f"非流式问答：提取到 {len(citations)} 个引用来源")

            # 验证引用准确性（仅在DEBUG模式下启用）
            if config.DEBUG and citations_data:
                try:
                    citation_validation_result = validate_response_citations(answer, citations_data)
                    logger.info(f"引用验证完成：{citation_validation_result['valid_citations']}/{citation_validation_result['total_citations']} 有效")

                    # 如果验证发现问题，记录警告
                    if citation_validation_result['summary_issues']:
                        logger.warning(f"引用验证发现问题: {citation_validation_result['summary_issues']}")

                except Exception as validation_error:
                    logger.warning(f"引用验证失败: {validation_error}")

        except Exception as citations_error:
            logger.error(f"提取引用信息失败: {citations_error}")
            # 即使引用提取失败，也要继续返回正常响应

    return QAResponse(
        answer=answer,
        question=request.question,
        context_used=context_used,
        ragflow_docs_count=len(quality_documents),  # 返回实际使用的高质量文档数
        ragflow_status=ragflow_status,
        prompt_template_used=request.prompt_template,
        performance_stats=performance_stats,
        debug_info=debug_info,
        citations=citations
    )

@app.post("/analyze_question")
async def analyze_question(request: dict):
    """分析问题类型并返回优化参数"""
    try:
        question = request.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 使用智能检索策略分析问题
        question_type = smart_retrieval.analyze_question_type(question)

        # 获取基础参数（使用当前配置的默认值）
        base_params = {
            'similarity_threshold': config.DEFAULT_SIMILARITY_THRESHOLD,
            'vector_similarity_weight': config.DEFAULT_VECTOR_WEIGHT,
            'top_k': config.DEFAULT_TOP_K
        }

        # 获取优化后的参数
        optimized_params = smart_retrieval.get_optimized_params(question, base_params)

        return {
            "question_type": question_type,
            "original_params": base_params,
            "optimized_params": optimized_params,
            "optimization_description": get_optimization_description(question_type),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"问题分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

def get_optimization_description(question_type: str) -> str:
    """获取优化策略描述"""
    descriptions = {
        'name_query': '人名查询：增强关键词匹配，降低向量权重',
        'action_query': '行为查询：降低相似度阈值，提升语义理解，增加候选数量',
        'definition_query': '定义查询：平衡关键词与语义权重',
        'method_query': '方法查询：增加检索数量以获取更多上下文',
        'comparison_query': '比较查询：综合优化各项参数',
        'general': '通用查询：使用标准参数配置'
    }
    return descriptions.get(question_type, '未知查询类型')

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "service": "ragflow-qwen-qa-v2",
        "version": "2.1.0",  # 更新版本号
        "config": {
            "default_dataset_ids": config.DEFAULT_DATASET_IDS,
            "qwen_model": config.QWEN_MODEL,
            "debug_mode": config.DEBUG,
            "ragflow_timeout": config.RAGFLOW_TIMEOUT,
            "qwen_timeout": config.QWEN_TIMEOUT,
            "smart_retrieval_enabled": True  # 标识智能检索功能已启用
        },
        "timestamp": datetime.now().isoformat()
    }

# ======================== 配置管理API ========================
@app.get("/config")
async def get_config():
    """获取当前配置信息"""
    return {
        "ragflow": {
            "url": config.RAGFLOW_URL,
            "api_key_configured": bool(config.RAGFLOW_API_KEY),
            "timeout": config.RAGFLOW_TIMEOUT,
            "max_retries": config.RAGFLOW_MAX_RETRIES
        },
        "qwen": {
            "url": config.QWEN_URL,
            "model": config.QWEN_MODEL,
            "api_key_configured": bool(config.QWEN_API_KEY),
            "timeout": config.QWEN_TIMEOUT,
            "stream_timeout": config.QWEN_STREAM_TIMEOUT,
            "max_retries": config.QWEN_MAX_RETRIES
        },
        "datasets": {
            "default_dataset_ids": config.DEFAULT_DATASET_IDS,
            "default_document_ids": config.DEFAULT_DOCUMENT_IDS,
            "dataset_count": len(config.DEFAULT_DATASET_IDS)
        },
        "retrieval": {
            "similarity_threshold": config.DEFAULT_SIMILARITY_THRESHOLD,
            "top_k": config.DEFAULT_TOP_K,
            "vector_weight": config.DEFAULT_VECTOR_WEIGHT,
            "max_chunk_length": config.MAX_CHUNK_LENGTH
        },
        "service": {
            "host": config.HOST,
            "port": config.PORT,
            "debug": config.DEBUG,
            "log_level": config.LOG_LEVEL
        },
        "last_reload_time": config._last_reload_time.isoformat() if config._last_reload_time else None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/config/reload")
async def reload_config():
    """重新加载配置文件"""
    try:
        old_dataset_ids = config.DEFAULT_DATASET_IDS.copy()
        success = config.reload()

        if success:
            changes_detected = old_dataset_ids != config.DEFAULT_DATASET_IDS

            return {
                "status": "success",
                "message": "配置重新加载成功",
                "changes_detected": changes_detected,
                "old_dataset_ids": old_dataset_ids,
                "new_dataset_ids": config.DEFAULT_DATASET_IDS,
                "reload_time": config._last_reload_time.isoformat() if config._last_reload_time else None
            }
        else:
            raise HTTPException(status_code=500, detail="配置重新加载失败")

    except Exception as e:
        logger.error(f"配置重新加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置重新加载失败: {str(e)}")

@app.get("/config/datasets")
async def get_datasets_config():
    """获取数据集配置"""
    dataset_config = config.get_dataset_config()
    return {
        **dataset_config,
        "timestamp": datetime.now().isoformat()
    }

class DatasetConfigUpdate(BaseModel):
    dataset_ids: List[str] = Field(..., description="数据集ID列表", min_items=1)
    persist: bool = Field(default=False, description="是否持久化到.env文件")

@app.put("/config/datasets")
async def update_datasets_config(update_data: DatasetConfigUpdate):
    """更新数据集配置"""
    try:
        old_dataset_ids = config.DEFAULT_DATASET_IDS.copy()

        # 验证数据集ID格式
        validation_errors = []
        for i, dataset_id in enumerate(update_data.dataset_ids):
            dataset_id = dataset_id.strip()
            if not dataset_id:
                validation_errors.append(f"第{i+1}个数据集ID不能为空")
                continue

            is_valid, error_msg = config.validate_dataset_id(dataset_id)
            if not is_valid:
                validation_errors.append(f"第{i+1}个数据集ID无效: {error_msg}")

        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"数据集ID验证失败:\n" + "\n".join(validation_errors)
            )

        # 更新配置
        success = config.update_dataset_ids(update_data.dataset_ids)

        if success:
            result = {
                "status": "success",
                "message": "数据集配置更新成功",
                "old_dataset_ids": old_dataset_ids,
                "new_dataset_ids": config.DEFAULT_DATASET_IDS,
                "updated_count": len(config.DEFAULT_DATASET_IDS),
                "update_time": datetime.now().isoformat()
            }

            # 如果需要持久化到.env文件
            if update_data.persist:
                try:
                    from pathlib import Path
                    env_file = Path(__file__).parent / ".env"

                    if env_file.exists():
                        # 读取现有.env文件
                        with open(env_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # 更新DEFAULT_DATASET_IDS行
                        updated_lines = []
                        dataset_line_found = False
                        new_dataset_line = f"DEFAULT_DATASET_IDS={','.join(update_data.dataset_ids)}\n"

                        for line in lines:
                            if line.strip().startswith('DEFAULT_DATASET_IDS='):
                                updated_lines.append(new_dataset_line)
                                dataset_line_found = True
                            else:
                                updated_lines.append(line)

                        # 如果没找到，添加到文件末尾
                        if not dataset_line_found:
                            updated_lines.append('\n' + new_dataset_line)

                        # 写回文件
                        with open(env_file, 'w', encoding='utf-8') as f:
                            f.writelines(updated_lines)

                        result["persisted"] = True
                        result["message"] += " 并已保存到.env文件"
                    else:
                        result["persisted"] = False
                        result["message"] += " 但.env文件不存在，未能持久化"

                except Exception as persist_error:
                    logger.error(f"持久化配置失败: {persist_error}")
                    result["persisted"] = False
                    result["persist_error"] = str(persist_error)
                    result["message"] += f" 但持久化失败: {persist_error}"

            logger.info(f"数据集配置更新: {old_dataset_ids} -> {config.DEFAULT_DATASET_IDS}")
            return result
        else:
            raise HTTPException(status_code=500, detail="数据集配置更新失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据集配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新数据集配置失败: {str(e)}")

@app.post("/qa/stream")
async def qa_stream(request: QAStreamRequest):
    """流式智能问答接口 - 实时返回生成的回答"""
    start_time = time.time()
    logger.info(f"流式问答请求 - 问题: {request.question[:50]}...")

    async def generate_stream():
        client_disconnected = False
        try:
            # 1. 智能优化检索参数（流式版本）
            base_params = {
                'similarity_threshold': request.similarity_threshold,
                'vector_similarity_weight': request.vector_similarity_weight,
                'top_k': request.top_k
            }
            optimized_params = smart_retrieval.get_optimized_params(request.question, base_params)

            final_similarity_threshold = optimized_params['similarity_threshold']
            final_vector_weight = optimized_params['vector_similarity_weight']
            final_top_k = optimized_params['top_k']

            # 2. 调用RAGflow获取知识碎片
            ragflow_result = {"documents": []}
            ragflow_error = None
            ragflow_start = time.time()

            try:
                logger.info(f"流式问答-调用RAGflow（智能优化）- 阈值:{final_similarity_threshold:.3f}, TopK:{final_top_k}")
                ragflow_result = await ragflow_client.retrieve(
                    question=request.question,
                    dataset_ids=request.dataset_ids,
                    document_ids=request.document_ids,
                    similarity_threshold=final_similarity_threshold,
                    vector_similarity_weight=final_vector_weight,
                    top_k=final_top_k,
                    page_size=request.page_size,
                    keyword=request.keyword,
                    highlight=request.highlight,
                    rerank_id=request.rerank_id
                )
                ragflow_time = time.time() - ragflow_start
                logger.info(f"RAGflow返回 {len(ragflow_result.get('documents', []))} 个文档")

            except Exception as e:
                ragflow_error = str(e)
                ragflow_time = time.time() - ragflow_start
                logger.warning(f"RAGflow请求失败，将使用Qwen直接回答: {e}")

            # 发送检索状态
            yield f"data: {json.dumps({'type': 'retrieval_complete', 'documents_found': len(ragflow_result.get('documents', [])), 'time': ragflow_time}, ensure_ascii=False)}\n\n"

            # 2. 构建上下文
            documents = ragflow_result.get("documents", [])
            context_used = False

            # 过滤低质量文档（流式版本使用优化阈值）
            quality_documents = []
            if documents:
                for doc in documents:
                    score = doc.get('score', 0)
                    if score >= final_similarity_threshold:
                        quality_documents.append(doc)

            # 在调试模式下输出文档信息
            if config.DEBUG:
                logger.debug(f"流式接口 - 检索到的总文档数: {len(documents)}")
                logger.debug(f"流式接口 - 高质量文档数: {len(quality_documents)}")
                for i, doc in enumerate(quality_documents):
                    logger.debug(f"流式接口 - 文档 {i+1}:")
                    logger.debug(f"  内容: {doc.get('content', '')[:200]}...")
                    logger.debug(f"  分数: {doc.get('score', 0)}")
                    logger.debug(f"  元数据: {doc.get('metadata', {})}")

            # 获取提示词模板
            prompt_template = prompt_manager.get_template(request.prompt_template)

            # 检查是否使用citation_focused或enhanced_citation模板（流式版本）
            if request.prompt_template in ["citation_focused", "enhanced_citation"] and quality_documents:
                logger.info(f"流式接口：开始使用{request.prompt_template}模板处理{len(quality_documents)}个文档")
                # 使用增强解析器处理文档（流式版本）
                try:
                    logger.info("流式接口：调用enhanced_parser.batch_parse_documents（快速模式）")
                    enhanced_documents = enhanced_parser.batch_parse_documents(quality_documents, fast_mode=True)
                    logger.info(f"流式接口：enhanced_parser处理完成，返回{len(enhanced_documents)}个文档")
                    parsed_documents = []

                    for doc in enhanced_documents:
                        content = doc.get('content', '').strip()
                        metadata = doc.get('metadata', {})

                        if content:
                            # 限制文本内容长度
                            if len(content) > config.MAX_CHUNK_LENGTH:
                                content = content[:config.MAX_CHUNK_LENGTH] + "..."

                            # 从metadata中获取链接信息，优先使用智能匹配的结果
                            wechat_link = metadata.get('wechat_link')
                            source_link = metadata.get('source_link') or metadata.get('matched_link')

                            parsed_documents.append({
                                'text': content,
                                'wechat_link': wechat_link,
                                'source_link': source_link,
                                'parsing_metadata': metadata.get('parsing_metadata', {})
                            })

                    logger.info(f"流式接口增强解析处理了 {len(enhanced_documents)} 个文档，获得 {len(parsed_documents)} 个有效解析结果")

                except Exception as e:
                    logger.warning(f"流式接口增强解析失败，回退到标准解析: {e}")
                    # 回退到原有解析逻辑
                    parsed_documents = []
                    for doc in quality_documents:
                        content = doc.get('content', '').strip()
                        if content:
                            text_content, wechat_link, source_link = parse_content_and_link(content)
                            if text_content:
                                if len(text_content) > config.MAX_CHUNK_LENGTH:
                                    text_content = text_content[:config.MAX_CHUNK_LENGTH] + "..."
                                parsed_documents.append({
                                    'text': text_content,
                                    'wechat_link': wechat_link,
                                    'source_link': source_link
                                })

                if parsed_documents:
                    # 使用增强的引用格式构建上下文
                    context = prompt_template + "\n\n"
                    context += "相关资料：\n"
                    for idx, parsed_doc in enumerate(parsed_documents):
                        context += f"{idx+1}. [正文]: {parsed_doc['text']}\n"
                        if parsed_doc['wechat_link']:
                            context += f"   [微信链接]: {parsed_doc['wechat_link']}\n"
                        if parsed_doc['source_link']:
                            context += f"   [原文链接]: {parsed_doc['source_link']}\n"
                        context += "\n"

                    context += f"问题：{request.question}\n"
                    context_used = True
                    logger.info(f"流式接口使用{request.prompt_template}模板，解析出 {len(parsed_documents)} 个带链接的文档")
                else:
                    # 容错处理：没有任何文档能解析出有效的正文内容，回退到原有逻辑
                    logger.warning(f"流式接口citation_focused模板下未能解析出任何有效的正文内容，回退到原有上下文构建逻辑")
                    context = prompt_template + "\n\n"
                    context += "相关资料：\n"
                    for idx, doc in enumerate(quality_documents):
                        content = doc.get('content', '').strip()
                        if content:
                            if len(content) > config.MAX_CHUNK_LENGTH:
                                content = content[:config.MAX_CHUNK_LENGTH] + "..."
                            context += f"{idx+1}. {content}\n\n"
                    context += f"问题：{request.question}\n\n请基于上述资料回答问题，如果资料中没有明确答案，请基于你的专业知识补充回答。不需要提及相似度、评分或技术参数等信息。"
                    context_used = True
            elif quality_documents:
                # 原有的上下文构建逻辑（非citation_focused模板）
                context = prompt_template + "\n\n"
                context += "相关资料：\n"
                for idx, doc in enumerate(quality_documents):
                    content = doc.get('content', '').strip()
                    if content:
                        if len(content) > config.MAX_CHUNK_LENGTH:
                            content = content[:config.MAX_CHUNK_LENGTH] + "..."
                        context += f"{idx+1}. {content}\n\n"
                context += f"问题：{request.question}\n\n请基于上述资料回答问题，如果资料中没有明确答案，请基于你的专业知识补充回答。不需要提及相似度、评分或技术参数等信息。"
                context_used = True
                logger.info(f"使用了 {len(quality_documents)} 个高质量文档作为上下文")
            else:
                # 没有检索到相关文档时的处理策略
                if ragflow_error:
                    context = f"请基于你的知识回答以下问题：{request.question}\n\n注意：由于文档检索服务暂时不可用，我将基于我的训练知识为您提供答案。"
                elif documents:
                    context = f"请基于你的知识详细回答以下问题：{request.question}\n\n注意：我在知识库中检索到了一些文档，但相似度较低，因此主要基于我的训练知识为您提供答案。"
                else:
                    context = f"请基于你的知识详细回答以下问题：{request.question}\n\n注意：我在当前知识库中没有找到直接相关的资料，因此基于我的训练知识为您提供答案。"

            # 发送上下文准备完成状态
            logger.info(f"流式接口：上下文准备完成，context_used={context_used}, quality_documents={len(quality_documents)}")
            yield f"data: {json.dumps({'type': 'context_ready', 'context_used': context_used, 'documents_used': len(quality_documents)}, ensure_ascii=False)}\n\n"

            # 3. 调用Qwen流式生成
            logger.info(f"流式接口：开始调用Qwen流式生成，上下文长度={len(context)}")

            # 在调试模式下输出完整的上下文内容
            if config.DEBUG:
                logger.debug(f"完整上下文内容:\n{context}")
                logger.debug(f"上下文长度: {len(context)} 字符")

            try:
                qwen_stream = qwen_client.chat(
                    messages=[{"role": "user", "content": context}],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )

                full_content = ""
                chunk_count = 0

                async for chunk in qwen_stream:
                    # 检查客户端是否断开连接
                    if client_disconnected:
                        logger.info("检测到客户端断开连接，停止流式生成")
                        break

                    chunk_count += 1

                    if chunk.get("is_complete"):
                        # 在发送完成信号之前，先发送引用溯源信息
                        if context_used and quality_documents:
                            try:
                                citations = extract_citations_from_documents(quality_documents)
                                if citations:  # 只有当有引用信息时才发送
                                    citations_data = {
                                        'type': 'citations',
                                        'data': citations
                                    }
                                    yield f"data: {json.dumps(citations_data, ensure_ascii=False)}\n\n"
                                    logger.info(f"发送了 {len(citations)} 个引用来源")
                            except Exception as citations_error:
                                logger.error(f"提取引用信息失败: {citations_error}")
                                # 即使引用提取失败，也要继续正常的流程

                        # ==================================================
                        # 流式接口人物主页链接后处理逻辑 - 修复版（与普通接口保持一致）
                        # ==================================================
                        try:
                            # 在问题和答案中搜索提及的人物
                            if config.ENABLE_NAME_LINKER:
                                mentioned_in_question = name_linker.find_mentioned_names(request.question)
                                mentioned_in_answer = name_linker.find_mentioned_names(full_content)
                            else:
                                mentioned_in_question = {}
                                mentioned_in_answer = {}
                                logger.info("已跳过人名检测")

                            # 合并结果并去重，优先保留有链接的人物
                            all_mentioned = {}
                            all_mentioned.update(mentioned_in_question)
                            all_mentioned.update(mentioned_in_answer)

                            # 只保留有有效链接的人物
                            valid_mentioned = {name: url for name, url in all_mentioned.items() if url}

                            if valid_mentioned:
                                # 使用与普通接口相同的格式化方式
                                person_links_md = name_linker.format_person_links_markdown(valid_mentioned)

                                if person_links_md:
                                    # 将格式化的人物链接附加到完整答案中
                                    updated_content = full_content + person_links_md

                                    # 发送更新后的完整内容
                                    chunk['full_content'] = updated_content
                                    logger.info(f"流式接口为答案附加了 {len(valid_mentioned)} 个人物主页链接: {list(valid_mentioned.keys())}")

                        except Exception as e:
                            logger.error(f"流式接口处理人物主页链接时出错: {e}")
                        # ==================================================
                        # 流式接口人物链接处理逻辑结束
                        # ==================================================

                        # 发送完成信号
                        total_time = time.time() - start_time
                        completion_data = {
                            'type': 'complete',
                            'full_content': chunk.get('full_content', full_content),
                            'total_time': total_time,
                            'context_used': context_used,
                            'chunks_generated': chunk_count
                        }
                        yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        logger.info(f"流式生成完成 - 总块数: {chunk_count}, 耗时: {total_time:.2f}s")
                        break
                    else:
                        # 发送内容块
                        content = chunk.get("content", "")
                        if content:  # 只发送有内容的块
                            full_content += content
                            content_data = {
                                'type': 'content',
                                'content': content,
                                'timestamp': chunk.get('timestamp', time.time() - start_time),
                                'chunk_index': chunk_count
                            }
                            yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"

                            # 每10个块输出一次进度日志
                            if chunk_count % 10 == 0:
                                logger.debug(f"流式生成进度 - 已生成 {chunk_count} 块, 内容长度: {len(full_content)}")

            except asyncio.CancelledError:
                logger.info("Qwen流式生成被取消")
                yield f"data: {json.dumps({'type': 'cancelled', 'message': '生成已被取消'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as qwen_error:
                logger.error(f"Qwen流式生成异常: {qwen_error}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'AI生成出错: {str(qwen_error)}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

        except Exception as e:
            logger.error(f"流式问答异常: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'流式问答服务异常: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    # 创建增强的流式响应，支持连接断开检测
    async def enhanced_stream():
        try:
            async for chunk in generate_stream():
                yield chunk
        except asyncio.CancelledError:
            logger.info("流式响应被取消")
            yield f"data: {json.dumps({'type': 'cancelled', 'message': '连接已断开'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            raise
        except Exception as e:
            logger.error(f"流式响应异常: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'响应异常: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        enhanced_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Access-Control-Allow-Origin": "*",  # 支持跨域
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动RAGFlow + Qwen 智能问答系统 v2.0")
    logger.info(f"服务地址: http://{config.HOST}:{config.PORT}")
    logger.info(f"API文档: http://{config.HOST}:{config.PORT}/docs")

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )