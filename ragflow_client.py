import httpx
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from config import config
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# 全局HTTP客户端配置
HTTP_TIMEOUT = httpx.Timeout(
    timeout=config.RAGFLOW_TIMEOUT,
    connect=5.0,
    read=config.RAGFLOW_TIMEOUT - 5,
    write=5.0
)

HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=5,
    max_connections=10,
    keepalive_expiry=30.0
)

@asynccontextmanager
async def get_http_client():
    """获取配置好的HTTP客户端"""
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        limits=HTTP_LIMITS,
        follow_redirects=True,
        headers={"Content-Type": "application/json; charset=utf-8"}
    ) as client:
        yield client

class RAGFlowClient:
    """增强版RAGFlow客户端"""
    
    def __init__(self):
        self.base_url = config.RAGFLOW_URL
        self.api_key = config.RAGFLOW_API_KEY
        self.max_retries = config.RAGFLOW_MAX_RETRIES
        
    async def health_check(self) -> Dict:
        """检查RAGFlow服务健康状态 - 优化版，不消耗token"""
        start_time = time.time()
        try:
            # 直接测试API端点，不发送实际检索请求以节省token
            headers = self._get_headers()
            
            async with get_http_client() as client:
                # 发送一个空的测试请求或GET请求来检查连接
                test_url = self.base_url.replace('/retrieval', '/health') if '/retrieval' in self.base_url else f"{self.base_url}/health"
                
                try:
                    # 先尝试health端点
                    resp = await client.get(test_url, headers=headers, timeout=5.0)
                    response_time = time.time() - start_time
                    
                    if resp.status_code == 200:
                        return {
                            "status": "healthy",
                            "message": "RAGFlow服务连接正常" if config.DEBUG else "RAGFlow service connection normal",
                            "response_time": response_time
                        }
                except:
                    pass
                
                # 如果health端点不存在，测试原始端点但不发送完整请求
                try:
                    resp = await client.get(self.base_url, headers=headers, timeout=5.0)
                    response_time = time.time() - start_time
                    
                    # 即使返回405 Method Not Allowed也表示服务可达
                    if resp.status_code in [200, 405, 404]:
                        return {
                            "status": "healthy",
                            "message": "RAGFlow服务连接正常" if config.DEBUG else "RAGFlow service connection normal",
                            "response_time": response_time
                        }
                except:
                    pass
                    
                # 最后尝试基础URL连接测试
                base_url = self.base_url.split('/api')[0] if '/api' in self.base_url else self.base_url
                resp = await client.get(base_url, timeout=5.0)
                response_time = time.time() - start_time
                
                if resp.status_code < 500:  # 任何非5xx错误都表示服务基本可达
                    return {
                        "status": "healthy",
                        "message": "RAGFlow服务连接正常" if config.DEBUG else "RAGFlow service connection normal", 
                        "response_time": response_time
                    }
                else:
                    raise Exception(f"服务返回错误状态码: {resp.status_code}")
                
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status": "unhealthy", 
                "message": f"RAGFlow服务连接失败: {str(e)}" if config.DEBUG else f"RAGFlow service connection failed: {str(e)}",
                "response_time": response_time
            }
    
    async def get_dataset_info(self, dataset_id: str) -> Dict:
        """获取数据集信息（如果RAGFlow提供此接口）"""
        # 这是一个示例方法，实际需要根据RAGFlow的API文档调整
        try:
            headers = self._get_headers()
            dataset_url = f"{self.base_url.replace('/retrieval', '')}/datasets/{dataset_id}"
            
            async with get_http_client() as client:
                resp = await client.get(dataset_url, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    return {"error": f"无法获取数据集信息: {resp.status_code}"}
        except Exception as e:
            return {"error": f"获取数据集信息失败: {str(e)}"}
    
    async def retrieve(self, question: str, dataset_ids: Optional[List[str]] = None,
                      document_ids: Optional[List[str]] = None,
                      similarity_threshold: float = None,
                      vector_similarity_weight: float = None,
                      top_k: int = None, page_size: int = None,
                      keyword: bool = True, highlight: bool = True,
                      rerank_id: str = "") -> Dict:
        """
        增强版检索方法，包含详细的调试信息和性能监控
        """
        start_time = time.time()
        
        # 使用默认值 - 修复falsy值(如0)被默认值覆盖的问题
        dataset_ids = dataset_ids if dataset_ids is not None else config.DEFAULT_DATASET_IDS
        document_ids = document_ids if document_ids is not None else config.DEFAULT_DOCUMENT_IDS
        similarity_threshold = config.DEFAULT_SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
        vector_similarity_weight = config.DEFAULT_VECTOR_WEIGHT if vector_similarity_weight is None else vector_similarity_weight
        top_k = config.DEFAULT_TOP_K if top_k is None else top_k
        page_size = top_k if page_size is None else page_size
        
        # 确保参数类型正确
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]
        if isinstance(document_ids, str):
            document_ids = [document_ids]
        
        payload = {
            "question": question,
            "dataset_ids": dataset_ids,
            "document_ids": document_ids,
            "page": 1,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "rerank_id": rerank_id,
            "keyword": keyword,
            "highlight": highlight
        }
        
        headers = self._get_headers()
        
        if config.DEBUG:
            logger.info(f"RAGFlow检索请求 - 问题: {question[:50]}...")
            logger.info(f"数据集ID: {dataset_ids}")
            logger.debug(f"完整请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        else:
            logger.info(f"RAGFlow retrieval request - question: {question[:50]}...")
            logger.info(f"Dataset IDs: {dataset_ids}")
            logger.debug(f"Full request params: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with get_http_client() as client:
                    if config.DEBUG:
                        logger.debug(f"第{attempt + 1}次尝试调用RAGFlow")
                    else:
                        logger.debug(f"RAGFlow attempt {attempt + 1}")
                    
                    # 确保payload能正确编码
                    json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                    resp = await client.post(
                        self.base_url, 
                        content=json_data, 
                        headers={**headers, "Content-Type": "application/json; charset=utf-8"}
                    )
                    response_time = time.time() - start_time
                    
                    if config.DEBUG:
                        logger.info(f"RAGFlow响应状态: {resp.status_code} (耗时: {response_time:.2f}s)")
                    else:
                        logger.info(f"RAGFlow response status: {resp.status_code} (time: {response_time:.2f}s)")
                    
                    # 记录响应头信息
                    if config.DEBUG:
                        logger.debug(f"响应头: {dict(resp.headers)}")
                    else:
                        logger.debug(f"Response headers: {dict(resp.headers)}")
                    
                    # 获取响应文本用于调试
                    response_text = resp.text
                    logger.debug(f"原始响应前500字符: {response_text[:500]}")
                    
                    resp.raise_for_status()
                    
                    try:
                        result = resp.json()
                    except json.JSONDecodeError as json_error:
                        logger.error(f"JSON解析失败: {json_error}")
                        logger.error(f"完整响应内容: {response_text}")
                        raise Exception(f"RAGFlow返回的不是有效JSON格式: {json_error}")
                    
                    # 验证响应格式
                    if not isinstance(result, dict):
                        raise Exception(f"RAGFlow返回数据格式异常，期望dict，实际: {type(result)}")
                    
                    # 检查是否有错误信息
                    if "error" in result:
                        raise Exception(f"RAGFlow服务错误: {result.get('error')}")
                    
                    # 处理文档数据 - 修复RAGflow数据结构解析并支持来源信息提取
                    documents = []
                    if 'data' in result and 'chunks' in result['data']:
                        # RAGflow新版本数据结构: result.data.chunks
                        chunks = result['data']['chunks']
                        for chunk in chunks:
                            # 转换RAGflow chunk格式到我们期望的document格式
                            doc = {
                                'content': chunk.get('content', ''),
                                'score': chunk.get('similarity', chunk.get('vector_similarity', 0)),
                                'document_name': chunk.get('document_keyword', '未知文档'),
                                'document_id': chunk.get('document_id', ''),
                                'chunk_id': chunk.get('id', ''),
                                'highlights': chunk.get('highlight', ''),
                                'metadata': {
                                    'dataset_id': chunk.get('dataset_id', ''),
                                    'term_similarity': chunk.get('term_similarity', 0),
                                    'vector_similarity': chunk.get('vector_similarity', 0),
                                    'positions': chunk.get('positions', []),
                                    'important_keywords': chunk.get('important_keywords', []),
                                    # 提取来源信息 - 支持多种可能的字段名
                                    'source': chunk.get('source', chunk.get('document_keyword', chunk.get('filename', '未知文档'))),
                                    'link': chunk.get('link', chunk.get('document_url', chunk.get('url', ''))),
                                    'document_path': chunk.get('document_path', chunk.get('file_path', '')),
                                    'page_number': chunk.get('page_number', chunk.get('page', '')),
                                    'section': chunk.get('section', chunk.get('title', ''))
                                }
                            }
                            documents.append(doc)
                    elif 'documents' in result:
                        # 兼容旧版本数据结构，同样支持来源信息提取
                        raw_documents = result.get('documents', [])
                        for doc in raw_documents:
                            # 确保每个文档都有metadata字段，并提取来源信息
                            metadata = doc.get('metadata', {})
                            if not isinstance(metadata, dict):
                                metadata = {}
                            
                            # 添加来源信息到metadata
                            metadata.update({
                                'source': metadata.get('source', doc.get('document_name', doc.get('filename', '未知文档'))),
                                'link': metadata.get('link', doc.get('document_url', doc.get('url', ''))),
                                'document_path': metadata.get('document_path', doc.get('file_path', '')),
                                'page_number': metadata.get('page_number', doc.get('page', '')),
                                'section': metadata.get('section', doc.get('title', ''))
                            })
                            
                            doc['metadata'] = metadata
                            documents.append(doc)
                    processed_docs = self._process_documents(documents)
                    
                    # 添加调试信息
                    result.update({
                        "documents": documents,  # 确保返回解析后的documents
                        "response_time": response_time,
                        "processed_documents": processed_docs,
                        "retrieval_stats": {
                            "total_documents": len(documents),
                            "avg_score": sum(doc.get('score', 0) for doc in documents) / len(documents) if documents else 0,
                            "score_range": {
                                "min": min(doc.get('score', 0) for doc in documents) if documents else 0,
                                "max": max(doc.get('score', 0) for doc in documents) if documents else 0
                            },
                            "dataset_coverage": self._analyze_dataset_coverage(documents, dataset_ids)
                        },
                        "request_info": {
                            "question_length": len(question),
                            "dataset_ids": dataset_ids,
                            "similarity_threshold": similarity_threshold,
                            "top_k": top_k,
                            "attempt": attempt + 1
                        }
                    })
                    
                    logger.info(f"RAGFlow检索成功: {len(documents)}个文档, 平均得分: {result['retrieval_stats']['avg_score']:.3f}")
                    
                    return result
                    
            except httpx.TimeoutException as e:
                last_error = f"请求超时: {e}"
                logger.warning(f"RAGFlow请求超时 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
            except httpx.HTTPStatusError as e:
                error_detail = self._extract_error_detail(e.response)
                last_error = f"HTTP错误 {e.response.status_code}: {error_detail}"
                
                if config.DEBUG:
                    logger.error(f"RAGFlow HTTP错误: {e.response.status_code}")
                    logger.error(f"错误详情: {error_detail}")
                else:
                    logger.error(f"RAGFlow HTTP error: {e.response.status_code}")
                    logger.error(f"Error details: {error_detail}")
                
                # 对于某些错误不进行重试
                if e.response.status_code in [400, 401, 403, 404, 422]:
                    break
                    
            except Exception as e:
                last_error = str(e)
                if config.DEBUG:
                    logger.error(f"RAGFlow请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.error(f"RAGFlow request error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                if config.DEBUG:
                    logger.info(f"等待 {wait_time}s 后重试...")
                else:
                    logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        # 所有重试都失败了
        total_time = time.time() - start_time
        if config.DEBUG:
            error_msg = f"RAGFlow请求失败 (总耗时: {total_time:.2f}s): {last_error}"
        else:
            error_msg = f"RAGFlow request failed (total time: {total_time:.2f}s): {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _extract_error_detail(self, response: httpx.Response) -> str:
        """提取错误详情"""
        try:
            error_data = response.json()
            return error_data.get("message", error_data.get("error", response.text))
        except:
            return response.text[:200]
    
    def _process_documents(self, documents: List[Dict]) -> List[Dict]:
        """处理和增强文档数据"""
        processed = []
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '').strip()
            
            processed_doc = {
                "index": i + 1,
                "content": content,
                "content_length": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "score": doc.get('score', 0),
                "document_name": doc.get('document_name', f'文档{i+1}'),
                "document_id": doc.get('document_id', ''),
                "chunk_id": doc.get('chunk_id', ''),
                "highlights": doc.get('highlights', []),
                "metadata": doc.get('metadata', {}),
                "score_level": self._categorize_score(doc.get('score', 0))
            }
            
            processed.append(processed_doc)
        
        return processed
    
    def _categorize_score(self, score: float) -> str:
        """将相似度分数分类"""
        if score >= 0.8:
            return "高相关"
        elif score >= 0.6:
            return "中等相关"
        elif score >= 0.3:
            return "低相关"
        else:
            return "弱相关"
    
    def _analyze_dataset_coverage(self, documents: List[Dict], dataset_ids: List[str]) -> Dict:
        """分析数据集覆盖情况 - 增强版多知识库分析"""
        if not documents:
            return {
                "coverage": 0, 
                "datasets_with_results": [], 
                "datasets_without_results": dataset_ids,
                "dataset_stats": {ds_id: {"count": 0, "avg_score": 0} for ds_id in dataset_ids}
            }
        
        # 统计每个数据集的文档数量和得分
        dataset_stats = {}
        for ds_id in dataset_ids:
            dataset_docs = [doc for doc in documents if doc.get('metadata', {}).get('dataset_id') == ds_id]
            if dataset_docs:
                dataset_stats[ds_id] = {
                    "count": len(dataset_docs),
                    "avg_score": sum(doc.get('score', 0) for doc in dataset_docs) / len(dataset_docs),
                    "max_score": max(doc.get('score', 0) for doc in dataset_docs),
                    "min_score": min(doc.get('score', 0) for doc in dataset_docs)
                }
            else:
                dataset_stats[ds_id] = {"count": 0, "avg_score": 0, "max_score": 0, "min_score": 0}
        
        datasets_with_results = [ds_id for ds_id, stats in dataset_stats.items() if stats["count"] > 0]
        datasets_without_results = [ds_id for ds_id, stats in dataset_stats.items() if stats["count"] == 0]
        
        return {
            "coverage": len(datasets_with_results) / max(len(dataset_ids), 1),
            "datasets_with_results": datasets_with_results,
            "datasets_without_results": datasets_without_results,
            "dataset_stats": dataset_stats,
            "total_datasets": len(dataset_ids),
            "active_datasets": len(datasets_with_results)
        }

# 全局客户端实例
ragflow_client = RAGFlowClient()

# 保持向后兼容的函数接口
async def query_ragflow(question: str, dataset_ids: Optional[List[str]] = None,
                       document_ids: Optional[List[str]] = None,
                       similarity_threshold: float = None,
                       vector_similarity_weight: float = None,
                       top_k: int = None, page_size: int = None,
                       keyword: bool = True, highlight: bool = True,
                       rerank_id: str = "") -> Dict:
    """向后兼容的查询函数"""
    return await ragflow_client.retrieve(
        question=question,
        dataset_ids=dataset_ids,
        document_ids=document_ids,
        similarity_threshold=similarity_threshold,
        vector_similarity_weight=vector_similarity_weight,
        top_k=top_k,
        page_size=page_size,
        keyword=keyword,
        highlight=highlight,
        rerank_id=rerank_id
    )

import asyncio  # 添加这个导入 