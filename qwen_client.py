import asyncio
import httpx
import logging
import json
import time
from typing import Dict, List, Optional
from config import config
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# 全局HTTP客户端配置
HTTP_TIMEOUT = httpx.Timeout(
    timeout=config.QWEN_TIMEOUT,
    connect=10.0,
    read=config.QWEN_TIMEOUT - 10,
    write=10.0
)

# 流式输出专用HTTP超时配置
HTTP_STREAM_TIMEOUT = httpx.Timeout(
    timeout=config.QWEN_STREAM_TIMEOUT,
    connect=15.0,
    read=config.QWEN_STREAM_TIMEOUT - 15,  # 为流式输出预留更长的读取时间
    write=15.0
)

HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=20,
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

@asynccontextmanager
async def get_stream_http_client():
    """获取流式输出专用的HTTP客户端"""
    async with httpx.AsyncClient(
        timeout=HTTP_STREAM_TIMEOUT,  # 使用流式专用超时配置
        limits=HTTP_LIMITS,
        follow_redirects=True,
        headers={"Content-Type": "application/json; charset=utf-8"}
    ) as client:
        yield client

class QwenClient:
    """增强版Qwen大模型客户端"""
    
    def __init__(self):
        self.base_url = config.QWEN_URL
        self.api_key = config.QWEN_API_KEY
        self.model = config.QWEN_MODEL
        self.max_retries = config.QWEN_MAX_RETRIES
        
    async def health_check(self) -> Dict:
        """检查Qwen服务健康状态 - 优化版，使用最小token测试"""
        start_time = time.time()
        try:
            # 使用最短的测试内容和最小token数来节省费用
            result = await self.chat(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,  # 最小token数
                temperature=0.1  # 降低随机性
            )
            response_time = time.time() - start_time
            return {
                "status": "healthy",
                "message": "Qwen服务连接正常" if config.DEBUG else "Qwen service connection normal",
                "model": self.model,
                "response_time": response_time
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status": "unhealthy",
                "message": f"Qwen服务连接失败: {str(e)}" if config.DEBUG else f"Qwen service connection failed: {str(e)}",
                "model": self.model, 
                "response_time": response_time
            }
    
    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2000, top_p: float = 0.8,
             stream: bool = False):
        """
        调用Qwen聊天接口
        
        :param messages: 对话消息列表
        :param temperature: 温度参数，控制回答的创造性
        :param max_tokens: 最大token数
        :param top_p: top_p参数
        :param stream: 是否使用流式响应
        :return: 如果stream=True，返回异步生成器；否则返回协程
        """
        if stream:
            return self._stream_chat(messages, temperature, max_tokens, top_p)
        else:
            return self._non_stream_chat(messages, temperature, max_tokens, top_p)
    
    async def _stream_chat(self, messages: List[Dict], temperature: float, max_tokens: int, top_p: float):
        """流式聊天响应"""
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream"
        }
        
        if config.DEBUG:
            logger.info(f"Qwen流式请求 - 模型: {self.model}, 消息数: {len(messages)}")
        else:
            logger.info(f"Qwen streaming request - model: {self.model}, messages: {len(messages)}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with get_stream_http_client() as client:  # 使用流式专用HTTP客户端
                    json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                    
                    async with client.stream(
                        "POST",
                        self.base_url,
                        content=json_data,
                        headers={**headers, "Content-Type": "application/json; charset=utf-8"}
                    ) as response:
                        response.raise_for_status()
                        
                        content_buffer = []
                        async for line in response.aiter_lines():
                            if line.strip():
                                if line.startswith("data: "):
                                    data_str = line[6:]  # 移除 "data: " 前缀
                                    
                                    if data_str.strip() == "[DONE]":
                                        break
                                    
                                    try:
                                        chunk_data = json.loads(data_str)
                                        choices = chunk_data.get("choices", [])
                                        if choices and "delta" in choices[0]:
                                            delta = choices[0]["delta"]
                                            content = delta.get("content", "")
                                            if content:
                                                content_buffer.append(content)
                                                yield {
                                                    "content": content,
                                                    "is_complete": False,
                                                    "timestamp": time.time() - start_time
                                                }
                                    except json.JSONDecodeError:
                                        continue
                        
                        # 发送完成信号
                        yield {
                            "content": "",
                            "is_complete": True,
                            "full_content": "".join(content_buffer),
                            "timestamp": time.time() - start_time,
                            "total_tokens": len("".join(content_buffer))  # 简单估算
                        }
                        return
                        
            except Exception as e:
                last_error = str(e)
                if config.DEBUG:
                    logger.error(f"Qwen流式请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.error(f"Qwen streaming error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        raise Exception(f"Qwen流式请求失败: {last_error}")
    
    async def _non_stream_chat(self, messages: List[Dict], temperature: float, max_tokens: int, top_p: float) -> Dict:
        """非流式聊天响应"""
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if config.DEBUG:
            logger.info(f"Qwen请求 - 模型: {self.model}, 消息数: {len(messages)}")
            logger.debug(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        else:
            logger.info(f"Qwen request - model: {self.model}, messages: {len(messages)}")
            logger.debug(f"Request params: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with get_http_client() as client:
                    if config.DEBUG:
                        logger.debug(f"第{attempt + 1}次尝试调用Qwen API")
                    else:
                        logger.debug(f"Qwen API attempt {attempt + 1}")
                    
                    # 确保payload能正确编码
                    json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                    resp = await client.post(
                        self.base_url, 
                        content=json_data,
                        headers={**headers, "Content-Type": "application/json; charset=utf-8"}
                    )
                    response_time = time.time() - start_time
                    
                    if config.DEBUG:
                        logger.info(f"Qwen响应状态: {resp.status_code} (耗时: {response_time:.2f}s)")
                    else:
                        logger.info(f"Qwen response status: {resp.status_code} (time: {response_time:.2f}s)")
                    
                    resp.raise_for_status()
                    
                    try:
                        data = resp.json()
                    except json.JSONDecodeError as json_error:
                        logger.error(f"JSON decode failed: {json_error}")
                        logger.error(f"Response content: {resp.text}")
                        raise Exception(f"Qwen returned invalid JSON format: {json_error}")
                    
                    # 验证响应格式
                    choices = data.get("choices", [])
                    if not choices:
                        raise Exception("Qwen returned data format error: missing choices field")
                    
                    message = choices[0].get("message", {})
                    content = message.get("content", "").strip()
                    
                    if not content:
                        raise Exception("Qwen returned empty answer")
                    
                    # 构建返回结果
                    result = {
                        "content": content,
                        "response_time": response_time,
                        "model": self.model,
                        "usage": data.get("usage", {}),
                        "finish_reason": choices[0].get("finish_reason", ""),
                        "request_info": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "input_length": sum(len(msg.get("content", "")) for msg in messages),
                            "attempt": attempt + 1
                        }
                    }
                    
                    if config.DEBUG:
                        logger.info(f"Qwen调用成功 - 内容长度: {len(content)}, "
                                  f"Token使用: {data.get('usage', {}).get('total_tokens', 'N/A')}")
                    else:
                        logger.info(f"Qwen call successful - content length: {len(content)}, "
                                  f"Tokens used: {data.get('usage', {}).get('total_tokens', 'N/A')}")
                    
                    return result
                    
            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                if config.DEBUG:
                    logger.warning(f"Qwen请求超时 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.warning(f"Qwen request timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                
            except httpx.HTTPStatusError as e:
                error_detail = self._extract_error_detail(e.response)
                last_error = f"HTTP error {e.response.status_code}: {error_detail}"
                
                if config.DEBUG:
                    logger.error(f"Qwen HTTP错误: {e.response.status_code}")
                    logger.error(f"错误详情: {error_detail}")
                else:
                    logger.error(f"Qwen HTTP error: {e.response.status_code}")
                    logger.error(f"Error details: {error_detail}")
                
                # 对于某些错误不进行重试
                if e.response.status_code in [400, 401, 403]:
                    break
                    
            except Exception as e:
                last_error = str(e)
                if config.DEBUG:
                    logger.error(f"Qwen请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.error(f"Qwen request error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
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
            error_msg = f"Qwen请求失败 (总耗时: {total_time:.2f}s): {last_error}"
        else:
            error_msg = f"Qwen request failed (total time: {total_time:.2f}s): {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract error details"""
        try:
            error_data = response.json()
            error_info = error_data.get("error", {})
            if isinstance(error_info, dict):
                return error_info.get("message", error_data.get("message", response.text))
            else:
                return str(error_info)
        except:
            return response.text[:200]

# 全局客户端实例
qwen_client = QwenClient()

# 保持向后兼容的函数接口
async def query_qwen(prompt: str, max_retries: int = None) -> str:
    """
    向后兼容的查询函数
    
    :param prompt: 拼接后的上下文和问题
    :param max_retries: 最大重试次数（已废弃，使用配置中的值）
            :return: Answer string returned by Qwen
    """
    try:
        result = await qwen_client.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        return result["content"]
    except Exception as e:
        if config.DEBUG:
            logger.error(f"query_qwen调用失败: {e}")
        else:
            logger.error(f"query_qwen call failed: {e}")
        raise

