"""
增强内容解析器 - 改进文档内容和链接的解析逻辑

主要功能：
1. 智能解析文档内容中的链接信息
2. 支持多种链接格式和模式
3. 与智能引用匹配器集成
4. 提供容错和回退机制
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from citation_matcher import citation_matcher

logger = logging.getLogger(__name__)

class EnhancedContentParser:
    """增强内容解析器"""
    
    def __init__(self):
        self.link_patterns = [
            # 标准格式
            r'微信链接\s*[:：]\s*(https?://[^\s\n]+)',
            r'原文链接\s*[:：]\s*(https?://[^\s\n]+)',
            r'链接\s*[:：]\s*(https?://[^\s\n]+)',
            r'文章链接\s*[:：]\s*(https?://[^\s\n]+)',
            r'来源链接\s*[:：]\s*(https?://[^\s\n]+)',
            
            # 英文格式
            r'Link\s*[:：]\s*(https?://[^\s\n]+)',
            r'URL\s*[:：]\s*(https?://[^\s\n]+)',
            r'Source\s*[:：]\s*(https?://[^\s\n]+)',
            r'Reference\s*[:：]\s*(https?://[^\s\n]+)',
            
            # 宽松格式
            r'详见\s*[:：]?\s*(https?://[^\s\n]+)',
            r'参考\s*[:：]?\s*(https?://[^\s\n]+)',
            r'出处\s*[:：]?\s*(https?://[^\s\n]+)',
        ]
        
        # s.caixuan.cc 特定模式
        self.caixuan_pattern = r'(https?://s\.caixuan\.cc/\w+)'
        
    def parse_content_and_links(self, content: str, use_intelligent_matching: bool = True) -> Tuple[str, Optional[str], Optional[str], Dict]:
        """
        增强版内容解析函数
        
        Args:
            content: 要解析的内容
            use_intelligent_matching: 是否启用智能匹配（可以设为False以提高速度）
        
        Returns:
            Tuple[正文文本, 微信链接, 原文链接, 解析元数据]
        """
        if not content:
            return "", None, None, {"parsing_method": "empty"}
        
        # 首先尝试标准解析
        text, wechat_link, source_link, metadata = self._standard_parsing(content)
        
        # 如果标准解析没有找到链接，并且启用了智能匹配，尝试智能匹配
        if not wechat_link and not source_link and use_intelligent_matching:
            matched_link, match_metadata = self._intelligent_link_matching(text or content)
            if matched_link:
                # 将匹配到的链接作为原文链接
                source_link = matched_link
                metadata.update(match_metadata)
                metadata["parsing_method"] = "intelligent_matching"
        
        # 如果还是没有找到，尝试URL提取
        if not wechat_link and not source_link:
            extracted_links = self._extract_all_urls(content)
            if extracted_links:
                # 优先使用s.caixuan.cc链接
                caixuan_links = [link for link in extracted_links if 's.caixuan.cc' in link]
                if caixuan_links:
                    source_link = caixuan_links[0]
                    metadata["parsing_method"] = "url_extraction_caixuan"
                else:
                    source_link = extracted_links[0]
                    metadata["parsing_method"] = "url_extraction_generic"
        
        # 清理正文内容
        if text:
            text = self._clean_text_content(text)
        
        return text, wechat_link, source_link, metadata
    
    def _standard_parsing(self, content: str) -> Tuple[str, Optional[str], Optional[str], Dict]:
        """标准解析逻辑"""
        metadata = {"parsing_method": "standard"}
        
        wechat_link = None
        source_link = None
        text_content = content.strip()
        
        # 尝试匹配微信链接
        wechat_match = re.search(r'微信链接\s*[:：]\s*(https?://[^\s\n]+)', content, re.IGNORECASE)
        if wechat_match:
            wechat_link = wechat_match.group(1).strip()
            # 从文本中移除链接部分
            text_content = content[:wechat_match.start()].strip()
            metadata["wechat_link_found"] = True
        
        # 尝试匹配原文链接
        source_match = re.search(r'原文链接\s*[:：]\s*(https?://[^\s\n]+)', content, re.IGNORECASE)
        if source_match:
            source_link = source_match.group(1).strip()
            # 更新正文内容
            if wechat_match:
                # 如果两个链接都存在，取较早出现位置之前的内容
                earliest_pos = min(wechat_match.start(), source_match.start())
                text_content = content[:earliest_pos].strip()
            else:
                text_content = content[:source_match.start()].strip()
            metadata["source_link_found"] = True
        
        # 如果没有找到特定格式的链接，尝试通用链接模式
        if not wechat_link and not source_link:
            for pattern in self.link_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    link = match.group(1).strip()
                    # 根据上下文判断是微信链接还是原文链接
                    if 'weixin' in link or 'wx' in link or '微信' in content[:match.start()]:
                        wechat_link = link
                    else:
                        source_link = link
                    
                    text_content = content[:match.start()].strip()
                    metadata["generic_link_found"] = True
                    break
        
        return text_content, wechat_link, source_link, metadata
    
    def _intelligent_link_matching(self, text: str) -> Tuple[Optional[str], Dict]:
        """基于内容智能匹配链接"""
        metadata = {"intelligent_matching_attempted": True}
        
        try:
            # 快速检查：如果文本太短，跳过智能匹配
            if len(text.strip()) < 20:
                metadata["skipped_reason"] = "text_too_short"
                return None, metadata
            
            # 使用智能引用匹配器（数据应该已经在初始化时加载了）
            if not citation_matcher.loaded:
                logger.warning("Citation matcher未加载，跳过智能匹配")
                metadata["skipped_reason"] = "matcher_not_loaded"
                return None, metadata
                
            best_match = citation_matcher.get_best_match(text, threshold=0.2)  # 降低阈值提高匹配率
            
            if best_match:
                url, source = best_match
                metadata.update({
                    "matched_url": url,
                    "matched_source": source,
                    "matching_successful": True
                })
                logger.info(f"智能匹配找到链接: {url}")
                return url, metadata
            else:
                metadata["matching_successful"] = False
                logger.debug("智能匹配未找到合适的链接")
                
        except Exception as e:
            logger.warning(f"智能链接匹配失败: {e}")
            metadata["matching_error"] = str(e)
        
        return None, metadata
    
    def _extract_all_urls(self, content: str) -> List[str]:
        """提取内容中的所有URL"""
        # 通用URL模式
        url_pattern = r'https?://[^\s\n\]）】,，。！？；;]+[^\s\n\]）】,，。！？；;，.!?]'
        urls = re.findall(url_pattern, content)
        
        # 清理URL末尾的标点符号
        cleaned_urls = []
        for url in urls:
            # 移除末尾的常见标点符号
            cleaned_url = re.sub(r'[,，。！？；;）】\]\s]+$', '', url)
            if cleaned_url and len(cleaned_url) > 10:  # 基本长度验证
                cleaned_urls.append(cleaned_url)
        
        return cleaned_urls
    
    def _clean_text_content(self, text: str) -> str:
        """清理正文内容"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除可能残留的链接标记
        text = re.sub(r'(微信|原文|来源|详见|参考|出处)?链接\s*[:：]?\s*$', '', text, flags=re.IGNORECASE)
        
        # 移除末尾的标点符号和空格
        text = text.strip(' \n\r\t.,;:！？。，；：')
        
        return text
    
    def batch_parse_documents(self, documents: List[Dict], fast_mode: bool = True) -> List[Dict]:
        """批量解析文档"""
        logger.info(f"enhanced_parser: 开始批量解析{len(documents)}个文档（快速模式: {fast_mode}）")
        enhanced_documents = []
        
        for i, doc in enumerate(documents):
            if i % 3 == 0:  # 每3个文档输出一次进度
                logger.info(f"enhanced_parser: 处理进度 {i+1}/{len(documents)}")
            
            content = doc.get('content', '').strip()
            if not content:
                enhanced_documents.append(doc)
                continue
                
            # 解析内容和链接（快速模式下禁用智能匹配）
            use_intelligent = not fast_mode
            text, wechat_link, source_link, parse_metadata = self.parse_content_and_links(content, use_intelligent_matching=use_intelligent)
            
            # 创建增强的文档副本
            enhanced_doc = doc.copy()
            
            # 更新内容
            if text:
                enhanced_doc['content'] = text
            
            # 添加链接信息到metadata
            if 'metadata' not in enhanced_doc:
                enhanced_doc['metadata'] = {}
            
            if wechat_link:
                enhanced_doc['metadata']['wechat_link'] = wechat_link
            
            if source_link:
                enhanced_doc['metadata']['source_link'] = source_link
                enhanced_doc['metadata']['link'] = source_link  # 保持兼容性
            
            # 添加解析元数据
            enhanced_doc['metadata'].update({
                'parsing_metadata': parse_metadata,
                'content_parsing_applied': True
            })
            
            enhanced_documents.append(enhanced_doc)
        
        logger.info(f"enhanced_parser: 批量解析完成，处理了{len(documents)}个文档，返回{len(enhanced_documents)}个增强文档")
        return enhanced_documents

# 全局实例
enhanced_parser = EnhancedContentParser()

# 向后兼容的函数
def parse_content_and_link(content: str) -> Tuple[str, Optional[str], Optional[str]]:
    """向后兼容的解析函数（快速模式）"""
    text, wechat_link, source_link, _ = enhanced_parser.parse_content_and_links(content, use_intelligent_matching=False)
    return text, wechat_link, source_link
