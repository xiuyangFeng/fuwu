"""
智能引用匹配器 - 解决链接与内容不匹配问题

主要功能：
1. 从 cleaned_links.txt 构建内容-链接映射索引
2. 基于内容相似度匹配正确的链接
3. 提供多种匹配策略和容错机制
"""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
import jieba
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class LinkEntry:
    """链接条目数据结构"""
    id: int
    url: str
    source: str
    keywords: Set[str]
    content_hash: str

class CitationMatcher:
    """智能引用匹配器"""
    
    def __init__(self, links_file: str = "cleaned_links.txt"):
        self.links_file = Path(links_file)
        self.link_entries: List[LinkEntry] = []
        self.content_to_link: Dict[str, str] = {}
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        self.institution_index: Dict[str, List[int]] = defaultdict(list)
        self.author_index: Dict[str, List[int]] = defaultdict(list)
        self.loaded = False
        
    def load_links_data(self):
        """加载并解析 cleaned_links.txt 文件"""
        if self.loaded:
            return
            
        if not self.links_file.exists():
            logger.warning(f"链接文件 {self.links_file} 不存在，将使用基础匹配模式")
            return
            
        try:
            with open(self.links_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析文件内容
            entries = self._parse_links_file(content)
            self.link_entries = entries
            
            # 构建索引
            self._build_indexes()
            
            logger.info(f"成功加载 {len(self.link_entries)} 个链接条目并构建索引")
            self.loaded = True
            
        except Exception as e:
            logger.error(f"加载链接数据失败: {e}")
    
    def _parse_links_file(self, content: str) -> List[LinkEntry]:
        """解析 cleaned_links.txt 文件格式"""
        entries = []
        
        # 使用正则表达式匹配条目
        pattern = r'(\d+)\.\s*(https://s\.caixuan\.cc/\w+)\s*\n\s*来源:\s*(.+?)\n'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            entry_id, url, source = match
            
            # 提取关键词
            keywords = self._extract_keywords(source)
            
            # 生成内容哈希（用于快速匹配）
            content_hash = self._generate_content_hash(source)
            
            entry = LinkEntry(
                id=int(entry_id),
                url=url.strip(),
                source=source.strip(),
                keywords=keywords,
                content_hash=content_hash
            )
            entries.append(entry)
            
        return entries
    
    def _extract_keywords(self, source: str) -> Set[str]:
        """从来源描述中提取关键词"""
        keywords = set()
        
        # 提取机构名称
        institutions = self._extract_institutions(source)
        keywords.update(institutions)
        
        # 提取作者名称
        authors = self._extract_authors(source)
        keywords.update(authors)
        
        # 提取技术关键词
        tech_keywords = self._extract_tech_keywords(source)
        keywords.update(tech_keywords)
        
        # 使用jieba分词提取更多关键词
        words = jieba.cut(source)
        for word in words:
            if len(word.strip()) > 1:
                keywords.add(word.strip())
                
        return keywords
    
    def _extract_keywords_fast(self, content: str) -> Set[str]:
        """快速版关键词提取，仅提取核心关键词"""
        keywords = set()
        
        # 只提取重要的技术关键词，跳过jieba分词以节省时间
        tech_terms = [
            'CFD', 'CNN', 'DRL', 'PINN', 'MRI', 'CT', 'FSI', 'DeepONet',
            '深度学习', '强化学习', '神经网络', '机器学习', '计算流体力学',
            '血流动力学', '主动脉', '动脉瘤', '支架', '血栓', '医学图像',
            '图像分割', '医工交叉', '生物力学', '数值模拟', '北京理工大学',
            '清华大学', '上海交通大学', '浙江大学', '复旦大学'
        ]
        
        content_lower = content.lower()
        for term in tech_terms:
            if term.lower() in content_lower:
                keywords.add(term)
        
        # 快速提取大学名称
        institutions = self._extract_institutions(content)
        keywords.update(institutions)
        
        return keywords
    
    def _calculate_similarity_score_fast(self, content: str, content_keywords: Set[str], entry: LinkEntry) -> float:
        """快速版相似度计算"""
        # 简化计算：主要基于关键词匹配
        keyword_overlap = len(content_keywords & entry.keywords)
        
        if keyword_overlap == 0:
            return 0.0
        
        # 基础分数：关键词重叠率
        keyword_union = len(content_keywords | entry.keywords)
        keyword_score = keyword_overlap / keyword_union if keyword_union > 0 else 0
        
        # 机构匹配加成
        content_institutions = self._extract_institutions(content)
        entry_institutions = self._extract_institutions(entry.source)
        if content_institutions & entry_institutions:
            keyword_score *= 1.5  # 机构匹配给予加成
        
        return min(keyword_score, 1.0)  # 确保分数不超过1.0
    
    def _extract_institutions(self, text: str) -> Set[str]:
        """提取机构名称"""
        institutions = set()
        
        # 常见机构模式
        patterns = [
            r'([^\s]+大学)',
            r'([^\s]+学院)',
            r'([^\s]+研究所)',
            r'([^\s]+医院)',
            r'([^\s]+中心)',
            r'([^\s]+实验室)',
            r'([^\s]+(Corporation|Inc|Ltd|University|Institute|Hospital|Center))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            institutions.update(matches)
            
        return institutions
    
    def _extract_authors(self, text: str) -> Set[str]:
        """提取作者名称"""
        authors = set()
        
        # 提取中文姓名
        chinese_name_pattern = r'([一-龯]{2,4}(?:教授|博士|副教授|研究员)?)'
        chinese_names = re.findall(chinese_name_pattern, text)
        authors.update(chinese_names)
        
        # 提取英文姓名
        english_name_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        english_names = re.findall(english_name_pattern, text)
        authors.update(english_names)
        
        return authors
    
    def _extract_tech_keywords(self, text: str) -> Set[str]:
        """提取技术关键词"""
        keywords = set()
        
        # 预定义的技术关键词
        tech_terms = [
            'CFD', 'CNN', 'DRL', 'PINN', 'MRI', 'CT', 'FSI', 'DeepONet',
            '深度学习', '强化学习', '神经网络', '机器学习', '计算流体力学',
            '血流动力学', '主动脉', '动脉瘤', '支架', '血栓', '医学图像',
            '图像分割', '医工交叉', '生物力学', '数值模拟'
        ]
        
        for term in tech_terms:
            if term in text:
                keywords.add(term)
                
        return keywords
    
    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希（简化版本，用于快速匹配）"""
        # 移除标点符号和空格，生成简化的内容指纹
        simplified = re.sub(r'[^\w]', '', content.lower())
        return simplified[:50]  # 取前50个字符作为哈希
    
    def _build_indexes(self):
        """构建各种索引以提高匹配速度"""
        for i, entry in enumerate(self.link_entries):
            # 关键词索引
            for keyword in entry.keywords:
                self.keyword_index[keyword].append(i)
            
            # 机构索引
            institutions = self._extract_institutions(entry.source)
            for inst in institutions:
                self.institution_index[inst].append(i)
            
            # 作者索引  
            authors = self._extract_authors(entry.source)
            for author in authors:
                self.author_index[author].append(i)
    
    def find_matching_links(self, content: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        为给定内容找到最匹配的链接
        
        Returns:
            List[Tuple[url, score, source]]: 匹配结果列表，按分数降序
        """
        if not self.loaded:
            return []  # 如果没有加载，直接返回空结果避免阻塞
            
        if not self.link_entries:
            return []
        
        # 快速预筛选：基于关键词索引
        content_keywords = self._extract_keywords_fast(content)
        candidate_indices = set()
        
        # 基于关键词索引快速筛选候选项
        for keyword in content_keywords:
            if keyword in self.keyword_index:
                candidate_indices.update(self.keyword_index[keyword])
                if len(candidate_indices) > 50:  # 限制候选数量，避免过度计算
                    break
        
        # 如果没有基于关键词的候选项，回退到机构匹配
        if not candidate_indices:
            institutions = self._extract_institutions(content)
            for inst in institutions:
                if inst in self.institution_index:
                    candidate_indices.update(self.institution_index[inst])
        
        # 如果还是没有候选项，限制搜索范围到前100个条目
        if not candidate_indices:
            candidate_indices = set(range(min(100, len(self.link_entries))))
        
        scores = []
        for idx in candidate_indices:
            if idx < len(self.link_entries):
                entry = self.link_entries[idx]
                score = self._calculate_similarity_score_fast(content, content_keywords, entry)
                if score > 0.15:  # 提高阈值，减少计算
                    scores.append((entry.url, score, entry.source))
        
        # 按分数降序排序，返回前top_k个结果
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _calculate_similarity_score(self, content: str, content_keywords: Set[str], entry: LinkEntry) -> float:
        """计算内容与链接条目的相似度分数"""
        total_score = 0.0
        
        # 1. 关键词匹配分数 (权重: 0.4)
        keyword_overlap = len(content_keywords & entry.keywords)
        keyword_union = len(content_keywords | entry.keywords)
        keyword_score = keyword_overlap / keyword_union if keyword_union > 0 else 0
        total_score += keyword_score * 0.4
        
        # 2. 文本相似度分数 (权重: 0.3)
        text_similarity = SequenceMatcher(None, content.lower(), entry.source.lower()).ratio()
        total_score += text_similarity * 0.3
        
        # 3. 机构匹配分数 (权重: 0.2)
        content_institutions = self._extract_institutions(content)
        entry_institutions = self._extract_institutions(entry.source)
        inst_overlap = len(content_institutions & entry_institutions)
        inst_score = inst_overlap / max(len(content_institutions | entry_institutions), 1)
        total_score += inst_score * 0.2
        
        # 4. 作者匹配分数 (权重: 0.1)
        content_authors = self._extract_authors(content)
        entry_authors = self._extract_authors(entry.source)
        author_overlap = len(content_authors & entry_authors)
        author_score = author_overlap / max(len(content_authors | entry_authors), 1)
        total_score += author_score * 0.1
        
        return total_score
    
    def get_best_match(self, content: str, threshold: float = 0.3) -> Optional[Tuple[str, str]]:
        """
        获取最佳匹配的链接
        
        Returns:
            Optional[Tuple[url, source]]: 最佳匹配结果，如果没有达到阈值则返回None
        """
        matches = self.find_matching_links(content, top_k=1)
        if matches and matches[0][1] >= threshold:
            return matches[0][0], matches[0][2]  # (url, source)
        return None
    
    def batch_match_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        批量为文档匹配链接
        
        Args:
            documents: 文档列表，每个文档包含 'content' 字段
            
        Returns:
            增强后的文档列表，添加了匹配的链接信息
        """
        enhanced_documents = []
        
        for doc in documents:
            content = doc.get('content', '').strip()
            if not content:
                enhanced_documents.append(doc)
                continue
                
            # 尝试匹配链接
            best_match = self.get_best_match(content)
            
            enhanced_doc = doc.copy()
            if best_match:
                url, source = best_match
                # 更新或添加链接信息到metadata
                if 'metadata' not in enhanced_doc:
                    enhanced_doc['metadata'] = {}
                
                enhanced_doc['metadata']['matched_link'] = url
                enhanced_doc['metadata']['matched_source'] = source
                enhanced_doc['metadata']['link_matching_applied'] = True
                
                logger.debug(f"为文档匹配到链接: {url}")
            else:
                if 'metadata' not in enhanced_doc:
                    enhanced_doc['metadata'] = {}
                enhanced_doc['metadata']['link_matching_applied'] = False
                logger.debug("未找到匹配的链接")
                
            enhanced_documents.append(enhanced_doc)
            
        return enhanced_documents

# 全局实例
citation_matcher = CitationMatcher()
