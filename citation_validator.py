"""
引用验证器 - 确保链接与内容的准确匹配

主要功能：
1. 验证引用链接与内容的匹配度
2. 检测并报告可能的不匹配情况
3. 提供修正建议和替代方案
4. 支持多种验证策略和阈值设置
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from difflib import SequenceMatcher
from citation_matcher import citation_matcher
from config import config

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证严格程度"""
    STRICT = "strict"      # 严格验证，要求高匹配度
    MODERATE = "moderate"  # 适中验证，允许一定偏差
    LENIENT = "lenient"    # 宽松验证，只检查基本一致性

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]

class CitationValidator:
    """引用验证器"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.thresholds = self._get_thresholds()
        
    def _get_thresholds(self) -> Dict[str, float]:
        """根据验证级别获取阈值"""
        thresholds = {
            ValidationLevel.STRICT: {
                'content_similarity': 0.7,
                'keyword_overlap': 0.6,
                'institution_match': 0.8,
                'author_match': 0.7
            },
            ValidationLevel.MODERATE: {
                'content_similarity': 0.4,
                'keyword_overlap': 0.3,
                'institution_match': 0.5,
                'author_match': 0.4
            },
            ValidationLevel.LENIENT: {
                'content_similarity': 0.2,
                'keyword_overlap': 0.1,
                'institution_match': 0.2,
                'author_match': 0.2
            }
        }
        return thresholds[self.validation_level]
    
    def validate_citation(self, content: str, link: str, source: str = "") -> ValidationResult:
        """
        验证单个引用的准确性
        
        Args:
            content: 文档内容
            link: 引用链接
            source: 链接对应的来源描述（如果有）
            
        Returns:
            ValidationResult: 验证结果
        """
        issues = []
        suggestions = []
        metadata = {}
        
        try:
            # 1. 基本格式验证
            if not self._validate_link_format(link):
                issues.append("链接格式不正确")
            
            # 2. 如果有来源描述，验证内容匹配度
            if source:
                content_score = self._calculate_content_similarity(content, source)
                metadata['content_similarity'] = content_score
                
                if content_score < self.thresholds['content_similarity']:
                    issues.append(f"内容匹配度过低 ({content_score:.3f})")
                    suggestions.append("建议重新匹配更相关的文献")
            
            # 3. 使用智能匹配器验证
            validation_score = self._validate_with_matcher(content, link)
            metadata['matcher_validation'] = validation_score
            
            # 4. 关键词匹配验证
            if source:
                keyword_score = self._validate_keywords(content, source)
                metadata['keyword_match'] = keyword_score
                
                if keyword_score < self.thresholds['keyword_overlap']:
                    issues.append(f"关键词匹配度不足 ({keyword_score:.3f})")
            
            # 5. 机构和作者验证
            if source:
                institution_score, author_score = self._validate_entities(content, source)
                metadata['institution_match'] = institution_score
                metadata['author_match'] = author_score
                
                if institution_score < self.thresholds['institution_match']:
                    issues.append("机构信息不匹配")
                    
                if author_score < self.thresholds['author_match']:
                    issues.append("作者信息不匹配")
            
            # 6. 计算总体置信度
            confidence = self._calculate_overall_confidence(metadata)
            
            # 7. 生成建议
            if issues:
                suggestions.extend(self._generate_suggestions(content, link, issues))
            
            is_valid = len(issues) == 0 or confidence > 0.6
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=issues,
                suggestions=suggestions,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"验证过程中发生错误: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=[f"验证失败: {str(e)}"],
                suggestions=["建议手动检查引用准确性"],
                metadata={"error": str(e)}
            )
    
    def validate_citations_batch(self, citations: List[Dict]) -> List[ValidationResult]:
        """批量验证引用"""
        results = []
        
        for citation in citations:
            content = citation.get('content', '')
            link = citation.get('link', '')
            source = citation.get('source', '')
            
            if content and link:
                result = self.validate_citation(content, link, source)
                results.append(result)
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=["缺少必要的验证信息"],
                    suggestions=["确保提供内容和链接信息"],
                    metadata={}
                ))
        
        return results
    
    def _validate_link_format(self, link: str) -> bool:
        """验证链接格式"""
        if not link:
            return False
        
        # 基本URL格式验证
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, link):
            return False
        
        # 检查是否为期望的链接格式（如s.caixuan.cc）
        if 's.caixuan.cc' in link:
            return re.match(r'https://s\.caixuan\.cc/\w+', link) is not None
        
        return True
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算两个文本的相似度"""
        if not content1 or not content2:
            return 0.0
        
        # 使用SequenceMatcher计算相似度
        similarity = SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
        return similarity
    
    def _validate_with_matcher(self, content: str, link: str) -> float:
        """使用智能匹配器验证"""
        try:
            # 确保匹配器已加载
            if config.ENABLE_CITATION_MATCHER:
                citation_matcher.load_links_data()
                
                # 查找最佳匹配
                matches = citation_matcher.find_matching_links(content, top_k=5)
                
                # 检查给定链接是否在匹配结果中
                for url, score, source in matches:
                    if url == link:
                        return score
                
                # 如果没有直接匹配，返回最佳匹配的分数作为参考
                return matches[0][1] if matches else 0.0
            else:
                # citation_matcher已禁用，返回默认分数
                return 0.5
        except Exception as e:
            logger.warning(f"使用智能匹配器验证时出错: {e}")
            return 0.0
    
    def _validate_keywords(self, content: str, source: str) -> float:
        """验证关键词匹配度"""
        try:
            # 提取关键词
            content_keywords = self._extract_simple_keywords(content)
            source_keywords = self._extract_simple_keywords(source)
            
            if not content_keywords or not source_keywords:
                return 0.0
            
            # 计算重叠度
            overlap = len(content_keywords & source_keywords)
            union = len(content_keywords | source_keywords)
            
            return overlap / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _validate_entities(self, content: str, source: str) -> Tuple[float, float]:
        """验证机构和作者匹配度"""
        try:
            # 提取机构
            content_institutions = self._extract_institutions(content)
            source_institutions = self._extract_institutions(source)
            
            # 提取作者
            content_authors = self._extract_authors(content)
            source_authors = self._extract_authors(source)
            
            # 计算机构匹配度
            inst_overlap = len(content_institutions & source_institutions)
            inst_score = inst_overlap / max(len(content_institutions | source_institutions), 1)
            
            # 计算作者匹配度
            author_overlap = len(content_authors & source_authors)
            author_score = author_overlap / max(len(content_authors | source_authors), 1)
            
            return inst_score, author_score
            
        except Exception:
            return 0.0, 0.0
    
    def _extract_simple_keywords(self, text: str) -> set:
        """提取简单关键词"""
        # 提取技术术语
        keywords = set()
        
        # 常见技术关键词
        tech_terms = [
            'CFD', 'CNN', 'DRL', 'PINN', 'MRI', 'CT', 'FSI', 'DeepONet',
            '深度学习', '强化学习', '神经网络', '机器学习', '计算流体力学',
            '血流动力学', '主动脉', '动脉瘤', '支架', '血栓', '医学图像',
            '图像分割', '医工交叉', '生物力学', '数值模拟'
        ]
        
        for term in tech_terms:
            if term.lower() in text.lower():
                keywords.add(term)
        
        return keywords
    
    def _extract_institutions(self, text: str) -> set:
        """提取机构名称"""
        institutions = set()
        
        patterns = [
            r'([^\s]+大学)',
            r'([^\s]+学院)',
            r'([^\s]+研究所)',
            r'([^\s]+医院)',
            r'([^\s]+中心)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            institutions.update(matches)
        
        return institutions
    
    def _extract_authors(self, text: str) -> set:
        """提取作者名称"""
        authors = set()
        
        # 中文姓名
        chinese_pattern = r'([一-龯]{2,4}(?:教授|博士|副教授|研究员)?)'
        chinese_names = re.findall(chinese_pattern, text)
        authors.update(chinese_names)
        
        # 英文姓名
        english_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)'
        english_names = re.findall(english_pattern, text)
        authors.update(english_names)
        
        return authors
    
    def _calculate_overall_confidence(self, metadata: Dict[str, Any]) -> float:
        """计算总体置信度"""
        scores = []
        weights = []
        
        # 内容相似度（权重：0.4）
        if 'content_similarity' in metadata:
            scores.append(metadata['content_similarity'])
            weights.append(0.4)
        
        # 智能匹配分数（权重：0.3）
        if 'matcher_validation' in metadata:
            scores.append(metadata['matcher_validation'])
            weights.append(0.3)
        
        # 关键词匹配（权重：0.2）
        if 'keyword_match' in metadata:
            scores.append(metadata['keyword_match'])
            weights.append(0.2)
        
        # 实体匹配（权重：0.1）
        if 'institution_match' in metadata and 'author_match' in metadata:
            entity_score = (metadata['institution_match'] + metadata['author_match']) / 2
            scores.append(entity_score)
            weights.append(0.1)
        
        if not scores:
            return 0.0
        
        # 加权平均
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_suggestions(self, content: str, link: str, issues: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if "内容匹配度过低" in str(issues):
            # 尝试找到更好的匹配
            try:
                if config.ENABLE_CITATION_MATCHER:
                    citation_matcher.load_links_data()
                    matches = citation_matcher.find_matching_links(content, top_k=3)
                    if matches:
                        suggestions.append(f"建议使用更匹配的链接: {matches[0][0]}")
            except Exception:
                pass
            
        if "链接格式不正确" in str(issues):
            suggestions.append("检查链接格式是否正确，确保以http://或https://开头")
            
        if "关键词匹配度不足" in str(issues):
            suggestions.append("验证引用的研究领域是否与内容相关")
            
        if "机构信息不匹配" in str(issues):
            suggestions.append("确认引用来源的机构信息")
            
        if "作者信息不匹配" in str(issues):
            suggestions.append("检查作者姓名的准确性")
        
        return suggestions

# 全局验证器实例
citation_validator = CitationValidator(ValidationLevel.MODERATE)

def validate_response_citations(response_text: str, citations: List[Dict]) -> Dict[str, Any]:
    """
    验证回答中的引用准确性
    
    Args:
        response_text: AI生成的回答文本
        citations: 引用列表
        
    Returns:
        验证结果汇总
    """
    results = []
    
    for citation in citations:
        # 为验证准备内容
        citation_with_content = citation.copy()
        if 'content' not in citation_with_content:
            # 如果引用中没有内容，尝试从回答文本中提取相关部分
            citation_with_content['content'] = response_text[:500]  # 使用回答的前500字符
        
        result = citation_validator.validate_citation(
            content=citation_with_content.get('content', ''),
            link=citation_with_content.get('link', ''),
            source=citation_with_content.get('source', '')
        )
        results.append(result)
    
    # 汇总结果
    valid_count = sum(1 for r in results if r.is_valid)
    average_confidence = sum(r.confidence for r in results) / len(results) if results else 0
    
    all_issues = []
    all_suggestions = []
    
    for result in results:
        all_issues.extend(result.issues)
        all_suggestions.extend(result.suggestions)
    
    return {
        'total_citations': len(citations),
        'valid_citations': valid_count,
        'average_confidence': average_confidence,
        'validation_results': results,
        'summary_issues': list(set(all_issues)),
        'summary_suggestions': list(set(all_suggestions))
    }
