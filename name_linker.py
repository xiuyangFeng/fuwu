"""
人物姓名与个人主页链接匹配器

主要功能：
1. 加载 name_output.txt 文件中的人物姓名和主页链接数据
2. 在文本中智能识别和匹配人物姓名
3. 为匹配到的人物提供主页链接
4. 支持中英文姓名的智能匹配和去重
"""

import logging
import re
from typing import Dict, List, Set, Optional
from pathlib import Path
from dataclasses import dataclass
import jieba

logger = logging.getLogger(__name__)

@dataclass(frozen=True, eq=True)
class PersonInfo:
    """人物信息数据结构"""
    id: int
    name: str
    homepage_url: Optional[str]
    name_variations: frozenset  # 使用frozenset以支持hashable

class NameLinker:
    """
    人物姓名与个人主页链接匹配器
    支持高效的姓名识别和链接匹配
    """
    
    def __init__(self, file_path: str = "name_output.txt"):
        self.file_path = Path(file_path)
        self.persons: List[PersonInfo] = []
        self.name_to_person: Dict[str, PersonInfo] = {}
        self.name_variations: Dict[str, PersonInfo] = {}  # 包含姓名变体的映射
        self.loaded = False
        
        # 常见的学术头衔和职位，用于清理姓名
        self.academic_titles = {
            '教授', '副教授', '助教授', '讲师', '研究员', '副研究员', '助理研究员',
            '博士', '硕士', '院士', '主任', '副主任', '所长', '副所长', '主任医师',
            '副主任医师', '主治医师', '住院医师', '教授', 'Prof', 'Dr', 'PhD'
        }
    
    def load_name_data(self, file_path: Optional[str] = None) -> bool:
        """
        从指定文件加载姓名和主页URL数据
        
        Args:
            file_path: 可选的文件路径，如果不提供则使用初始化时的路径
            
        Returns:
            bool: 加载是否成功
        """
        if self.loaded:
            return True
            
        target_file = Path(file_path) if file_path else self.file_path
        if not target_file.exists():
            logger.warning(f"人物姓名数据文件 {target_file} 不存在，人物主页链接功能将不可用")
            return False
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件内容
            self.persons = self._parse_name_file(content)
            
            # 构建快速查询索引
            self._build_name_indexes()
            
            self.loaded = True
            logger.info(f"成功加载 {len(self.persons)} 个人物数据，其中 {len([p for p in self.persons if p.homepage_url])} 个有主页链接")
            return True
            
        except Exception as e:
            logger.error(f"加载人物姓名数据失败: {e}")
            return False
    
    def _parse_name_file(self, content: str) -> List[PersonInfo]:
        """
        解析 name_output.txt 文件格式
        
        预期格式：
        编号.姓名
        个人主页地址：URL
        (空行)
        """
        persons = []
        
        # 使用更精确的正则表达式匹配
        pattern = r'(\d+)\.(.+?)\n个人主页地址：(.*?)\n'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            entry_id, raw_name, raw_url = match
            
            # 清理姓名
            name = self._clean_name(raw_name.strip())
            
            # 清理URL
            url = raw_url.strip() if raw_url.strip() else None
            
            # 如果URL为空或者无效，设置为None
            if url and not self._is_valid_url(url):
                url = None
            
            # 生成姓名变体
            name_variations = self._generate_name_variations(name)
            
            person = PersonInfo(
                id=int(entry_id),
                name=name,
                homepage_url=url,
                name_variations=frozenset(name_variations)
            )
            
            persons.append(person)
        
        return persons
    
    def _clean_name(self, name: str) -> str:
        """清理姓名，移除头衔和多余的空格"""
        cleaned = name.strip()
        
        # 移除常见的学术头衔
        for title in self.academic_titles:
            if cleaned.endswith(title):
                cleaned = cleaned[:-len(title)].strip()
            elif cleaned.startswith(title):
                cleaned = cleaned[len(title):].strip()
        
        # 移除多余的空格和特殊字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip('.,，。')
        
        return cleaned
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        if not url:
            return False
        return url.startswith(('http://', 'https://')) and len(url) > 10
    
    def _generate_name_variations(self, name: str) -> Set[str]:
        """生成姓名的各种变体形式，提高匹配准确率"""
        variations = {name}
        
        # 添加去除空格的版本
        variations.add(name.replace(' ', ''))
        
        # 如果包含空格，添加各种空格变体
        if ' ' in name:
            # 中文姓名的英文变体处理
            parts = name.split()
            if len(parts) == 2:
                # 姓名倒置（如：Zhang Wei -> Wei Zhang）
                variations.add(f"{parts[1]} {parts[0]}")
                # 姓名连接（如：Zhang Wei -> ZhangWei）
                variations.add(f"{parts[0]}{parts[1]}")
        
        # 处理中文姓名的常见变体
        if self._is_chinese_name(name):
            # 添加可能的英文拼音变体（这里简化处理）
            variations.add(name.replace('·', '').replace('•', ''))
        
        return variations
    
    def _is_chinese_name(self, name: str) -> bool:
        """判断是否为中文姓名"""
        chinese_chars = sum(1 for char in name if '\u4e00' <= char <= '\u9fff')
        return chinese_chars > 0
    
    def _build_name_indexes(self):
        """构建姓名查询索引"""
        self.name_to_person.clear()
        self.name_variations.clear()
        
        for person in self.persons:
            # 主要姓名映射
            self.name_to_person[person.name] = person
            
            # 所有变体映射
            for variation in person.name_variations:
                self.name_variations[variation] = person
        
        logger.debug(f"构建了 {len(self.name_to_person)} 个主要姓名和 {len(self.name_variations)} 个姓名变体的索引")
    
    def find_mentioned_names(self, text: str, include_all: bool = False) -> Dict[str, str]:
        """
        在给定文本中查找所有提到的人物姓名，并返回其主页链接
        
        Args:
            text: 要搜索的文本
            include_all: 是否包含没有主页链接的人物
        
        Returns:
            Dict[str, str]: 一个字典，键是找到的姓名，值是对应的主页URL（如果有的话）
        """
        if not self.loaded or not text:
            return {}
        
        found_names = {}
        text_lower = text.lower()
        
        # 使用优化的匹配策略：先精确匹配，再模糊匹配
        matched_persons = set()
        
        # 1. 精确匹配所有姓名变体
        for name_variation, person in self.name_variations.items():
            if self._is_name_mentioned(text, name_variation):
                matched_persons.add(person)
        
        # 2. 构建结果字典
        for person in matched_persons:
            # 如果不包含没有链接的人物，且该人物没有链接，则跳过
            if not include_all and not person.homepage_url:
                continue
                
            # 使用原始姓名作为键
            found_names[person.name] = person.homepage_url
        
        logger.debug(f"在文本中找到 {len(found_names)} 个有链接的人物姓名")
        return found_names
    
    def _is_name_mentioned(self, text: str, name: str) -> bool:
        """
        判断姓名是否在文本中被提及
        使用智能匹配策略，避免误匹配
        """
        if not name or not text:
            return False
        
        # 简化匹配策略：对于中文姓名，直接使用字符串包含匹配
        # 这样更适合中文文本的特点
        try:
            # 对于长度小于等于2的姓名，使用更严格的匹配
            if len(name) <= 2:
                # 确保姓名前后不是中文字符（避免误匹配）
                pattern = r'(?<!\u4e00-\u9fff)' + re.escape(name) + r'(?!\u4e00-\u9fff)'
                return bool(re.search(pattern, text))
            else:
                # 对于较长的姓名（包括英文姓名），直接使用子字符串匹配
                return name in text
                
        except re.error:
            # 如果正则表达式有问题，回退到简单的子字符串匹配
            return name in text
    
    def get_person_info(self, name: str) -> Optional[PersonInfo]:
        """
        根据姓名获取人物信息
        
        Args:
            name: 人物姓名
            
        Returns:
            Optional[PersonInfo]: 人物信息，如果未找到则返回None
        """
        if not self.loaded:
            return None
        
        # 首先在主要姓名中查找
        if name in self.name_to_person:
            return self.name_to_person[name]
        
        # 然后在变体中查找
        if name in self.name_variations:
            return self.name_variations[name]
        
        return None
    
    def format_person_links_markdown(self, mentioned_names: Dict[str, str]) -> str:
        """
        将找到的人物姓名和链接格式化为美观的Markdown格式
        
        Args:
            mentioned_names: find_mentioned_names返回的结果
            
        Returns:
            str: 格式化后的Markdown字符串
        """
        if not mentioned_names:
            return ""
        
        # 过滤出有有效链接的人物
        valid_links = {name: url for name, url in mentioned_names.items() if url and self._is_valid_url(url)}
        
        if not valid_links:
            return ""
        
        # 统一样式：无论单个还是多个，都使用一致的格式，并在链接中嵌入人物姓名信息
        if len(valid_links) == 1:
            # 单个人物链接：使用与多个链接一致的格式
            name, url = next(iter(valid_links.items()))
            markdown = f"\n\n---\n\n👤 **相关人物主页** (1个)\n\n"
            # 在链接文本中明确包含姓名信息，便于前端提取
            markdown += f"1️⃣ **{name}** - [点击访问{name}的个人主页]({url})\n"
        else:
            # 多个人物链接：列表式显示
            markdown = f"\n\n---\n\n👥 **相关人物主页** ({len(valid_links)}个)\n\n"
            for i, (name, url) in enumerate(valid_links.items(), 1):
                # 确保每个链接都使用完全相同的格式，在链接文本中包含姓名
                markdown += f"{i}️⃣ **{name}** - [点击访问{name}的个人主页]({url})\n"
        
        return markdown
    
    def format_person_links_html_enhanced(self, mentioned_names: Dict[str, str]) -> str:
        """
        将找到的人物姓名和链接格式化为美观的HTML格式（用于前端特殊处理）
        
        Args:
            mentioned_names: find_mentioned_names返回的结果
            
        Returns:
            str: 格式化后的HTML字符串
        """
        if not mentioned_names:
            return ""
        
        # 过滤出有有效链接的人物
        valid_links = {name: url for name, url in mentioned_names.items() if url and self._is_valid_url(url)}
        
        if not valid_links:
            return ""
        
        html = """
<div class="person-links-container">
    <div class="person-links-header">
        <span class="person-links-icon">👤</span>
        <span class="person-links-title">相关人物主页</span>
        <span class="person-links-count">{count}个</span>
    </div>
    <div class="person-links-list">
""".format(count=len(valid_links))
        
        for name, url in valid_links.items():
            # 提取域名用于显示
            domain = self._extract_domain(url)
            html += f'''
        <div class="person-link-item">
            <div class="person-link-info">
                <span class="person-name">{name}</span>
                <span class="person-domain">{domain}</span>
            </div>
            <a href="{url}" target="_blank" rel="noopener noreferrer" class="person-link-btn">
                <span class="person-link-icon">🔗</span>
                访问主页
            </a>
        </div>
'''
        
        html += """
    </div>
</div>"""
        
        return html
    
    def _extract_domain(self, url: str) -> str:
        """从URL中提取域名"""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            # 移除www前缀
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "未知域名"
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        if not self.loaded:
            return {"loaded": False}
        
        total_persons = len(self.persons)
        persons_with_links = len([p for p in self.persons if p.homepage_url])
        
        return {
            "loaded": True,
            "total_persons": total_persons,
            "persons_with_links": persons_with_links,
            "link_coverage": persons_with_links / total_persons if total_persons > 0 else 0,
            "total_name_variations": len(self.name_variations)
        }

# 创建全局实例
name_linker = NameLinker()
