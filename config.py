import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from datetime import datetime

class Config:
    """增强版应用配置类 - 支持动态重新加载"""
    
    _instance = None
    _initialized = False
    _last_reload_time = None
    
    def __new__(cls):
        """单例模式，确保配置的一致性"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Config._initialized:
            self._reload_config()
            Config._initialized = True
    
    def _reload_config(self):
        """重新加载配置 - 支持动态更新"""
        # 重新加载环境变量
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)  # 使用override=True强制重新加载
        
        # RAGflow配置
        self.RAGFLOW_URL = os.getenv("RAGFLOW_URL", "http://localhost:8000/api/v1/retrieval")
        self.RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
        self.RAGFLOW_TIMEOUT = int(os.getenv("RAGFLOW_TIMEOUT", "30"))
        self.RAGFLOW_MAX_RETRIES = int(os.getenv("RAGFLOW_MAX_RETRIES", "3"))
        
        # 默认数据集和文档ID配置 - 支持动态更新
        default_dataset = "b7f46d027f6311f0bdffda8800efa43b"  # 默认数据集ID
        self.DEFAULT_DATASET_IDS = [id.strip() for id in os.getenv("DEFAULT_DATASET_IDS", default_dataset).split(",") if id.strip()]
        self.DEFAULT_DOCUMENT_IDS = [id.strip() for id in os.getenv("DEFAULT_DOCUMENT_IDS", "").split(",") if id.strip()]
        
        # Qwen配置
        self.QWEN_URL = os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.QWEN_API_KEY = os.getenv("QWEN_API_KEY")
        self.QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
        self.QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "120"))  # 增加普通请求超时到120秒
        self.QWEN_STREAM_TIMEOUT = int(os.getenv("QWEN_STREAM_TIMEOUT", "300"))  # 流式输出专用超时300秒
        self.QWEN_MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "3"))
        
        # 检索配置 - 优化检索参数以提高语义问题的检索效果
        self.DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.05"))  # 降低阈值
        self.DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "15"))  # 增加检索数量
        self.DEFAULT_VECTOR_WEIGHT = float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.7"))  # 提高向量权重
        self.MAX_CHUNK_LENGTH = int(os.getenv("MAX_CHUNK_LENGTH", "1200"))  # 增加上下文长度
        
        # 智能检索配置 - 为不同问题类型使用不同的检索策略
        self.ADAPTIVE_SIMILARITY_THRESHOLD = float(os.getenv("ADAPTIVE_SIMILARITY_THRESHOLD", "0.02"))
        self.SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "20"))  # 语义问题使用更多候选
        
        # 应用配置
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8001"))  # 使用不同端口避免冲突
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # 安全配置
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
        self.API_KEY = os.getenv("API_KEY")  # 可选的API密钥保护
        
        # 缓存配置
        self.ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false").lower() == "true"
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 缓存时间（秒）
        
        # 记录重新加载时间
        Config._last_reload_time = datetime.now()
        
        # 验证必需的环境变量
        self._validate_config()
        self._log_config()
        
    def reload(self):
        """手动重新加载配置"""
        self._reload_config()
        return True
    
    def get_dataset_config(self):
        """获取最新的数据集配置"""
        # 每次获取数据集配置时都检查是否需要重新加载
        return {
            "DEFAULT_DATASET_IDS": self.DEFAULT_DATASET_IDS,
            "DEFAULT_DOCUMENT_IDS": self.DEFAULT_DOCUMENT_IDS,
            "last_reload_time": Config._last_reload_time.isoformat() if Config._last_reload_time else None
        }
    
    def update_dataset_ids(self, new_dataset_ids):
        """动态更新数据集ID"""
        if isinstance(new_dataset_ids, str):
            new_dataset_ids = [id.strip() for id in new_dataset_ids.split(",") if id.strip()]
        
        self.DEFAULT_DATASET_IDS = new_dataset_ids
        
        # 同时更新环境变量（可选，用于持久化）
        os.environ["DEFAULT_DATASET_IDS"] = ",".join(new_dataset_ids)
        
        return True
    
    def _validate_config(self):
        """验证必需的配置项"""
        missing_configs = []
        warnings = []
        errors = []
        
        # 检查必需的API密钥
        if not self.RAGFLOW_API_KEY:
            missing_configs.append("RAGFLOW_API_KEY")
        
        if not self.QWEN_API_KEY:
            missing_configs.append("QWEN_API_KEY")
        
        # 检查数据集配置
        if not self.DEFAULT_DATASET_IDS:
            warnings.append("DEFAULT_DATASET_IDS未配置，可能无法检索到文档")
        else:
            # 验证数据集ID格式
            for dataset_id in self.DEFAULT_DATASET_IDS:
                if not dataset_id or len(dataset_id) < 10:
                    warnings.append(f"数据集ID格式可能有误: {dataset_id}")
        
        # 检查URL配置
        if not self.RAGFLOW_URL.startswith(('http://', 'https://')):
            errors.append(f"RAGFlow URL格式无效: {self.RAGFLOW_URL}")
        
        if not self.QWEN_URL.startswith(('http://', 'https://')):
            errors.append(f"Qwen URL格式无效: {self.QWEN_URL}")
        
        # 检查数值配置
        if not (0.0 <= self.DEFAULT_SIMILARITY_THRESHOLD <= 1.0):
            errors.append(f"相似度阈值超出范围 [0-1]: {self.DEFAULT_SIMILARITY_THRESHOLD}")
        
        if self.DEFAULT_TOP_K <= 0 or self.DEFAULT_TOP_K > 100:
            errors.append(f"TopK值超出合理范围 [1-100]: {self.DEFAULT_TOP_K}")
        
        if not (0.0 <= self.DEFAULT_VECTOR_WEIGHT <= 1.0):
            errors.append(f"向量权重超出范围 [0-1]: {self.DEFAULT_VECTOR_WEIGHT}")
        
        # 检查端口配置
        if not (1 <= self.PORT <= 65535):
            errors.append(f"端口号超出有效范围 [1-65535]: {self.PORT}")
        
        # 处理错误
        if errors:
            error_msg = f"配置错误:\n" + "\n".join([f"  • {error}" for error in errors])
            if self.DEBUG:
                print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # 处理缺失的必需配置
        if missing_configs:
            error_msg = f"缺少必需的环境变量: {', '.join(missing_configs)}\n"
            error_msg += "请检查 .env 文件或环境变量配置。"
            if self.DEBUG:
                print(f"⚠️ 警告: {error_msg}")
            else:
                raise ValueError(error_msg)
        
        # 显示警告
        if warnings and self.DEBUG:
            for warning in warnings:
                print(f"⚠️ 配置警告: {warning}")
                
        return len(errors) == 0 and len(missing_configs) == 0
    
    def validate_dataset_id(self, dataset_id):
        """验证单个数据集ID的有效性"""
        if not dataset_id or not isinstance(dataset_id, str):
            return False, "数据集ID不能为空"
        
        dataset_id = dataset_id.strip()
        if len(dataset_id) < 10:
            return False, "数据集ID长度太短，可能无效"
        
        if len(dataset_id) > 100:
            return False, "数据集ID长度超出限制"
        
        # 检查是否包含无效字符
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_id):
            return False, "数据集ID包含无效字符，只能包含字母、数字、下划线和连字符"
        
        return True, "数据集ID格式正确"
    
    def _log_config(self):
        """记录配置信息（敏感信息脱敏）"""
        if self.DEBUG:
            reload_time_str = Config._last_reload_time.strftime('%Y-%m-%d %H:%M:%S') if Config._last_reload_time else "未知"
            print("=== 配置信息 ===")
            print(f"配置加载时间: {reload_time_str}")
            print(f"RAGFlow URL: {self.RAGFLOW_URL}")
            print(f"RAGFlow API Key: {'已配置' if self.RAGFLOW_API_KEY else '未配置'}")
            print(f"默认数据集ID: {self.DEFAULT_DATASET_IDS} (共{len(self.DEFAULT_DATASET_IDS)}个)")
            print(f"Qwen模型: {self.QWEN_MODEL}")
            print(f"Qwen API Key: {'已配置' if self.QWEN_API_KEY else '未配置'}")
            print(f"服务地址: {self.HOST}:{self.PORT}")
            print(f"调试模式: {self.DEBUG}")
            print("================")

# 创建全局配置实例
config = Config() 