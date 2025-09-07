#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow + Qwen 智能问答系统启动脚本 v2.0
增强版启动脚本，支持提示词管理功能
"""

import os
import sys
import locale
import time
import subprocess
from pathlib import Path
from datetime import datetime



def fix_encoding():
    """修复Windows编码问题"""
    print("🔧 配置系统编码...")
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSIOENCODING'] = '0'
    
    # Windows特殊处理
    if sys.platform.startswith('win'):
        import codecs
        
        # 重定向标准输出和错误流
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        if hasattr(sys.stderr, 'detach'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        
        # 尝试设置locale
        locale_variants = [
            'Chinese_China.UTF-8',
            'zh_CN.UTF-8', 
            'C.UTF-8',
            'en_US.UTF-8'
        ]
        
        for loc in locale_variants:
            try:
                locale.setlocale(locale.LC_ALL, loc)
                print(f"✅ 编码设置成功: {loc}")
                break
            except locale.Error:
                continue
        else:
            print("⚠️  警告: 无法设置UTF-8 locale，可能出现中文显示问题")

def check_dependencies():
    """检查依赖包"""
    print("📦 检查系统依赖...")
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'httpx',
        'pydantic',
        
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (缺失)")
    
    if missing_packages:
        print(f"\n🚨 发现 {len(missing_packages)} 个缺失的依赖包:")
        for pkg in missing_packages:
            print(f"   • {pkg}")
        
        install_choice = input("\n是否自动安装缺失的依赖? (y/N): ").lower()
        if install_choice in ['y', 'yes']:
            install_dependencies(missing_packages)
        else:
            print("⚠️  请手动安装依赖后重新运行: pip install -r requirements.txt")
            return False
    
    return True

def install_dependencies(packages):
    """自动安装依赖包"""
    print("\n🔄 正在安装依赖包...")
    
    for package in packages:
        try:
            print(f"   安装 {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print(f"   ✅ {package} 安装成功")
            else:
                print(f"   ❌ {package} 安装失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ {package} 安装异常: {e}")
            return False
    
    print("🎉 所有依赖包安装完成！")
    return True

def create_env_file():
    """创建环境配置文件"""
    env_content = """# RAGFlow + Qwen 智能问答系统配置文件 v2.0
# ====================================================

# RAGFlow 配置
RAGFLOW_URL=http://localhost:8000/api/v1/retrieval
RAGFLOW_API_KEY=your_ragflow_api_key_here
RAGFLOW_TIMEOUT=30
RAGFLOW_MAX_RETRIES=3

# Qwen 配置  
QWEN_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
QWEN_API_KEY=your_qwen_api_key_here
QWEN_MODEL=qwen-plus
QWEN_TIMEOUT=60
QWEN_MAX_RETRIES=3

# 数据集配置
DEFAULT_DATASET_IDS=4bf59b0263e411f0a1ed66e67c03082d
DEFAULT_DOCUMENT_IDS=

# 检索配置
DEFAULT_SIMILARITY_THRESHOLD=0.1
DEFAULT_TOP_K=10
DEFAULT_VECTOR_WEIGHT=0.8
MAX_CHUNK_LENGTH=1000

# 服务配置
HOST=0.0.0.0
PORT=8001
DEBUG=true
LOG_LEVEL=INFO

# 安全配置
CORS_ORIGINS=*
API_KEY=

# 缓存配置
ENABLE_CACHE=false
CACHE_TTL=3600

# ====================================================
# 配置说明：
# 1. 请替换 your_ragflow_api_key_here 为实际的 RAGflow API密钥
# 2. 请替换 your_qwen_api_key_here 为实际的 Qwen API密钥  
# 3. DEFAULT_DATASET_IDS 可以设置多个，用逗号分隔
# 4. 相似度阈值越高检索越严格（0.1-0.8推荐）
# 5. DEBUG=true 时会显示详细日志信息
"""
    return env_content

def check_config():
    """检查配置文件"""
    print("⚙️  检查配置文件...")
    
    current_dir = Path(__file__).parent
    env_file = current_dir / ".env"
    
    if not env_file.exists():
        print("📝 .env 文件不存在，正在创建...")
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(create_env_file())
        print("✅ 已创建 .env 配置文件")
        print("⚠️  请编辑 .env 文件，填入正确的API密钥！")
        
        # 询问是否现在编辑配置文件
        edit_choice = input("是否现在打开配置文件进行编辑? (y/N): ").lower()
        if edit_choice in ['y', 'yes']:
            try:
                if sys.platform.startswith('win'):
                    os.startfile(str(env_file))
                elif sys.platform.startswith('darwin'):  # macOS
                    subprocess.run(['open', str(env_file)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(env_file)])
                
                input("编辑完成后按 Enter 继续...")
            except Exception as e:
                print(f"无法自动打开文件: {e}")
                print(f"请手动打开并编辑: {env_file}")
                input("编辑完成后按 Enter 继续...")
    
    return env_file

def test_imports():
    """测试导入主要模块"""
    print("🧪 测试模块导入...")
    
    try:
        print("   导入 config...")
        from config import config
        print("   ✅ config 导入成功")
        
        print("   导入 main...")
        import main
        print("   ✅ main 导入成功")
        
        print("   导入 ragflow_client...")
        from ragflow_client import ragflow_client
        print("   ✅ ragflow_client 导入成功")
        
        print("   导入 qwen_client...")
        from qwen_client import qwen_client  
        print("   ✅ qwen_client 导入成功")
        
        return True
    except Exception as e:
        print(f"   ❌ 模块导入失败: {e}")
        return False

def show_startup_info():
    """显示启动信息"""
    from config import config
    
    print("\n🌟 系统信息:")
    print(f"   🏠 工作目录: {Path.cwd()}")
    print(f"   🐍 Python版本: {sys.version.split()[0]}")
    
    # 显示正确的访问地址
    if config.HOST == '0.0.0.0':
        display_host = 'localhost'
    else:
        display_host = config.HOST
    
    print(f"   🌐 Web界面: http://{display_host}:{config.PORT}")
    print(f"   📖 API文档: http://{display_host}:{config.PORT}/docs")
    print(f"   🎯 提示词管理: 点击Web界面中的'⚙️ 管理模板'按钮")
    print(f"   🔧 调试模式: {'开启' if config.DEBUG else '关闭'}")
    print(f"   📊 默认数据集: {len(config.DEFAULT_DATASET_IDS)} 个")
    
    if config.DEBUG:
        print(f"   🔑 RAGflow API: {'已配置' if config.RAGFLOW_API_KEY else '未配置'}")
        print(f"   🔑 Qwen API: {'已配置' if config.QWEN_API_KEY else '未配置'}")

def main():
    """主启动函数"""
    start_time = time.time()
    
 
    print(f"🕐 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    try:
        # 1. 修复编码
        fix_encoding()
        
        # 2. 切换到脚本目录
        current_dir = Path(__file__).parent
        os.chdir(current_dir)
        print(f"📁 工作目录: {current_dir}")
        
        # 3. 检查依赖
        if not check_dependencies():
            print("❌ 依赖检查失败")
            return 1
        
        # 4. 检查配置
        env_file = check_config()
        
        # 5. 测试模块导入
        if not test_imports():
            print("❌ 模块导入测试失败")
            return 1
        
        # 6. 显示启动信息
        show_startup_info()
        
        # 7. 启动服务
        print("\n🚀 启动Web服务...")
        print("=" * 65)
        
        import uvicorn
        from config import config
        
        # 计算启动耗时
        startup_time = time.time() - start_time
        print(f"⚡ 启动准备完成，耗时: {startup_time:.2f}s")
        print("🎉 系统已就绪，正在启动服务...")
        print("\n💡 使用技巧:")
        print("   • Ctrl+C 停止服务")
        print("   • 浏览器访问上述地址使用系统")
        print("   • 查看API文档了解详细接口")
        print("=" * 65)
        
        # 启动uvicorn服务
        uvicorn.run(
            "main:app",
            host=config.HOST,
            port=config.PORT,
            reload=config.DEBUG,
            log_level=config.LOG_LEVEL.lower(),
            access_log=config.DEBUG,
            reload_excludes=["*.log", "*.json", "__pycache__", ".git"]
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，服务已停止")
        return 0
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print(f"📍 错误类型: {type(e).__name__}")
        
        if "config" in str(e).lower():
            print("💡 建议: 检查 .env 配置文件中的API密钥设置")
        elif "port" in str(e).lower() or "address" in str(e).lower():
            print("💡 建议: 端口可能被占用，尝试修改 .env 中的 PORT 值")
        elif "module" in str(e).lower():
            print("💡 建议: 运行 pip install -r requirements.txt 安装依赖")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 