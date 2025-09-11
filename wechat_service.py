# -*- coding: utf-8 -*-

import logging
import requests
import threading
from flask import Flask, request, abort
from wechatpy import parse_message
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.client import WeChatClient

from config import config

# --- 1. 初始化与配置 ---

# 配置日志记录器，采用更详细的格式
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 WeChat 客户端，用于发送客服消息
# 这是异步回复的关键，必须配置 AppID 和 AppSecret
wechat_client = None
if not all([config.WECHAT_APPID, config.WECHAT_APPSECRET]):
    logger.error("致命错误：WECHAT_APPID 或 WECHAT_APPSECRET 未配置！客服消息功能将不可用。")
else:
    try:
        wechat_client = WeChatClient(config.WECHAT_APPID, config.WECHAT_APPSECRET, timeout=120)
        logger.info("WeChatClient 初始化成功。")
    except Exception as e:
        logger.error(f"WeChatClient 初始化失败: {e}")


# --- 2. 核心后台处理函数 ---

def handle_ragflow_request(user_openid, user_question):
    """
    在独立的后台线程中执行，负责调用 RAGflow 并通过客服消息接口回复用户。
    这个函数是异步执行的核心。
    """
    logger.info(f"[后台任务] 开始为用户 '{user_openid}' 处理问题: '{user_question}'")
    
    final_answer = ""
    try:
        # 步骤 1: 调用 RAGFlow API
        logger.info(f"[后台任务] 正在调用 RAGFlow API: {config.RAGFLOW_URL}")
        # 注意：根据你的 RAGflow API 要求调整 payload
        # 第一个脚本用 "query", 第二个用 "question"。这里我们统一使用 "query"
        # 同时传入 user_id 便于 RAGflow 进行会话管理
        payload = {"query": user_question, "user_id": user_openid} 
        
        # 设置请求头，如果需要的话
        headers = {"Content-Type": "application/json"}

        response = requests.post(config.RAGFLOW_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status() # 如果HTTP状态码不是2xx，则抛出异常
        
        response_data = response.json()
        
        # 解析 RAGflow 的响应。根据 RAGflow API 的实际返回结构进行调整
        # 假设答案在 'answer' 字段中
        final_answer = response_data.get("answer", "抱歉，知识库中暂未找到相关内容。")
        logger.info(f"[后台任务] 已从 RAGFlow 成功获取答案。")

    except requests.exceptions.Timeout:
        logger.error(f"[后台任务] 调用 RAGFlow 服务超时。")
        final_answer = "抱歉，机器人大脑处理超时，请您稍后再试一次。"
    except requests.exceptions.RequestException as e:
        logger.error(f"[后台任务] 调用 RAGFlow 服务时发生网络错误: {e}")
        final_answer = "抱歉，机器人大脑暂时连接不上，请稍后再试一次哦。"
    except Exception as e:
        logger.error(f"[后台任务] 处理过程中发生未知错误: {e}", exc_info=True)
        final_answer = "哎呀，系统出了一点小问题，工程师正在火速赶来修复！"

    # 步骤 2: 使用客服消息接口推送最终答案
    if wechat_client and final_answer:
        try:
            wechat_client.message.send_text(user_openid, final_answer)
            logger.info(f"[后台任务] 成功将答案推送给用户 '{user_openid}'。")
        except Exception as e:
            # 这里可能包括API调用频率限制、用户拒收等错误
            logger.error(f"[后台任务] 推送客服消息给用户 '{user_openid}' 时失败: {e}")
    elif not wechat_client:
        logger.error("[后台任务] WeChatClient 未初始化，无法发送客服消息。")


# --- 3. Web 服务路由 (Web Handler) ---

@app.route('/wechat', methods=['GET', 'POST'])
def wechat_handler():
    """
    处理所有来自微信服务器的请求。
    这个函数必须在5秒内完成执行。
    """
    if request.method == 'GET':
        # 验证服务器所有权
        try:
            check_signature(
                config.WECHAT_TOKEN,
                request.args.get('signature', ''),
                request.args.get('timestamp', ''),
                request.args.get('nonce', '')
            )
            logger.info("微信服务器验证成功！")
            return request.args.get('echostr', '')
        except InvalidSignatureException:
            logger.error("微信服务器验证失败！请检查 Token 是否一致。")
            abort(403)

    elif request.method == 'POST':
        # 接收用户消息，立即响应，然后将任务转交给后台线程
        try:
            msg = parse_message(request.data)
            
            if msg.type == 'text':
                # 对于文本消息，创建并启动新线程来处理耗时的 RAGflow 请求
                logger.info(f"收到来自用户 '{msg.source}' 的文本消息: '{msg.content}'。转交后台处理。")
                task = threading.Thread(
                    target=handle_ragflow_request,
                    args=(msg.source, msg.content) # msg.source 是用户的 OpenID
                )
                task.start() # 启动线程后，此函数会立即继续执行下面的 return
            
            else:
                # 对于非文本消息，也可以通过客服消息回复
                logger.info(f"收到不支持的消息类型 '{msg.type}' จากผู้ใช้ '{msg.source}'")
                if wechat_client:
                     wechat_client.message.send_text(msg.source, "我暂时只能理解文字消息哦~")
            
            # 立即回复 "success" 告诉微信服务器“我已收到”，避免超时重试
            return "success"
        
        except Exception as e:
            logger.error(f"解析消息或启动后台任务时出错: {e}", exc_info=True)
            # 即使出现解析错误，也要回复 success，防止微信不断重发消息
            return "success"


# --- 4. 程序入口 ---

if __name__ == '__main__':
    # 启动前的配置检查
    if not config.WECHAT_TOKEN:
        logger.error("致命错误：WECHAT_TOKEN 未在 config.py 文件中设置！程序无法启动。")
    elif not wechat_client:
        logger.error("致命错误：WeChatClient 未能成功初始化，请检查 AppID 和 AppSecret！程序无法启动。")
    else:
        # 为了与 RAGflow 服务区分，端口号建议设置为配置中端口号+1
        wechat_service_port = config.PORT + 1 
        
        logger.info("==============================================")
        logger.info("=== 微信 RAG Bot 服务启动 (异步客服消息模式) ===")
        logger.info("==============================================")
        logger.info(f"服务运行在: http://0.0.0.0:{wechat_service_port}")
        logger.info(f"请将以下URL配置到您的微信公众号后台：")
        logger.info(f"URL: http://<您的公网IP或域名>:{wechat_service_port}/wechat")
        logger.info(f"Token: {config.WECHAT_TOKEN}")
        logger.info("请确保服务器的IP地址已添加到公众号后台的IP白名单中！")
        
        # 在生产环境中，建议使用 Gunicorn, uWSGI, 或 Waitress 代替 Flask 自带的服务器
        # 例如: gunicorn -w 4 -b 0.0.0.0:8081 your_script_name:app
        app.run(host='0.0.0.0', port=wechat_service_port, debug=False)