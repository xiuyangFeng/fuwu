import logging
import requests
from flask import Flask, request, abort, make_response
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from config import config

# 配置日志
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 定义您的RAGFlow FastAPI服务的地址
RAGFLOW_QA_URL = f"http://{config.HOST}:{config.PORT}/qa"

@app.route('/wechat', methods=['GET', 'POST'])
def wechat():
    """
    处理来自微信服务器的请求
    """
    if request.method == 'GET':
        # --- 处理微信服务器的验证请求 ---
        try:
            signature = request.args.get('signature', '')
            timestamp = request.args.get('timestamp', '')
            nonce = request.args.get('nonce', '')
            echostr = request.args.get('echostr', '')
            
            # 使用wechatpy的工具函数验证签名
            check_signature(config.WECHAT_TOKEN, signature, timestamp, nonce)
            
            logger.info("微信服务器验证成功！")
            return echostr
        except InvalidSignatureException:
            logger.error("微信服务器验证失败！")
            abort(403)
        except Exception as e:
            logger.error(f"处理GET请求时发生未知错误: {e}")
            abort(500)

    elif request.method == 'POST':
        # --- 处理用户发送的消息 ---
        try:
            # 解析微信服务器发送过来的XML消息
            msg = parse_message(request.data)

            if msg.type == 'text':
                # 如果是文本消息
                user_question = msg.content
                logger.info(f"收到来自用户 '{msg.source}' 的问题: {user_question}")

                # --- 调用RAGFlow后端服务 ---
                try:
                    # 构造请求体
                    payload = {
                        "question": user_question,
                        # 您可以在这里传递其他参数，例如指定数据集ID
                        # "dataset_ids": config.DEFAULT_DATASET_IDS 
                    }
                    
                    logger.info(f"正在调用RAGFlow QA接口: {RAGFLOW_QA_URL}")
                    response = requests.post(RAGFLOW_QA_URL, json=payload, timeout=120)
                    response.raise_for_status()  # 如果请求失败（非2xx状态码），则抛出异常

                    response_data = response.json()
                    ai_answer = response_data.get("answer", "抱歉，我暂时无法回答这个问题。")
                    logger.info(f"从RAGFlow获取到答案，长度: {len(ai_answer)}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"调用RAGFlow服务失败: {e}")
                    ai_answer = "抱歉，我的大脑暂时短路了，请稍后再试。"
                
                # 使用wechatpy创建回复消息
                reply = create_reply(ai_answer, msg)
                
                # 将回复消息序列化为XML
                response_xml = reply.render()
                
                # 返回XML响应给微信服务器
                resp = make_response(response_xml)
                resp.headers['Content-Type'] = 'application/xml'
                return resp

            else:
                # 对于非文本消息，可以回复一个提示
                logger.info(f"收到不支持的消息类型: {msg.type}")
                reply = create_reply('我暂时只能理解文字消息哦~', msg)
                response_xml = reply.render()
                resp = make_response(response_xml)
                resp.headers['Content-Type'] = 'application/xml'
                return resp

        except Exception as e:
            logger.error(f"处理POST请求时发生未知错误: {e}", exc_info=True)
            # 返回一个空的成功响应，避免微信服务器重试
            return ""

if __name__ == '__main__':
    # 注意：请确保这个端口与您的FastAPI服务端口不同
    wechat_service_port = config.PORT + 1 
    logger.info(f"启动微信公众号服务...")
    logger.info(f"请将以下URL配置到您的微信公众号后台：")
    logger.info(f"URL: http://<您的公网IP或域名>:{wechat_service_port}/wechat")
    logger.info(f"Token: {config.WECHAT_TOKEN}")
    
    # 使用Flask自带的开发服务器
    app.run(host='0.0.0.0', port=wechat_service_port, debug=False)