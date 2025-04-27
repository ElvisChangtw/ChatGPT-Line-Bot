from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
)
import os
import uuid
import json
from collections import OrderedDict

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

# 最大快取條目數
MAX_CACHE_SIZE = 1000
# 有序字典作為快取
message_cache = OrderedDict()

# 載入環境變數
load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
storage = None
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'), memory_message_count=2)
model_management = {}
api_keys = {}
BOT_USER_ID = os.getenv('LINE_BOT_USER_ID')


def auto_cache_text_messages(body):
    """自動快取所有文字訊息"""
    try:
        data = json.loads(body)
        events = data.get('events', [])
        for event in events:
            if event.get('type') == 'message' and event['message'].get('type') == 'text':
                msg = event['message']
                message_id = msg['id']
                text = msg['text'].strip()
                save_message_to_cache(message_id, text)
    except Exception as e:
        logger.error(f"Auto cache text message failed: {e}")


def save_message_to_cache(message_id, text):
    """存訊息到快取，超過上限自動移除最舊的"""
    if len(message_cache) >= MAX_CACHE_SIZE:
        message_cache.popitem(last=False)
    message_cache[message_id] = text


def should_process_message(event, text):
    """判斷是否要處理此訊息"""
    source_type = event.source.type
    if source_type == 'user':
        return True
    if source_type in ['group', 'room']:
        if text.startswith('/'):
            return True
        mention = getattr(event.message, 'mention', None)
        if mention and mention.mentionees:
            return any(m.user_id == BOT_USER_ID for m in mention.mentionees)
    return False


def get_replied_message_text(event):
    """如果此訊息是 reply 或 quote，從快取取得原文"""
    msg = event.message
    # 新版 quote message 支援 quoted_message_id
    quoted_id = getattr(msg, 'quoted_message_id', None) or getattr(msg, 'quotedMessageId', None)
    if quoted_id:
        logger.info(f"Found quoted_message_id: {quoted_id}")
        return message_cache.get(quoted_id)
    # 舊版 reference 物件
    reference = getattr(msg, 'reference', None)
    if reference:
        replied_id = getattr(reference, 'message_id', None) or getattr(reference, 'messageId', None)
        if replied_id:
            logger.info(f"Found reference message_id: {replied_id}")
            return message_cache.get(replied_id)
    logger.info("get_replied_message_text: no quoted or reference id")
    return None


def remove_bot_mention(event, text):
    """移除訊息中針對 Bot 的 @mention，只保留其他 mention"""
    mention = getattr(event.message, 'mention', None)
    if mention and mention.mentionees:
        sorted_mentions = sorted(mention.mentionees, key=lambda x: x.index, reverse=True)
        for m in sorted_mentions:
            if m.user_id == BOT_USER_ID and hasattr(m, 'index') and hasattr(m, 'length'):
                text = text[:m.index] + text[m.index + m.length:]
        text = text.strip()
    return text


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        auto_cache_text_messages(body)
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/secret.")
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    replied = get_replied_message_text(event)
    logger.info(f"{user_id}: {text}")

    if not should_process_message(event, text):
        logger.info(f"Message ignored: {text}")
        msg = TextSendMessage(text="尚未註冊")
        line_bot_api.reply_message(event.reply_token, msg)
        return

    # 移除對 Bot 的 mention
    text = remove_bot_mention(event, text)

    if replied:
        text = f"針對此訊息回應：{replied}\n使用者補充：{text}"
        logger.info(f"Merged replied text: {text}")

    try:
        if text.startswith('/註冊'):
            api_key = text[3:].strip()
            model = OpenAIModel(api_key=api_key)
            ok, _, _ = model.check_token_valid()
            if not ok:
                raise ValueError('Invalid API token')
            model_management[user_id] = model
            storage.save({user_id: api_key})
            msg = TextSendMessage(text='Token 有效，註冊成功')

        elif text.startswith('/指令說明'):
            msg = TextSendMessage(text="指令：...")

        elif text.startswith('/系統訊息'):
            memory.change_system_message(user_id, text[5:].strip())
            msg = TextSendMessage(text='輸入成功')

        elif text.startswith('/清除'):
            memory.remove(user_id)
            msg = TextSendMessage(text='歷史訊息清除成功')

        elif text.startswith('/圖像'):
            prompt = text[3:].strip()
            memory.append(user_id, 'user', prompt)
            ok, resp, err = model_management[user_id].image_generations(prompt)
            if not ok:
                raise Exception(err)
            url = resp['data'][0]['url']
            msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            memory.append(user_id, 'assistant', url)

        else:
            user_model = model_management[user_id]
            memory.append(user_id, 'user', text)
            url = website.get_url_from_text(text)
            if url:
                # 處理網址或 YouTube
                # ...省略詳細
                msg = TextSendMessage(text=response)
            else:
                ok, resp, err = user_model.chat_completions(memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'))
                if not ok:
                    raise Exception(err)
                role, content = get_role_and_content(resp)
                msg = TextSendMessage(text=content)
            memory.append(user_id, role, content)

    except ValueError:
        msg = TextSendMessage(text='Token 無效，請重新註冊')
    except KeyError:
        msg = TextSendMessage(text='請先註冊 Token')
    except Exception as e:
        memory.remove(user_id)
        msg = TextSendMessage(text=str(e))

    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    user_id = event.source.user_id
    audio = line_bot_api.get_message_content(event.message.id)
    path = f"{uuid.uuid4()}.m4a"
    with open(path, 'wb') as fd:
        for chunk in audio.iter_content():
            fd.write(chunk)

    try:
        if user_id not in model_management:
            raise ValueError('Invalid API token')
        ok, resp, err = model_management[user_id].audio_transcriptions(path, 'whisper-1')
        if not ok:
            raise Exception(err)
        memory.append(user_id, 'user', resp['text'])
        ok, resp, err = model_management[user_id].chat_completions(memory.get(user_id), 'gpt-3.5-turbo')
        if not ok:
            raise Exception(err)
        role, content = get_role_and_content(resp)
        memory.append(user_id, role, content)
        msg = TextSendMessage(text=content)
    except Exception as e:
        msg = TextSendMessage(text=str(e))
    finally:
        os.remove(path)
    line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
    return 'Hello World'


if __name__ == '__main__':
    if os.getenv('USE_MONGO'):
        mongodb.connect_to_database()
        storage = Storage(MongoStorage(mongodb.db))
    else:
        storage = Storage(FileStorage('db.json'))
    try:
        data = storage.load()
        for uid, key in data.items():
            model_management[uid] = OpenAIModel(api_key=key)
    except FileNotFoundError:
        pass
    app.run(host='0.0.0.0', port=8080)
