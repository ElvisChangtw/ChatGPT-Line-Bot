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
# 有序字典作為快取訊息內容
message_cache = OrderedDict()
# 快取引用關係 quotedMessageId -> 原訊息ID
quoted_cache = {}

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
    """自動快取所有文字訊息，並處理 quotedMessageId"""
    try:
        data = json.loads(body)
        for event in data.get('events', []):
            if event.get('type') == 'message' and event['message'].get('type') == 'text':
                msg = event['message']
                msg_id = msg['id']
                txt = msg['text'].strip()
                save_message_to_cache(msg_id, txt)
                # 若有 quotedMessageId，記錄引用對應
                qid = msg.get('quotedMessageId')
                if qid:
                    quoted_cache[msg_id] = qid
    except Exception as e:
        logger.error(f"Auto cache text message failed: {e}")


def save_message_to_cache(message_id, text):
    """存訊息到快取，超過上限自動移除最舊的"""
    if len(message_cache) >= MAX_CACHE_SIZE:
        message_cache.popitem(last=False)
    message_cache[message_id] = text


def should_process_message(event, text):
    """判斷是否要處理此訊息"""
    src = event.source.type
    if src == 'user':
        return True
    if src in ['group', 'room']:
        if text.startswith('/'):
            return True
        mention = getattr(event.message, 'mention', None)
        if mention and mention.mentionees:
            return any(m.user_id == BOT_USER_ID for m in mention.mentionees)
    return False


def get_replied_message_text(event):
    """如果是回覆/引用其他訊息，從快取取得原文"""
    mid = event.message.id
    # 先檢查 quoted_cache
    if mid in quoted_cache:
        orig_id = quoted_cache[mid]
        logger.info(f"Quoted cache lookup: {orig_id}")
        return message_cache.get(orig_id)
    # 再檢查 reference（舊版）
    ref = getattr(event.message, 'reference', None)
    if ref:
        rid = getattr(ref, 'message_id', None) or getattr(ref, 'messageId', None)
        if rid:
            logger.info(f"Reference lookup: {rid}")
            return message_cache.get(rid)
    return None


def remove_bot_mention(event, text):
    """移除訊息中的 @Bot 自己 mention，保留其他mention"""
    mention = getattr(event.message, 'mention', None)
    if mention and mention.mentionees:
        # 反向刪除以避免 index 位移
        for m in sorted(mention.mentionees, key=lambda x: x.index, reverse=True):
            if m.user_id == BOT_USER_ID and hasattr(m, 'index') and hasattr(m, 'length'):
                text = text[:m.index] + text[m.index + m.length:]
        text = text.strip()
    return text


@app.route('/callback', methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info('Request body: ' + body)
    try:
        auto_cache_text_messages(body)
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, 'Invalid signature.')
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    uid = event.source.user_id
    text = event.message.text.strip()
    reply_txt = get_replied_message_text(event)
    logger.info(f"{uid}: {text}")

    if not should_process_message(event, text):
        logger.info(f"Ignored: {text}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='尚未註冊'))
        return

    # 移除對bot的mention
    text = remove_bot_mention(event, text)
    if reply_txt:
        text = f"針對此訊息回應：{reply_txt}\n使用者補充：{text}"

    try:
        if text.startswith('/註冊'):
            key = text[3:].strip()
            mdl = OpenAIModel(api_key=key)
            ok, _, _ = mdl.check_token_valid()
            if not ok:
                raise ValueError('Invalid API token')
            model_management[uid] = mdl
            storage.save({uid: key})
            msg = TextSendMessage(text='Token 有效，註冊成功')
        elif text.startswith('/系統訊息'):
            memory.change_system_message(uid, text[5:].strip())
            msg = TextSendMessage(text='輸入成功')
        elif text.startswith('/清除'):
            memory.remove(uid)
            msg = TextSendMessage(text='歷史訊息清除成功')
        elif text.startswith('/圖像'):
            prm = text[3:].strip()
            memory.append(uid, 'user', prm)
            ok, resp, err = model_management[uid].image_generations(prm)
            if not ok:
                raise Exception(err)
            url = resp['data'][0]['url']
            msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            memory.append(uid, 'assistant', url)
        else:
            usr_mdl = model_management[uid]
            memory.append(uid, 'user', text)
            url = website.get_url_from_text(text)
            if url:
                # 處理網址
                msg = TextSendMessage(text='處理網址...')
            else:
                ok, resp, err = usr_mdl.chat_completions(memory.get(uid), os.getenv('OPENAI_MODEL_ENGINE'))
                if not ok:
                    raise Exception(err)
                role, content = get_role_and_content(resp)
                msg = TextSendMessage(text=content)
            memory.append(uid, role, content)
    except Exception as e:
        msg = TextSendMessage(text=str(e))
    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    uid = event.source.user_id
    audio = line_bot_api.get_message_content(event.message.id)
    path = f"{uuid.uuid4()}.m4a"
    with open(path, 'wb') as f:
        for chunk in audio.iter_content():
            f.write(chunk)
    try:
        if uid not in model_management:
            raise ValueError('Invalid API token')
        ok, resp, err = model_management[uid].audio_transcriptions(path, 'whisper-1')
        if not ok:
            raise Exception(err)
        memory.append(uid, 'user', resp['text'])
        ok, resp, err = model_management[uid].chat_completions(memory.get(uid), 'gpt-3.5-turbo')
        if not ok:
            raise Exception(err)
        role, content = get_role_and_content(resp)
        msg = TextSendMessage(text=content)
    except Exception as e:
        msg = TextSendMessage(text=str(e))
    finally:
        os.remove(path)
    line_bot_api.reply_message(event.reply_token, msg)


@app.route('/', methods=['GET'])
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
