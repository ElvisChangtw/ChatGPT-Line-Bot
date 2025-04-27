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
BOT_USER_ID = os.getenv('LINE_BOT_USER_ID')


def auto_cache_text_messages(body):
    """自動快取所有文字訊息，並處理 quotedMessageId"""
    try:
        data = json.loads(body)
        for evt in data.get('events', []):
            if evt.get('type') == 'message' and evt['message'].get('type') == 'text':
                msg = evt['message']
                mid = msg['id']; txt = msg['text'].strip()
                save_message_to_cache(mid, txt)
                qid = msg.get('quotedMessageId')
                if qid:
                    quoted_cache[mid] = qid
    except Exception as e:
        logger.error(f"Auto cache text message failed: {e}")


def save_message_to_cache(message_id, text):
    """存訊息，超過上限自動移除最舊"""
    if len(message_cache) >= MAX_CACHE_SIZE:
        message_cache.popitem(last=False)
    message_cache[message_id] = text


def should_process_message(event, text):
    """判斷是否該處理"""
    src = event.source.type
    if src == 'user':
        return True
    # 群組/Room 需要 slash 指令或 mention
    if src in ['group', 'room']:
        if text.startswith('/'):
            return True
        ment = getattr(event.message, 'mention', None)
        if ment and ment.mentionees:
            return any(m.user_id == BOT_USER_ID for m in ment.mentionees)
    return False


def get_replied_message_text(event):
    mid = event.message.id
    if mid in quoted_cache:
        return message_cache.get(quoted_cache[mid])
    ref = getattr(event.message, 'reference', None)
    if ref:
        rid = getattr(ref, 'message_id', None) or getattr(ref, 'messageId', None)
        return message_cache.get(rid)
    return None


def remove_bot_mention(event, text):
    ment = getattr(event.message, 'mention', None)
    if ment and ment.mentionees:
        for m in sorted(ment.mentionees, key=lambda x: x.index, reverse=True):
            if m.user_id == BOT_USER_ID:
                text = text[:m.index] + text[m.index+m.length:]
        text = text.strip()
    return text


@app.route('/callback', methods=['POST'])
def callback():
    sig = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info('Request body: '+body)
    try:
        auto_cache_text_messages(body)
        handler.handle(body, sig)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    uid = event.source.user_id
    raw = event.message.text.strip()
    reply_txt = get_replied_message_text(event)

    # /註冊 優先處理
    if raw.startswith('/註冊'):
        key = raw[3:].strip()
        mdl = OpenAIModel(api_key=key)
        ok, _, _ = mdl.check_token_valid()
        if ok:
            model_management[uid] = mdl
            storage.save({uid:key})
            resp = 'Token 有效，註冊成功'
        else:
            resp = 'Token 無效，請重新註冊'
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=resp))
        return

    # 確認是否要處理（群組需 slash 或 mention）
    if not should_process_message(event, raw):
        return

    # 清除 mention 再加上下文
    txt = remove_bot_mention(event, raw)
    if reply_txt:
        txt = f"針對此訊息回應：{reply_txt}\n使用者補充：{txt}"

    try:
        if txt.startswith('/系統訊息'):
            memory.change_system_message(uid, txt[5:].strip())
            msg = '輸入成功'
        elif txt.startswith('/清除'):
            memory.remove(uid)
            msg = '歷史訊息清除成功'
        elif txt.startswith('/圖像'):
            p = txt[3:].strip()
            memory.append(uid,'user',p)
            ok,resp,err = model_management[uid].image_generations(p)
            if not ok: raise Exception(err)
            url = resp['data'][0]['url']
            msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            memory.append(uid,'assistant',url)
        else:
            user_mdl = model_management[uid]
            memory.append(uid,'user',txt)
            u = website.get_url_from_text(txt)
            if u:
                msg = TextSendMessage(text='處理網址...')
            else:
                ok,resp,err = user_mdl.chat_completions(memory.get(uid), os.getenv('OPENAI_MODEL_ENGINE'))
                if not ok: raise Exception(err)
                role,content = get_role_and_content(resp)
                msg = TextSendMessage(text=content)
            memory.append(uid,role,content)
    except KeyError:
        msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
    except Exception as e:
        msg = TextSendMessage(text=str(e))

    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    uid = event.source.user_id
    cont = line_bot_api.get_message_content(event.message.id)
    path = f"{uuid.uuid4()}.m4a"
    with open(path,'wb') as f:
        for c in cont.iter_content(): f.write(c)
    try:
        if uid not in model_management: raise ValueError('Invalid API token')
        ok,resp,err = model_management[uid].audio_transcriptions(path,'whisper-1')
        if not ok: raise Exception(err)
        memory.append(uid,'user',resp['text'])
        ok,resp,err = model_management[uid].chat_completions(memory.get(uid),'gpt-3.5-turbo')
        if not ok: raise Exception(err)
        role,content = get_role_and_content(resp)
        msg = TextSendMessage(text=content)
    except Exception as e:
        msg = TextSendMessage(text=str(e))
    finally:
        os.remove(path)
    line_bot_api.reply_message(event.reply_token, msg)


@app.route('/', methods=['GET'])
def home(): return 'Hello World'


if __name__=='__main__':
    if os.getenv('USE_MONGO'):
        mongodb.connect_to_database(); storage=Storage(MongoStorage(mongodb.db))
    else:
        storage=Storage(FileStorage('db.json'))
    try:
        for uid,key in storage.load().items(): model_management[uid]=OpenAIModel(api_key=key)
    except: pass
    app.run(host='0.0.0.0',port=8080)
