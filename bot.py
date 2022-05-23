import telebot

from utils import get_image, preprocess_img
from model import ModelInference
from config import TOKEN, MODEL_PATH, STORAGE_DIR, CLASSES

DEFAULT_MSG = "Please, send me a photo with a butterfly and I'll try to determine its species"

bot = telebot.TeleBot(TOKEN)
model = ModelInference(MODEL_PATH)


@bot.message_handler(content_types=['text', 'photo'])
def handle_message(message: telebot.types.Message):
    if message.content_type != 'photo':
        bot.send_message(message.chat.id, DEFAULT_MSG)
        return
    file_id = message.photo[-1].file_id
    file_url = bot.get_file_url(file_id)

    img = get_image(file_url, STORAGE_DIR + f'{message.chat.id}_{message.id}.jpg')
    img = preprocess_img(img)
    probs = model(img[None], proba=True)
    argmax = probs.argmax()
    max_prob = probs[argmax]
    pred_cls = CLASSES[argmax]

    bot.reply_to(message, f"I think it's <b>{pred_cls}</b>. Confidence: {max_prob:.2%}", parse_mode='html')


if __name__ == '__main__':
    bot.infinity_polling()
