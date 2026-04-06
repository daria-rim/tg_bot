import telebot
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import time

load_dotenv()

OPENROUTER_BASE = "http://localhost:11434/v1"  # LOCAL_API_URL
MODEL_NAME = "gemma3:4b"  # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free

llm = ChatOpenAI(
    openai_api_key='fake_key',
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

# Словарь для хранения истории сообщений для каждого пользователя
chat_history = {}

BOT_TOKEN = os.getenv('BOT_TOKEN')

if not BOT_TOKEN:
    print("Ошибка: BOT_TOKEN не найден в переменных окружения!")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text

        # Инициализируем историю для нового пользователя
        if chat_id not in chat_history:
            chat_history[chat_id] = []

        # Добавляем вопрос пользователя в историю
        chat_history[chat_id].append(HumanMessage(content=user_message))

        # Оставляем только последние 10 сообщений (5 пар вопрос-ответ)
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        # Отправляем сообщение в LLM с историей
        print(f"User ({chat_id}): {user_message}")
        print(f"History length: {len(chat_history[chat_id])}")

        response = llm.invoke(chat_history[chat_id]).content

        # Добавляем ответ бота в историю
        chat_history[chat_id].append(AIMessage(content=response))

        # Снова ограничиваем историю последними 10 сообщениями
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        print(f"Bot: {response}")

        # Отвечаем бота ответом LLM
        bot.reply_to(message, response)

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}. Попробуйте позже."
        print(f"Error: {error_msg}")
        try:
            bot.reply_to(message, error_msg)
        except:
            pass


# Запуск бота с увеличенными таймаутами и повторными попытками
if __name__ == "__main__":
    print("Бот запускается...")
    print(f"Используется модель: {MODEL_NAME}")
    print(f"API Base: {OPENROUTER_BASE}")
    
    while True:
        try:
            # Увеличиваем таймауты: polling interval, timeout, long_polling_timeout
            bot.polling(
                non_stop=True,           # Не останавливаться при ошибках
                interval=1,               # Интервал между запросами в секундах
                timeout=60,               # Таймаут соединения
                long_polling_timeout=60   # Долгий polling таймаут
            )
        except Exception as e:
            print(f"Ошибка в polling: {e}")
            print("Переподключение через 5 секунд...")
            time.sleep(5)