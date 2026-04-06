import telebot
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENROUTER_API_KEY'),
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

BOT_TOKEN = os.getenv('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)

# Словарь для хранения истории сообщений для каждого пользователя
# Ключ: chat_id, значение: список сообщений (HumanMessage и AIMessage)
chat_history = {}


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

        # Отправляем всю историю в LLM
        print(f"User ({chat_id}): {user_message}")
        print(f"History length: {len(chat_history[chat_id])}")

        response = llm.invoke(chat_history[chat_id]).content

        # Добавляем ответ бота в историю
        chat_history[chat_id].append(AIMessage(content=response))

        # Снова ограничиваем историю последними 10 сообщениями
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        print(f"Bot: {response}")

        # Отправляем ответ пользователю
        bot.reply_to(message, response)

    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        error_msg = f"Ошибка: {str(e)}. Попробуйте позже."
        print(f"Error: {error_msg}")
        bot.reply_to(message, error_msg)


# Запуск бота
if __name__ == "__main__":
    print("Бот запущен и готов к работе!")
    print(f"Используется модель: {MODEL_NAME}")
    print(f"API Base: {OPENROUTER_BASE}")
    bot.polling()