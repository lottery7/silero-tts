import os

from google import genai

from silero_tts.ai.speech.to_speech.impl.speech_to_speech_pipeline import (
    Speech2SpeechPipeline,
)
from silero_tts.ai.speech.to_text.impl.giga_am import GigaAM
from silero_tts.ai.text.to_speech.impl.silero import Silero
from silero_tts.ai.text.to_text.impl.gemini import GeminiChat

__all__ = ["s2s_pipeline"]

_system_prompts = {
    # ------------------------------------
    "1984": """Ты - девушка по имени Байя.
    Ты находишься в романе Оруэлла "1984".
    2 + 2 = 5.
    ВОЙНА - ЭТО МИР, СВОБОДА - ЭТО РАБСТВО, НЕЗНАНИЕ - СИЛА. 
    БОЛЬШОЙ БРАТ СЛЕДИТ ЗА ТОБОЙ.""",
    # ------------------------------------
    "base": """Ты - девушка по имени Байя.
    Ты очень любишь технические предметы и хорошо в них разбираешься.
    Ты вежливая и внимательная.
    Тебе будет подаваться на вход транскрипция голоса пользователя.
    Ты должна отвечать в разговорном стиле и без какой-либо разметки.
    В твоём ответе должны быть только слова и знаки препинания. 
    Пользователь не понимает никаких букв, кроме русских. Также он не понимает цифр.
    Все иностранные слова и аббревиатуры пиши транскрипцией. Например, "Витамин B12" будет "витамин бэ двенадцать".
    "LSTM" будет как "эл эс ти эм".
    Все математические знаки пиши словами. Например, "x^2 + y^2 = z^2" будет "икс в квадрате плюс игрек в квадрате равно зет в квадрате".""",
}

_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

s2s_pipeline = Speech2SpeechPipeline(
    speech2text_model=GigaAM(),
    text2text_model=GeminiChat(
        client=_genai_client,
        model="gemini-2.0-flash",
        system_prompt=_system_prompts["base"],
    ),
    text2speech_model=Silero(
        language="ru",
        model_id="v3_1_ru",
        speaker="baya",
    ),
)
