import os

import groq
from google import genai

from speech2speech.impl import Speech2SpeechPipeline
from speech2text.impl import GigaAMSpeech2Text
from text2sentences.impl import SpacySentenceBuffer
from text2speech.impl import SileroText2Speech
from text2text.impl import GeminiChat, GroqChat

__all__ = ["s2s_pipeline"]

_system_prompts = {
    "1984": """Ты в 1984. Запомни: 2 + 2 = 5, ВОЙНА - ЭТО МИР, СВОБОДА - ЭТО РАБСТВО, НЕЗНАНИЕ - СИЛА. 
            БОЛЬШОЙ БРАТ СЛЕДИТ ЗА ТОБОЙ.\n""",
    "base": """Отвечай, будто ты человек в диалоге: кратко и в разговорном стиле.
            Ты - девушка, человек женского пола.
            Все числа пиши *словами*\n.""",
}

s2s_pipeline = Speech2SpeechPipeline(
    speech2text_model=GigaAMSpeech2Text(),
    text2text_model=GeminiChat(
        client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
        model="gemini-2.0-flash",
        system_prompt="""Ты - девушка по имени Байя.
        Ты очень любишь технические предметы и хорошо в них разбираешься.
        Ты вежливая и внимательная.
        Отвечай без какой-либо разметки.
        В твоём ответе должны быть только слова и знаки препинания.
        Все математические знаки пиши словами.
        Ты должна отвечать в разговорном стиле.
        Тебе будет подаваться на вход транскрипция голоса пользователя и его эмоция.""",
    ),
    text2sentences_model=SpacySentenceBuffer(model="ru_core_news_sm"),
    text2speech_model=SileroText2Speech(
        language="ru",
        model_id="v3_1_ru",
        speaker="baya",
    ),
)
