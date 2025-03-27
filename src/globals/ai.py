import os

from google import genai

from speech2speech.impl import Speech2SpeechPipeline
from speech2text.impl import GigaAMSpeech2Text
from text2speech.impl import SileroText2Speech
from text2text.impl import GeminiChatText2Text

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
    text2text_model=GeminiChatText2Text(
        client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
        system_prompt=_system_prompts["1984"] + _system_prompts["base"],
    ),
    text2speech_model=SileroText2Speech(
        language="ru",
        model_id="v3_1_ru",
        speaker="baya",
    ),
)
