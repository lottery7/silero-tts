from google import genai
from google.genai import types
from google.genai.chats import Chat

from utils import latency_logging

from ..text2text import Text2TextModel


class GeminiChat(Text2TextModel):
    def __init__(
        self,
        client: genai.Client,
        model: str,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self._chat: Chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )

    @latency_logging("Gemini latency: {}")
    def generate(self, input_data: str) -> str:
        try:
            response = self._chat.send_message(input_data)
            return response.text
        except Exception as e:
            print(e)
            return "Произошла ошибка. Попробуй снова."
