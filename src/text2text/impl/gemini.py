from google import genai
from google.genai import types
from google.genai.chats import Chat

from ..text2text import Text2TextModel


class GeminiChatText2Text(Text2TextModel):
    def __init__(self, client: genai.Client, system_prompt: str | None = None):
        super().__init__()
        self._chat: Chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )

    def generate(self, input_data: str) -> str:
        while True:
            try:
                response = self._chat.send_message(input_data)
                return response.text or ""
            except ValueError:
                return ""
            except:
                continue
