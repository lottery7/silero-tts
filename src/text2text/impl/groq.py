import groq

from globals.utils import latency_logging

from ..text2text import Text2TextModel


class GroqChat(Text2TextModel):
    def __init__(self, client: groq.Groq, model: str, system_prompt: str | None = None):
        super().__init__()
        self._client = client
        self._model = model
        self._history = [
            {"role": "system", "content": system_prompt},
        ]

    @latency_logging("Groq latency: {}")
    def generate(self, input_data: str) -> str:
        try:
            self._history.append({"role": "user", "content": input_data})

            completion = self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            response = completion.choices[0].message
            self._history.append({"role": "assistant", "content": response.content})

            return response.content
        except Exception:
            return "Произошла ошибка. Попробуй снова."
