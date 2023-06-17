import openai
from config import TOKEN


class ChatGPT:
    openai.api_key = TOKEN

    def __init__(
                self,
                model_engine: str,
                max_tokens: int = 3000,
                temperature: int = 0.5,
                **kwargs
            ) -> None:
        self.model_engine = model_engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

    def generate_text(self, prompt: str):
        response = openai.Completion.create(
            prompt=prompt,
            engine=self.model_engine,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.kwargs
        )
        response_text = response['choices'][0]["text"]
        response_status = response['choices'][0]['finish_reason']

        return response_status, response_text


if __name__ == '__main__':

    prompt = str(input('Input prompt:\n'))

    status, text = ChatGPT(
            model_engine='text-davinci-003',
            max_tokens=15,
            temperature=0
        ).generate_text(prompt)

    print(text)
    print(status)
