from nlu.gpt import NLUOpenAI

class RobotNLU:
    engine = None

    def __init__(self, provider='openai') -> None:
        # For now, we're not checking the provider parameter, as only OpenAI interface is implemented.

        self.engine = NLUOpenAI()

    def extract_object_and_color(self, text) -> tuple:
        return self.engine.extract_object_and_color(text)
