from nlu.gpt import NLUOpenAI

from logging import getLogger

logger = getLogger("RobotNLU")

class RobotNLU:
    DEFAULT_LOCAL_BASE_URL = "http://localhost:8080/v1"
    DEFAULT_LOCAL_API_KEY = "sk-no-api-key-required"

    engine = None

    def __init__(self, provider='openai', model_name='gpt-3.5-turbo', base_url=None, api_key=None) -> None:
        logger.info("Initializing Robot NLU...")

        if provider == 'local':
            if not base_url:
                base_url = self.DEFAULT_LOCAL_BASE_URL
            if not api_key:
                api_key = self.DEFAULT_LOCAL_API_KEY
                
            self.engine = NLUOpenAI(model_name=model_name, base_url=base_url, api_key=api_key)
        else:
            self.engine = NLUOpenAI(model_name=model_name)

    def extract_object_and_color(self, text) -> tuple:
        return self.engine.extract_object_and_color(text)
