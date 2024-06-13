from nlu.interface import NLUInterface
from openai import OpenAI
import os

from logging import getLogger

logger = getLogger("NLUOpenAI")

class NLUOpenAI(NLUInterface):
    client = None
    model_name = None
    
    def __init__(self, model_name, base_url=None, api_key=None) -> None:
        if base_url:
            self.initialize_local(model_name=model_name, base_url=base_url, api_key=api_key)
        else:
            self.initialize_openai(model_name=model_name)

    def initialize_local(self, model_name, base_url, api_key) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name

        logger.info("Initialized NLU with Local LLM at: ", base_url)

    def initialize_openai(self, model_name) -> None:
        if 'OPENAI_KEY' not in os.environ:
            raise Exception("OPENAI_KEY not present in environment. Please specify it to use OpenAI.")
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.model_name = model_name

        logger.info("Initialized NLU with OpenAI")

    def extract_object_and_color(self, text) -> tuple:
        logger.info("Extracting object from the text...")
        
        prompt = f"You are a word filter. \
                   You can only reply in this format: object_name. \
                   You will get a sentence and reply with the targeted object. \
                   For example, if I ask, where is the bottle? You answer: bottle. \
                   {text}"

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}])
        
        response_content = chat_completion.choices[0].message.content

        return response_content
    
    def find_most_similar_word(self, list, word):
        logger.info("Finding most similar word...")

        prompt = f"Return the word that is closest to the word \"{word}\" from the following list. \
                   Reply with the word \"None\" if none of the words closely match the word \"{word}\": {str(list)}"
        
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}])
        
        response_content = chat_completion.choices[0].message.content

        return response_content

