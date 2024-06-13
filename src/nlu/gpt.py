from nlu.interface import NLUInterface
from openai import OpenAI
import os

from logging import getLogger

logger = getLogger("NLUOpenAI")

class NLUOpenAI(NLUInterface):
    client = None
    model_name = None
    
    def __init__(self, model_name, base_url, api_key) -> None:
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
        logger.info("Attempting to extract object and color...")
        
        prompt = f"This is the text: {text}. \
                   Now from the text give me a targeted object and its color in this format: color,object. \
                   If no color is present, then 0,object. If no object is present, then 0,0"

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}])
        
        response_content = chat_completion.choices[0].message.content
        
        obj, color = response_content.split(',')

        return obj, color