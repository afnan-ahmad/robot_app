from nlu.interface import NLUInterface
from openai import OpenAI
import os

class NLUOpenAI(NLUInterface):
    client = None
    model_name = None
    
    def __init__(self, model_name='gpt-3.5-turbo') -> None:
        if 'OPENAI_KEY' not in os.environ:
            raise Exception("OPENAI_KEY not present in environment. Please specify it to use OpenAI.")
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.model_name = model_name

    def extract_object_and_color(self, text) -> tuple:
        prompt = f"This is the text: {text}. \
                   Now from the text give me a targeted object and its color in this format: color,object. \
                   If no color is present, then 0,object. If no object is present, then 0,0"

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}])
        
        response_content = chat_completion.choices[0].message.content
        
        obj, color = response_content.split(',')

        return obj, color