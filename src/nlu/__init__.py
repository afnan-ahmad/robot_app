from nlu.gpt import NLUOpenAI
from nlu.sbert import NLUSBERT

from logging import getLogger

logger = getLogger("RobotNLU")

class RobotNLU:
    DEFAULT_LOCAL_BASE_URL = "http://localhost:8080/v1"
    DEFAULT_LOCAL_API_KEY = "sk-no-api-key-required"

    YOLO_CLASSES_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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
    
    def find_most_similar_class(self, word) -> str:
        return self.engine.find_most_similar_word(self.YOLO_CLASSES_LIST, word)

    # def find_most_similar_class(self, word) -> str:
    #     sbert_engine = NLUSBERT()
    #     return sbert_engine.find_most_similar_word(self.YOLO_CLASSES_LIST, word)