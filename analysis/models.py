from typing import List, Dict, Any, Optional
import random
from pydantic import BaseModel
from dataclasses import dataclass, field

class Utterance(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    
if __name__ == "__main__":
    utter = Utterance(question="hi, how are you?")
    print(utter.model_dump())
    