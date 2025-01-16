from pydantic import BaseModel
from typing import Optional

class InputSchema(BaseModel):
    tool_name: str
    prompt: str
    input_dir: Optional[str] = None