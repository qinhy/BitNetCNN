from __future__ import annotations

import json
import math
from typing import Optional, Sequence, Tuple, Union

from pydantic import BaseModel
from torch import nn
import torch
import torch.nn.functional as F

class LinearModels:
    class BasicModel(BaseModel):
        def build(self):
            mod = LinearModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)

    type = Union[]

class LinearModules:
    class Module(nn.Module):
        def __init__(self, para: BaseModel, para_cls):
            if isinstance(para, dict):
                para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())
            super().__init__()
