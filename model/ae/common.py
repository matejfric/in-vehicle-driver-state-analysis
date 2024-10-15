from typing import Any

import torch


# Newtype Design Pattern
class Encoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# Newtype Design Pattern
class Decoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(args, kwargs)
