

from typing import Optional
from litgpt.tokenizer import Tokenizer as LitGPTTokenizer
import torch


class Tokenizer(LitGPTTokenizer):

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]
        # if the processor misbehaves and adds `eos` token no matter what
        elif tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)
