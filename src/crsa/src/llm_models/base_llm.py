


import torch
from litgpt import LLM as LitGPTLLM
from .prompts import PromptStyle


class LLM(LitGPTLLM):

    @classmethod
    def load(cls, *args, **kwargs):
        model = super().load(*args, **kwargs)
        model.prompt_style = PromptStyle.from_config(model.config)
        return model

    @torch.inference_mode()
    def predict(self, prompt, endings, reduce="mean"):

        if self.model is None:
            raise AttributeError(
                "The model is not initialized yet; use the .distribute() "
                "or .trainer_setup() method to initialize the model."
            )
        input_ids = self.preprocessor.encode(prompt)
        endings = [self.preprocessor.encode(ending).long() for ending in endings]
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max([ending.size(0) for ending in endings])

        if self.fabric is not None:
            device = self.fabric.device
        else:
            device = self.preprocessor.device

        if not self.kv_cache_initialized:
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.kv_cache_initialized = True

        # Dynamically grow the kv cache size if necessary
        if not self.fixed_kv_cache_size and self.prev_generated_seq_length < max_returned_tokens:
            tmp_device = self.model.mask_cache.device
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=tmp_device)

        else:
            for block in self.model.transformer.h:
                block.attn.kv_cache.reset_parameters()

        self.prev_generated_seq_length = max_returned_tokens
        self.model.eval()

        # Input ids
        input_ids = input_ids.unsqueeze(0)

        # Input pos
        input_pos = torch.arange(0, prompt_length, device=device, dtype=torch.int64)
        
        # Input pos maxp1: introduces data-dependent shapes and control flow.
        # We want to skip if ThunderModules are involved, either directly or wrapped in LightningModule etc.
        input_pos_maxp1 = prompt_length if all(m.__class__.__name__ != "ThunderModule" for m in self.model.modules()) else None

        # Compute the output logits for the first part of the sentence
        output = self.model(idx=input_ids, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)
        
        endings_logits = []
        for ending in endings:

            # Input ids for the ending
            ending = ending.unsqueeze(0)
            
            # Input pos for the ending
            input_pos = torch.arange(prompt_length, ending.shape[1] + prompt_length, device=device, dtype=torch.int64)

            # Input pos maxp1 for the ending
            if input_pos_maxp1 is not None:
                input_pos_maxp1 = prompt_length + ending.shape[1]

            # Compute the output logits for the ending
            end_out = self.model(idx=ending, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)

            # Gather the log probabilities for the full sentence
            logprobs = torch.cat([output[:,-1:,:], end_out[:,:-1,:]], dim=1).log_softmax(dim=2)
            index = ending.unsqueeze(2)
            gather_probs = torch.gather(logprobs, -1, index).squeeze(2)

            if reduce == "mean":
                end_logit = gather_probs.mean()
            elif reduce == "sum":
                end_logit = gather_probs.sum()
            else:
                raise ValueError(f"Unknown reduction method: {reduce}")
            endings_logits.append(end_logit)
        
        endings_logits = torch.stack(endings_logits, dim=0).cpu().numpy()
        return endings_logits

