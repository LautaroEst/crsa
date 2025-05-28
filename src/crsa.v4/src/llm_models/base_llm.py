


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
    def predict(self, prompt, endings):

        if self.model is None:
            raise AttributeError(
                "The model is not initialized yet; use the .distribute() "
                "or .trainer_setup() method to initialize the model."
            )
        input_ids = self.preprocessor.encode(prompt)
        endings = [self.preprocessor.encode(ending).long() for ending in endings]
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max([ending.size(0) for ending in endings])

        if not self.kv_cache_initialized:
            if self.fabric is not None:
                device = self.fabric.device
            else:
                device = self.preprocessor.device
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

        # Get scores for every ending
        input_ids = input_ids.unsqueeze(0)
        with self.fabric.init_tensor():
            input_pos = torch.arange(0, prompt_length)
        output = self.model(idx=input_ids, input_pos=input_pos)
        endings_logits = []
        for ending in endings:
            ending = ending.unsqueeze(0)
            input_pos = torch.arange(prompt_length, ending.shape[1] + prompt_length, device=ending.device, dtype=torch.long)
            end_out = self.model(idx=ending, input_pos=input_pos)
            logprobs = torch.cat([output[:,-1:,:], end_out[:,:-1,:]], dim=1).log_softmax(dim=2)
            index = ending.unsqueeze(2)
            gather_probs = torch.gather(logprobs, -1, index).squeeze(2)
            end_logit = gather_probs.sum()
            endings_logits.append(end_logit)
        endings_logits = torch.stack(endings_logits, dim=0).cpu().numpy()
        return endings_logits

