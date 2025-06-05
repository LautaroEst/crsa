

from pathlib import Path
from typing import Literal, Optional
import torch
from litgpt.api import LLM as LitGPTLLM, Preprocessor, extend_checkpoint_dir
from .prompts import PromptStyle
from .tokenizer import Tokenizer


class LLMSpeaker(LitGPTLLM):

    @classmethod
    def load(
        cls, 
        model: str,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None,
        distribute: Optional[Literal["auto"]] = "auto"
    ):
        model = super().load(model, init, tokenizer_dir, access_token, distribute)
        model.prompt_style = PromptStyle.from_config(model.config)
        
        if tokenizer_dir is not None:
            tokenizer_dir = extend_checkpoint_dir(Path(tokenizer_dir))
            tokenizer = Tokenizer(tokenizer_dir)
        elif model.checkpoint_dir is not None:
            tokenizer = Tokenizer(model.checkpoint_dir)
        else:
            raise ValueError("Provide a path to a tokenizer directory via the `tokenizer_dir` setting.")
        
        model.preprocessor = Preprocessor(tokenizer=tokenizer, device=model.preprocessor.device)
        return model
    
    def encode_prompt_string(self, prompt_str: str):
        return self.preprocessor.tokenizer.encode(prompt_str, bos=False, eos=False, device=self.preprocessor.device)

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
    
    @torch.inference_mode()
    def get_dialog_speakers(self, dialog, meaning_system_prompts, speakers):
        
        logits = {}
        tokens = {}
        offsets = {}
        for speaker in speakers:
            logits[speaker] = []
            tokens[speaker] = []
            offsets[speaker] = []
            for m in range(len(meaning_system_prompts[speaker])):
                system_prompt = meaning_system_prompts[speaker][m]
                prompt = [{"role": "system", "content": system_prompt}]
                prompt_str = self.prompt_style.apply(prompt)
                encoded_prompt = self.encode_prompt_string(prompt_str)
                offsets[speaker].append([encoded_prompt.size(0)])
                for utterance in dialog:
                    role = "assistant" if speaker == utterance["speaker"] else "user"
                    prompt.append({"role": role, "content": utterance["content"]})
                    prompt_str = self.prompt_style.apply(prompt)
                    encoded_prompt = self.encode_prompt_string(prompt_str)
                    offsets[speaker][-1].append(encoded_prompt.size(0))
                prompt_str = self.prompt_style.apply(prompt)
                encoded_prompt = self.encode_prompt_string(prompt_str)
                logits[speaker].append(self(encoded_prompt.unsqueeze(0)).log_softmax(dim=-1).cpu())
                tokens[speaker].append(encoded_prompt[1:].tolist() + [self.preprocessor.tokenizer.eos_id])
            
        speakers_logits = []
        speaker_idx = 0
        for i, utterance in enumerate(dialog):
            speaker = utterance["speaker"]
            first = True
            for m in range(len(meaning_system_prompts[speaker])):
                start = offsets[speaker][m][i]
                end = offsets[speaker][m][i + 1]
                idx = 0
                for j in range(start, end):
                    if first:
                        speakers_logits.append({"speaker": speaker, "content": tokens[speaker][m][j],"logits": [logits[speaker][m][0, j, :]]})
                    else:
                        speakers_logits[speaker_idx + idx]["logits"].append(logits[speaker][m][0, j, :])
                    idx += 1
                first = False
            first = True
            speaker_idx += end - start

        for speaker_idx in range(len(speakers_logits)):
            speakers_logits[speaker_idx]["logits"] = torch.stack(speakers_logits[speaker_idx]["logits"], dim=0)

        return speakers_logits

            
            

            
            
            

        

            








