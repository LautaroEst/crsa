
import numpy as np
import lightning as L
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


class LLMDialogTurn:
    
    def __init__(
        self,
        system_prompt_S,
        system_prompt_L,
        utterances,
        categories,
        past_utterances,
        llm,
        game=None
    ):
        self.system_prompt_S = system_prompt_S
        self.system_prompt_L = system_prompt_L
        self.categories = categories
        self.utterances = utterances
        self.past_utterances = past_utterances
        self.llm = llm

        if game == "findA1":
            from .findA1_agents import Speaker, Listener
            self.speaker_cls = Speaker
            self.listener_cls = Listener
        else:
            raise ValueError(f"Game {game} not supported")

    def run(self):

        self.speaker = self.speaker_cls(
            system_prompt=self.system_prompt_S, 
            utterances=self.utterances,
            past_utterances=self.past_utterances,
            llm=self.llm,
        )
        # self.speaker.run()

        self.listener = self.listener_cls(
            system_prompt=self.system_prompt_L, 
            categories=self.categories,
            past_utterances=self.past_utterances,
            llm=self.llm,
        )
        # self.listener.run()



class LLMDialog:

    def __init__(
        self,
        system_prompt_A,
        system_prompt_B,
        utterances,
        categories,
        llm,
        game="findA1",
    ):
        
        self.system_prompt_A = system_prompt_A
        self.system_prompt_B = system_prompt_B
        self.categories = categories
        self.utterances = utterances
        self.llm = llm
        self.game = game

        # History
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []

    def sample_new_utterance_from_last_speaker(self, meaning_S):
        speaker = self.turns_history[-1].speaker
        if speaker.has_not_run_yet:
            speaker.run()
        utt_dist = speaker.as_df
        return utt_dist[utt_dist == utt_dist.max()].sample(n=1).index[0]

    def get_category_dist_from_last_listener(self, new_utt, meaning_L):
        listener = self.turns_history[-1].listener
        if listener.has_not_run_yet:
            listener.run(new_utt=new_utt)
        return listener.as_df.values.reshape(-1)

    def run(self, utterances, speaker_now="A"):

        turns_runned = len(self.turns_history)
        self.past_utterances.extend(utterances)
        self.speaker_now = speaker_now

        turns = len(self.past_utterances) + 1
        for turn in range(turns_runned + 1, turns + 1):

            # Determine the speaker and listener
            speaking_agent = self.past_utterances[turn-1]["speaker"] if turn <= len(self.past_utterances) else self.speaker_now
            past_utterances = self.past_utterances[:turn-1]

            # Run for the turn
            model = self._run_turn(past_utterances, speaking_agent)
            self.turns_history.append(model)
            
    def _run_turn(self, past_utterances, speaker="A"):
        system_prompt_S = self.system_prompt_A if speaker == "A" else self.system_prompt_B
        system_prompt_L = self.system_prompt_B if speaker == "A" else self.system_prompt_A
        past_utterances = [{"utterance": u["utterance"], "speaker": "S" if u["speaker"] == "A" else "L"} for u in past_utterances]
        
        model = LLMDialogTurn(
            system_prompt_S=system_prompt_S,
            system_prompt_L=system_prompt_L,
            utterances=self.utterances,
            categories=self.categories,
            past_utterances=past_utterances,
            llm=self.llm,
            game=self.game,
        )
        model.run()
        return model
    
