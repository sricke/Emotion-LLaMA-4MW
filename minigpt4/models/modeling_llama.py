import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig
from transformers import PreTrainedTokenizer

class LlamaForCausalLM(LlamaForCausalLMOrig):

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        emotion: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        reduction: Optional[str] = "mean",
        class_weights: Optional[torch.Tensor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        return_pred_texts: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            batch_size = logits.size(0)
            seq_len = shift_logits.size(0) // batch_size
            
            
            # Each token in a sample gets the weight of that sample's class
            # Expand weights: [batch_size] -> [batch_size * seq_len]
            # Each sample's tokens get the same weight
            token_weights = class_weights.unsqueeze(1).expand(batch_size, seq_len).contiguous()
            token_weights = token_weights.view(-1)  # Flatten to [batch_size * seq_len]
            
            # Only weight non-ignored tokens (labels != -100)
            # Set weight to 0 for ignored tokens so they don't contribute to loss
            ignore_mask = (shift_labels == -100)
            token_weights = token_weights * (~ignore_mask).float()
            
            pred_texts = []
            gt_texts = []
            # Decode predictions
            if tokenizer is not None:
                # Get predicted token IDs (before shifting, to match original sequence)
                pred_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

                # Decode only the answer tokens (where labels != -100)
                for batch_idx in range(batch_size):
                    answer_mask = (labels[batch_idx] != -100)
                    # Find answer token positions
                    # Get predicted and ground truth tokens for answer portion
                    pred_answer_tokens = pred_token_ids[batch_idx][answer_mask]
                    gt_answer_tokens = labels[batch_idx][answer_mask]
                    
                    # Filter out -100 before decoding
                    pred_token_list = pred_answer_tokens.cpu().tolist()
                    gt_token_list = gt_answer_tokens.cpu().tolist()
                    pred_token_list = [token for token in pred_token_list if token != -100]
                    gt_token_list = [token for token in gt_token_list if token != -100]
                    pred_text = tokenizer.decode(pred_token_list, skip_special_tokens=True)
                    gt_text = tokenizer.decode(gt_token_list, skip_special_tokens=True)
                    #print(f"Batch {batch_idx} - Predicted: '{pred_text}' | Ground Truth: '{gt_text}'")
                    pred_texts.append(pred_text)
                    gt_texts.append(gt_text)
           
            # Use weighted CrossEntropyLoss
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss_per_token = loss_fct(shift_logits, shift_labels)
            # Apply weights
            loss = (loss_per_token * token_weights).sum() / token_weights.sum()

            
            if reduction == "none":
                loss = loss.view(logits.size(0), -1).mean(1)

    

        if return_pred_texts:
            return (loss, pred_texts, gt_texts)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
