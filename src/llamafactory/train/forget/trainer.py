import json
import os
import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer
from transformers import BatchEncoding, Trainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
import torch.nn.functional as F
from transformers import PreTrainedModel
import copy
import deepspeed

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)

class CustomTrainerForgetting(Seq2SeqTrainer):
    def __init__(
        self, 
        model: Union["PreTrainedModel", torch.nn.Module],
        oracle_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        loss_type='KL',
        **kwargs
        ):
        self.model = model
        self.oracle_model = oracle_model
        self.loss_type = loss_type
        
        Trainer.__init__(self, model=model, **kwargs)
        if self.loss_type == "KL":
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
        # if oracle_model is not None:
        #     if self.is_deepspeed_enabled:
        #         if not (
        #             getattr(oracle_model, "is_loaded_in_8bit", False) or getattr(oracle_model, "is_loaded_in_4bit", False)
        #         ):  # quantized models are already set on the correct device
        #             self.oracle_model = self._prepare_deepspeed(self.oracle_model)
        #     else:
        #         self.oracle_model = self.accelerator.prepare_model(self.oracle_model, evaluation_mode=True)
    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ['retain_input_ids', 'retain_attention_mask', 'retain_labels']
    
    def compute_loss(self, model, inputs, return_outputs=False):
     
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_outputs = model(**retain_inputs)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            with torch.no_grad():
                retain_outputs = self.oracle_model(**retain_inputs)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(**retain_inputs)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        print("loss:", loss)
        return (loss, outputs) if return_outputs else loss


