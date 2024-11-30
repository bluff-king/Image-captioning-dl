import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    RobertaPreTrainedModel,
    PretrainedConfig,
    RobertaModel
)
from models.module import Pooler, SimilarityFunction

class DenseModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]

    def __init__(self,config: PretrainedConfig, args):
        super().__init__(config)
        self.args = args
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer= False)

        self.pooler = Pooler(self.args.pool_type)
        self.sim_fn = SimilarityFunction(name_fn = args.sim_fn)
        
        self.loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.post_init()
    
    def get_output(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        outputs = self.roberta(input_ids, attention_mask= attention_mask)
        pooled_output = self.pooler(attention_mask, outputs)

        return pooled_output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            input_ids_positive: Optional[torch.LongTensor] = None,
            attention_mask_positive: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        pooled_output = self.get_output(
            input_ids= input_ids,
            attention_mask= attention_mask,
        )

        if not kwargs.get('is_train', ""):
            return pooled_output
        
        pooled_output_positive = self.get_output(
            input_ids = input_ids_positive,
            attention_mask= attention_mask_positive,
        )

        sim_scores = self.sim_fn(pooled_output, pooled_output_positive)

        labels = torch.arange(sim_scores.size(0).long().to(pooled_output.device))

        loss = self.loss(sim_scores, labels)

        return loss