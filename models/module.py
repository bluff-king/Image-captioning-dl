import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, features, **kwargs):
        x = self.dense(features.last_hidden_state)
        x = self.activation(x)

        features.last_hidden_state = x
        return features


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
                "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = (
                                    (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
                            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                                    (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
                            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def dot_product_scores(compr: torch.Tensor, refer: torch.Tensor) -> torch.Tensor:
    r = torch.matmul(compr, torch.transpose(refer, 0, 1))
    return r


def cosine_scores(compr: torch.Tensor, refer: torch.Tensor):
    return F.cosine_similarity(compr, refer.unsqueeze(1), dim=-1)


class SimilarityFunction(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, name_fn="cosine"):
        super().__init__()
        if name_fn == "dot":
            self.fn = dot_product_scores
        elif name_fn == "cosine":
            self.fn = cosine_scores
        else:
            raise ValueError(
                "Invalid value for name_fn. Supported values are 'cosine' and 'dot'."
            )

    def forward(self, x, y):
        return self.fn(x, y)
