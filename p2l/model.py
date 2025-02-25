import torch
from transformers import (
    Qwen2Model,
    Qwen2PreTrainedModel,
    LlamaModel,
    LlamaPreTrainedModel,
    PreTrainedModel,
    AutoTokenizer,
)
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable, Optional


registered_transformers: Dict[str, Tuple[PreTrainedModel, PreTrainedModel]] = {
    "qwen2": (Qwen2PreTrainedModel, Qwen2Model),
    "llama": (LlamaPreTrainedModel, LlamaModel),
}

registered_losses: Dict[str, Callable] = {}
registered_heads: Dict[str, nn.Module] = {}
registered_inits: Dict[str, Callable] = {}

registered_aggr_models: Dict[str, nn.Module] = {}
registered_pairwise_losses: Dict[str, Callable] = {}


def register_loss(name: str):
    def decorator(func: Callable):
        registered_losses[name] = func
        return func

    return decorator


def register_head(name: str):
    def decorator(func: Callable):
        registered_heads[name] = func
        return func

    return decorator


def register_init(name: str):
    def decorator(func: Callable):
        registered_inits[name] = func
        return func

    return decorator


def register_aggr_model(name: str):
    def decorator(func: Callable):
        registered_aggr_models[name] = func
        return func

    return decorator


def register_pairwise_loss(name: str):
    def decorator(func: Callable):
        registered_pairwise_losses[name] = func
        return func

    return decorator


def register_init(name: str):
    def decorator(func: Callable):
        registered_inits[name] = func
        return func

    return decorator


@dataclass
class HeadOutputs(ModelOutput):
    coefs: torch.FloatTensor = None
    eta: Optional[torch.FloatTensor] = None
    gamma: Optional[torch.FloatTensor] = None


@dataclass
class P2LOutputs(ModelOutput):
    coefs: torch.FloatTensor = None
    eta: Optional[torch.FloatTensor] = None
    gamma: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None


@register_loss("bt")
def BT_loss(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    weights: torch.Tensor = None,
    **kwargs,
):
    # labels columns are in the form (winner_idx, loser_idx)

    coefs = head_output.coefs

    paired_coefs = coefs.gather(dim=-1, index=labels).contiguous()

    paired_delta_logit = (
        paired_coefs[:, 0] - paired_coefs[:, 1]
    )  # subtract winner bt from loser bt

    neg_log_sigma = -F.logsigmoid(paired_delta_logit)  # get neg log prob

    if weights is not None:
        neg_log_sigma = neg_log_sigma * weights

    loss = neg_log_sigma.mean()

    return loss


@register_loss("bt-tie")
def BT_tie_loss(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    weights: torch.Tensor = None,
    **kwargs,
):
    # labels columns are in the form (winner_idx, loser_idx, tie_indicator)

    coefs = head_output.coefs

    model_idx = labels[:, :2]  # (batch_dim, 2)
    tie_ind = labels[:, -1]

    paired_coefs = coefs.gather(dim=-1, index=model_idx).contiguous()

    paired_delta_logit = (
        paired_coefs[:, 0] - paired_coefs[:, 1]
    )  # subtract winner bt from loser bt

    # computes bradley-terry loss where tie is half win and half loss
    neg_log_sigma = -1 * torch.where(
        tie_ind == 0,
        F.logsigmoid(paired_delta_logit),
        0.5
        * (F.logsigmoid(paired_delta_logit) + F.logsigmoid(-1 * paired_delta_logit)),
    )

    if weights is not None:
        neg_log_sigma = neg_log_sigma * weights

    loss = neg_log_sigma.mean()

    return loss


BETA = 0.1


@register_loss("rk")
def RK_Loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    # labels columns are in form (winner_idx, loser_idx, tie_indicator)
    coefs = head_output.coefs
    # eta = torch.exp(head_output.eta).squeeze(-1)  # eta > 0
    eta = torch.clamp(
        torch.nn.functional.softplus(head_output.eta - 22.5, BETA).squeeze(-1), min=0.02
    )
    # eta = torch.abs(head_output.eta).squeeze(-1)
    model_idx = labels[:, :2]  # (batch_dim, 2)
    paired_coefs = coefs.gather(dim=-1, index=model_idx).contiguous()

    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    # compute RK probabilities
    p_w = torch.sigmoid(paired_delta_logit - eta)
    p_l = torch.sigmoid(-1 * paired_delta_logit - eta)
    p_t = 1 - p_w - p_l

    # point-wise likelihood
    A = torch.stack((p_w, p_t))  # (2, batch_dim)

    tie_ind = labels[:, -1].unsqueeze(0)  # (1, batch_dim)
    p = A.take_along_dim(dim=0, indices=tie_ind)

    # mathematically p_t < 1 always but bfloat smh
    p = torch.clamp(p, min=1e-3)

    # eps = 1e-10
    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()

    return loss


@register_loss("rk-reparam")
def RK_Reparam_Loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):

    coefs = head_output.coefs
    eta = head_output.eta

    theta = torch.exp(eta) + 1.000001

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()

    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)

    p_win = pi_win / (pi_win + theta * pi_lose + 1.0)

    p_lose = pi_lose / (pi_lose + theta * pi_win + 1.0)

    p_tie = 1.0 - p_win - p_lose

    assert p_win.shape == p_lose.shape == p_tie.shape

    P = torch.hstack((p_win, p_tie))
    tie_ind = labels[:, -1].unsqueeze(-1)

    p = P.gather(dim=-1, index=tie_ind).contiguous()

    p = torch.clamp(p, min=1e-6)

    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()

    return loss


@register_loss("ba")
def BA_loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    # labels are (winner_idx, loser_idx, tie_indicator (0 for no tie, 1 for tie, 2 for tie both bad))

    coefs = head_output.coefs
    eta = head_output.eta
    gamma = head_output.gamma

    theta = torch.exp(eta) + 1.02

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()

    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)
    pi_gamma = torch.exp(gamma)

    p_win = pi_win / (pi_win + theta * pi_lose + pi_gamma)

    p_lose = pi_lose / (pi_lose + theta * pi_win + pi_gamma)

    p_tie_bb = pi_gamma / (pi_gamma + pi_win + pi_lose)

    p_tie = 1.0 - p_win - p_lose - p_tie_bb

    P = torch.hstack((p_win, p_tie, p_tie_bb))

    tie_ind = labels[:, -1].unsqueeze(-1)

    p = P.gather(dim=-1, index=tie_ind).contiguous()

    p = torch.clamp(p, min=1e-2)

    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()

    print("loss: ", loss.item())

    return loss


@register_loss("bag")
@register_loss("grk")
def GRK_loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    # labels are (winner_idx, loser_idx, tie_indicator (0 for no tie, 1 for tie, 2 for tie both bad))

    coefs = head_output.coefs.float()
    eta = head_output.eta.float()

    theta = torch.exp(eta) + 1.000001

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()

    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)
    pi_gamma = 1.0

    p_win = pi_win / (pi_win + theta * pi_lose + pi_gamma)

    p_lose = pi_lose / (pi_lose + theta * pi_win + pi_gamma)

    p_tie_bb = pi_gamma / (pi_gamma + pi_win + pi_lose)

    p_tie = 1.0 - p_win - p_lose - p_tie_bb

    assert p_win.shape == p_lose.shape == p_tie_bb.shape == p_tie.shape
    P = torch.hstack((p_win, p_tie, p_tie_bb))

    tie_ind = labels[:, -1].unsqueeze(-1)

    p = P.gather(dim=-1, index=tie_ind).contiguous()

    p = torch.clamp(p, min=1e-6)

    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()

    # print("loss: ", loss.item())

    return loss


@register_head("bt")
class BTHead(nn.Module):
    def __init__(
        self, input_dim, output_dim, linear_head_downsize_factor=None, **kwargs
    ) -> None:
        super().__init__()

        if linear_head_downsize_factor:
            inner_dim = int(output_dim // linear_head_downsize_factor)
            self.head = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=inner_dim, bias=True),
                nn.Linear(in_features=inner_dim, out_features=output_dim, bias=True),
            )
        else:
            self.head = nn.Linear(
                in_features=input_dim, out_features=output_dim, bias=True
            )

    def forward(self, last_hidden_dim: torch.Tensor):
        coefs = self.head(last_hidden_dim)
        return HeadOutputs(coefs=coefs)


@register_head("rk")
class RKHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        eta_dim=1,
        linear_head_downsize_factor=None,
        eta_downsize=False,
        **kwargs,
    ) -> None:
        super().__init__()
        # If linear header downsize factor and eta downsize, then eta is calculated off of the downsized dim, not the hidden dim.
        if linear_head_downsize_factor:
            inner_dim = output_dim // linear_head_downsize_factor
            share_layer = nn.Linear(
                in_features=input_dim, out_features=inner_dim, bias=True
            )
            self.head = nn.Sequential(
                share_layer,
                nn.Linear(in_features=inner_dim, out_features=output_dim, bias=True),
            )
            if eta_downsize:
                self.eta_head = nn.Sequential(
                    share_layer,
                    nn.Linear(in_features=inner_dim, out_features=eta_dim, bias=True),
                )
            else:
                self.eta_head = nn.Linear(
                    in_features=output_dim, out_features=eta_dim, bias=True
                )
        else:
            self.head = nn.Linear(
                in_features=input_dim, out_features=output_dim, bias=True
            )
            self.eta_head = nn.Linear(
                in_features=input_dim, out_features=eta_dim, bias=True
            )

    def forward(self, last_hidden_dim: torch.Tensor):
        coefs = self.head(last_hidden_dim)
        eta = self.eta_head(last_hidden_dim)

        return HeadOutputs(coefs=coefs, eta=eta)


@register_head("ba")
class BAHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        linear_head_downsize_factor=None,
        **kwargs,
    ) -> None:
        super().__init__()

        if linear_head_downsize_factor:
            raise NotImplementedError("Sorry I didn't implement this.")

        self.head = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.eta_head = nn.Linear(in_features=input_dim, out_features=1, bias=True)
        self.gamma_head = nn.Linear(in_features=input_dim, out_features=1, bias=True)

    def forward(self, last_hidden_dim: torch.Tensor):

        coefs = self.head(last_hidden_dim)
        eta = self.eta_head(last_hidden_dim)
        gamma = self.gamma_head(last_hidden_dim)

        return HeadOutputs(coefs=coefs, eta=eta, gamma=gamma)


@register_init("reset_params")
def reset_params_init(module):
    return module.reset_parameters()


@register_init("he_unif")
def he_unif_init(module):
    return nn.init.kaiming_uniform_(module.weight, nonlinearity="sigmoid")


@register_init("xavier_unif")
def xavier_unif_init(module):
    return nn.init.xavier_uniform_(module.weight)


@register_init("tiny_normal")
def tiny_normal_init(module):
    return nn.init.kaiming_normal_(module.weight)


def get_p2l_model(
    model_type: str, loss_type: str, head_type: str, init_type: str = "reset_params"
) -> PreTrainedModel:
    pretrained_model_cls, model_cls = registered_transformers[model_type]

    criterion = registered_losses[loss_type]

    head_layer = registered_heads[head_type]

    init_func = registered_inits[init_type]

    class CustomPretrainedModel(pretrained_model_cls):
        """Defines the appropriate pretrained class for the given model name.  This is done so that the value head init scheme is correct."""

        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                init_func(module)  # was reset params
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    class P2LModel(CustomPretrainedModel):
        def __init__(
            self,
            config,
            CLS_id,
            num_models,
            linear_head_downsize_factor=None,
            head_kwargs={},
            **kwargs,
        ):
            super().__init__(config)

            self.num_models = num_models
            self.cls_token_id = CLS_id

            self.model = model_cls(config)

            self.head = head_layer(
                input_dim=config.hidden_size,
                output_dim=self.num_models,
                linear_head_downsize_factor=linear_head_downsize_factor,
                **head_kwargs,
            )

            self.post_init()

        def freeze_transformer(self):
            for param in self.model.parameters():
                param.requires_grad = False

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(self, input_ids, attention_mask, labels=None, weights=None):
            batch_size = input_ids.shape[0]

            hidden_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).last_hidden_state  # (bs, num_token, embed_dim)

            cls_mask = input_ids == self.cls_token_id

            # double check this is getting the current CLS token
            cls_hidden_dim = hidden_outputs[cls_mask]

            assert (
                cls_hidden_dim.shape[0] == batch_size
            ), f"input ids {input_ids.shape}, cls_mask {cls_mask.shape}, cls_logit {cls_hidden_dim.shape}"

            head_output = self.head(cls_hidden_dim)

            if labels is not None:
                loss = criterion(head_output, labels, weights=weights)

                outputs = P2LOutputs(
                    coefs=head_output.coefs,
                    last_hidden_state=cls_hidden_dim,
                    eta=head_output.eta,
                    gamma=head_output.gamma,
                    loss=loss,
                )

            else:
                outputs = P2LOutputs(
                    coefs=head_output.coefs,
                    last_hidden_state=cls_hidden_dim,
                    eta=head_output.eta,
                    gamma=head_output.gamma,
                )

            return outputs

    return P2LModel


def get_tokenizer(
    tokenizer_name,
    chat_template,
    pad_token_if_none="<|pad|>",
    cls_token_if_none="<|cls|>",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    if chat_template:
        tokenizer.chat_template = chat_template

    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"pad_token": pad_token_if_none})
    if "cls_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"cls_token": cls_token_if_none})

    return tokenizer


@register_aggr_model("bt")
@register_aggr_model("bt-tie")
class BTAggrModel(nn.Module):
    def __init__(self, num_models, batch_size=1):
        super().__init__()
        self.coefs = nn.Parameter(
            nn.init.constant_(torch.empty(batch_size, num_models), 0.5)
        )
        self.eta = None

    def forward(self):
        return self.coefs, self.eta


@register_aggr_model("rk")
@register_aggr_model("rk-reparam")
@register_aggr_model("bag")
@register_aggr_model("grk")
class RKAggrModel(nn.Module):
    def __init__(self, num_models, batch_size=1):
        super().__init__()
        self.coefs = nn.Parameter(
            nn.init.constant_(torch.empty(batch_size, num_models), 0.5)
        )
        self.eta = nn.Parameter(nn.init.constant_(torch.empty(batch_size, 1), 0.1))

    def forward(self):
        return self.coefs, self.eta


@register_pairwise_loss("bt")
@register_pairwise_loss("bt-tie")
def pairwise_batch_BT_loss(
    real_output: HeadOutputs, aggregated_output: HeadOutputs, true_probs: torch.tensor
):
    real_betas = real_output.coefs
    aggregated_betas = aggregated_output.coefs

    num_prompts, num_models = real_betas.shape[-2], real_betas.shape[-1]

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_agg = aggregated_betas[:, pair_indices[:, 0]]
    beta_j_agg = aggregated_betas[:, pair_indices[:, 1]]

    pred_probs = torch.sigmoid(beta_i_agg - beta_j_agg)

    pred_probs_expanded = pred_probs.unsqueeze(1).expand(-1, num_prompts, -1)

    eps = 1e-9
    neg_log_prob = -(
        true_probs * torch.log(pred_probs_expanded + eps)
        + (1 - true_probs) * torch.log(1 - pred_probs_expanded + eps)
    )

    batch_losses = neg_log_prob.mean(dim=(1, 2))
    loss = batch_losses.mean()

    return loss


# batched loss
@register_pairwise_loss("rk")
def pairwise_batch_RK_loss(
    real_output: HeadOutputs, aggregated_output: HeadOutputs, true_probs: torch.tensor
):
    real_betas = real_output.coefs
    num_prompts, num_models = real_betas.shape[-2], real_betas.shape[-1]

    aggregated_betas = aggregated_output.coefs
    BETA = 0.1
    aggregated_eta = torch.clamp(
        torch.nn.functional.softplus(aggregated_output.eta - 22.5, BETA).squeeze(-1),
        min=0.02,
    )

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_agg = aggregated_betas[:, pair_indices[:, 0]]
    beta_j_agg = aggregated_betas[:, pair_indices[:, 1]]

    aggregated_eta = aggregated_eta.unsqueeze(-1)
    pred_probs_win = torch.sigmoid(beta_i_agg - beta_j_agg - aggregated_eta)
    pred_probs_loss = torch.sigmoid(beta_j_agg - beta_i_agg - aggregated_eta)
    pred_probs_tie = 1 - pred_probs_win - pred_probs_loss

    pred_probs = torch.stack((pred_probs_win, pred_probs_loss, pred_probs_tie), dim=-1)

    pred_probs_expanded = pred_probs.unsqueeze(1).expand(-1, num_prompts, -1, -1)

    eps = 1e-9
    neg_log_prob = -torch.sum(true_probs * torch.log(pred_probs_expanded + eps), dim=-1)

    batch_losses = neg_log_prob.mean(dim=(1, 2))
    loss = batch_losses.mean()

    return loss


# batched
@register_pairwise_loss("rk-reparam")
def pairwise_batch_RK_reparam_loss(
    real_output: HeadOutputs,
    aggregated_output: HeadOutputs,
    true_probs: torch.tensor,
    **kwargs,
):
    real_betas = real_output.coefs
    num_prompts, num_models = real_betas.shape[-2], real_betas.shape[-1]

    aggregated_betas = aggregated_output.coefs
    aggregrated_theta = torch.exp(aggregated_output.eta) + 1.000001

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_agg = aggregated_betas[:, pair_indices[:, 0]]
    beta_j_agg = aggregated_betas[:, pair_indices[:, 1]]

    pi_win = torch.exp(beta_i_agg)
    pi_lose = torch.exp(beta_j_agg)

    p_win = pi_win / (pi_win + aggregrated_theta * pi_lose + 1.0)
    p_lose = pi_lose / (pi_lose + aggregrated_theta * pi_win + 1.0)
    p_tie = 1.0 - p_win - p_lose

    pred_probs = torch.stack((p_win, p_lose, p_tie), dim=-1)
    pred_probs_expanded = pred_probs.unsqueeze(1).expand(-1, num_prompts, -1, -1)

    eps = 1e-9
    neg_log_prob = -torch.sum(true_probs * torch.log(pred_probs_expanded + eps), dim=-1)
    batch_losses = neg_log_prob.mean(dim=(1, 2))
    loss = batch_losses.mean()

    return loss


def get_bag_probs(beta_win, beta_lose, gamma, theta):
    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)
    pi_gamma = 1.0

    p_win = pi_win / (pi_win + theta * pi_lose + pi_gamma)

    p_lose = pi_lose / (pi_lose + theta * pi_win + pi_gamma)

    p_tie_bb = pi_gamma / (pi_gamma + pi_win + pi_lose)

    p_tie = 1.0 - p_win - p_lose - p_tie_bb

    return torch.stack((p_win, p_lose, p_tie, p_tie_bb), dim=-1)


# batched
@register_pairwise_loss("bag")
@register_pairwise_loss("grk")
def pairwise_batch_bag_loss(
    real_output: HeadOutputs,
    aggregated_output: HeadOutputs,
    true_probs: torch.tensor,
    **kwargs,
):
    real_betas = real_output.coefs
    num_prompts, num_models = real_betas.shape[-2], real_betas.shape[-1]

    aggregated_betas = aggregated_output.coefs
    aggregrated_theta = torch.exp(aggregated_output.eta) + 1.000001

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_agg = aggregated_betas[:, pair_indices[:, 0]]
    beta_j_agg = aggregated_betas[:, pair_indices[:, 1]]

    pred_probs = get_bag_probs(beta_i_agg, beta_j_agg, 1.0, aggregrated_theta)

    pred_probs_expanded = pred_probs.unsqueeze(1).expand(-1, num_prompts, -1, -1)

    eps = 1e-9
    neg_log_prob = -torch.sum(true_probs * torch.log(pred_probs_expanded + eps), dim=-1)
    batch_losses = neg_log_prob.mean(dim=(1, 2))
    loss = batch_losses.mean()

    return loss


@register_loss("tie-rk")
def RK_Tie_Loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs
    eta = torch.clamp(
        torch.nn.functional.softplus(head_output.eta - 22.5, BETA).squeeze(-1), min=0.02
    )
    model_idx = labels[:, :2]
    paired_coefs = coefs.gather(dim=-1, index=model_idx).contiguous()

    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    p_w = torch.sigmoid(paired_delta_logit - eta)
    p_l = torch.sigmoid(-1 * paired_delta_logit - eta)
    p_t = 1 - p_w - p_l

    p_not_t = p_w + p_l
    p_t = p_t

    A = torch.stack((p_not_t, p_t))

    tie_ind = labels[:, -1].unsqueeze(0)
    p = A.take_along_dim(dim=0, indices=tie_ind)

    p = torch.clamp(p, min=1e-3)

    loss = -torch.log(p)
    if weights:
        loss = loss * weights
    loss = loss.mean()

    return loss


@register_loss("tie-bag")
@register_loss("tie-grk")
def bag_tie_loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs
    eta = head_output.eta

    theta = torch.exp(eta) + 1.000001

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()

    p_win, p_lose, p_tie, p_tie_bb = torch.unbind(
        get_bag_probs(beta_win, beta_lose, 1.0, theta), dim=-1
    )

    P = torch.hstack((p_win + p_lose, p_tie + p_tie_bb))

    tie_ind = labels[:, -1].unsqueeze(-1)
    tie_ind = torch.where(tie_ind == 0, 0, 1)  # segment into ties and not ties

    p = P.gather(dim=-1, index=tie_ind).contiguous()

    p = torch.clamp(p, min=1e-6)

    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()
    return loss


@register_loss("tie-bb-bag")
@register_loss("tie-bb-grk")
def bag_tie_bb_loss(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs
    eta = head_output.eta

    theta = torch.exp(eta) + 1.000001

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()

    p_win, p_lose, p_tie, p_tie_bb = torch.unbind(
        get_bag_probs(beta_win, beta_lose, 1.0, theta), dim=-1
    )

    P = torch.hstack((p_win + p_lose + p_tie, p_tie_bb))

    tie_ind = labels[:, -1].unsqueeze(-1)
    tie_ind = torch.where(tie_ind == 2, 1, 0)  # index should be 1 if tie-bb

    p = P.gather(dim=-1, index=tie_ind).contiguous()

    p = torch.clamp(p, min=1e-6)

    loss = -torch.log(p)

    if weights:
        loss = loss * weights

    loss = loss.mean()
    return loss
