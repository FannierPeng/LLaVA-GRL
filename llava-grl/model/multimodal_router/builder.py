import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import fcntl
import numpy as np
import random
from torch.nn.init import trunc_normal_
import math
from functools import partial

# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)
#
#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token:
#         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
#     return pos_embed
#
#
# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0
#
#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
#
#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float32)
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega  # (D/2,)
#
#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
#
#     emb_sin = np.sin(out) # (M, D/2)
#     emb_cos = np.cos(out) # (M, D/2)
#
#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb
#
# def interpolate_pos_embed(model, checkpoint_model):
#     if 'pos_embed' in checkpoint_model:
#         pos_embed_checkpoint = checkpoint_model['pos_embed']
#         embedding_size = pos_embed_checkpoint.shape[-1]
#         num_patches = model.patch_embed.num_patches
#         num_extra_tokens = model.pos_embed.shape[-2] - num_patches
#         # height (== width) for the checkpoint position embedding
#         orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
#         # height (== width) for the new position embedding
#         new_size = int(num_patches ** 0.5)
#         # class_token and dist_token are kept unchanged
#         if orig_size != new_size:
#             print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
#             extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#             # only the position tokens are interpolated
#             pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#             pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
#             pos_tokens = torch.nn.functional.interpolate(
#                 pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#             pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#             new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#             checkpoint_model['pos_embed'] = new_pos_embed
# def get_abs_pos(abs_pos, tgt_size):
#     # abs_pos: L, C
#     # tgt_size: M
#     # return: M, C
#     src_size = int(math.sqrt(abs_pos.size(0)))
#     tgt_size = int(math.sqrt(tgt_size))
#     dtype = abs_pos.dtype
#
#     if src_size != tgt_size:
#         return F.interpolate(
#             abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
#             size=(tgt_size, tgt_size),
#             mode="bicubic",
#             align_corners=False,
#         ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
#     else:
#         return abs_pos
#
# class Resampler(nn.Module):
#     """
#     A 2D perceiver-resampler network with one cross attention layers by
#         (grid_size**2) learnable queries and 2d sincos pos_emb
#     Outputs:
#         A tensor with the shape of (grid_size**2, embed_dim)
#     """
#
#     def __init__(
#             self,
#             num_queries,
#             embed_dim,
#             num_heads,
#             kv_dim=None,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6)
#     ):
#         super().__init__()
#         self.num_queries = num_queries
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#
#         self.pos_embed = nn.Parameter(
#             torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(self.num_queries, dtype=np.float32))).float().to(torch.float16)
#         ).requires_grad_(False)
#
#         self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
#         trunc_normal_(self.query, std=.02)
#
#         if kv_dim is not None and kv_dim != embed_dim:
#             self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
#         else:
#             self.kv_proj = nn.Identity()
#
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.ln_q = norm_layer(embed_dim)
#         self.ln_kv = norm_layer(embed_dim)
#
#         self.ln_post = norm_layer(embed_dim)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x, key_padding_mask=None):
#
#         x = self.kv_proj(x)
#         x = self.ln_kv(x).permute(1, 0, 2)
#         k = x.clone()
#
#         N = x.shape[1]
#         q = self.ln_q(self.query)
#
#         ori_type = x.dtype
#
#         self.attn.float()
#         out, weight = self.attn(
#             (self._repeat(q, N) + self.pos_embed.unsqueeze(1)).to(torch.float32),
#             k.to(torch.float32),
#             x.to(torch.float32),
#             key_padding_mask=~key_padding_mask)
#
#         # print("out: {}".format(out))
#         # print("weight: {}".format(weight))
#         # Check whether nan appears
#         if torch.isnan(out).any():
#             print("nan appears in resampler")
#             print("out: {}".format(out))
#             print("q: {}".format(q))
#             print("k: {}".format(k))
#             print("x: {}".format(x))
#             print("key_padding_mask: {}".format(key_padding_mask))
#             print("pos_embed: {}".format(self.pos_embed))
#             print("self._repeat(q, N) + self.pos_embed.unsqueeze(1): {}".format(self._repeat(q, N) + self.pos_embed.unsqueeze(1)))
#             print("q + self.pos_embed.unsqueeze(1): {}".format(q + self.pos_embed.unsqueeze(1)))
#
#         # print(key_padding_mask.dtype)
#         # print(out.dtype)
#         # out.to(torch.float16)
#         out = out.to(ori_type)
#         # print(out.dtype)
#         out = self.ln_post(out.permute(1, 0, 2))
#         return out
#
#     def _repeat(self, query, N: int):
#         return query.unsqueeze(1).repeat(1, N, 1)
#
#
# # def create_resampler(num_query_token=36, embed_dim=4096):
# #     attn_pool = Resampler(
# #             num_queries=num_query_token,
# #             embed_dim=embed_dim,
# #             num_heads=16,
# #             kv_dim=embed_dim,
# #         )
# #     return attn_pool
#


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, topk: int = 1, dim: int = -1) -> torch.Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        _, indices = torch.topk(y_soft, topk, dim=dim)
        # Randomly select one from topk and keep dim
        index = indices[..., [random.randint(0, topk-1)]]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def hard_softmax(logits: torch.Tensor, topk: int, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:

    y_soft = logits.softmax(dim)

    if hard:
        # Get the indices of the top-k elements
        topk_indices = logits.topk(topk, dim=dim, largest=True, sorted=False)[1]

        # Create a one-hot tensor for the top-k elements    else:
        y_hard = torch.zeros_like(logits).scatter_(dim, topk_indices, 1.0)        # Reparametrization trick.
        # Use the straight-through estimator for backpropagation    return ret, y_soft
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, y_soft

def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)

class TextFilterCosine(nn.Module): ##通过计算图像特征和文本特征之间的余弦相似度，选取与图像特征最相关的文本特征，用于后续处理。

    def __init__(self, pad_token_id, text_queries_num, temp=1.0, embed_dim=4096):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.text_queries_num = text_queries_num
        self.temp = temp

    def forward(self, image_features, text_embedding, attn_mask=None):
        #每个图像 token 和所有文本 token 之间的相似度
        similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_embedding.unsqueeze(0), dim=-1)

        ##对于每个文本 token，计算它与所有图像 token 的相似度的平均值，最终得到形状为 (num_text_tokens,) 的向量 similarity_matrix。
        ##masked_fill将被掩码的位置的相似度设置为负无穷
        if attn_mask is not None:
            attn_mask = attn_mask == False
            similarity_matrix = similarity_matrix.masked_fill(attn_mask.unsqueeze(0), float('-inf'))

            similarity_matrix = similarity_matrix.mean(dim=0)
        else:
            similarity_matrix = similarity_matrix.mean(dim=0)

        # Sort the probabilities in descending order and get the corresponding indices
        sorted_probs, sorted_indices = torch.sort(similarity_matrix, descending=True)

        selected_indices = sorted_indices[:self.text_queries_num]

        new_text_embedding = torch.zeros((self.text_queries_num, text_embedding.shape[-1]), device=text_embedding.device, dtype=text_embedding.dtype)
        new_text_embedding[:len(selected_indices)] = text_embedding[selected_indices]

        return new_text_embedding

class Router(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, num_patches):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = 577
        self.num_tokens = 577

        # original 32
        # self.text_queries_num = getattr(config, 'filtered_text_num', 32)

        # self.num_tokens += self.text_queries_num

        self.topk = 433  #576*0.75+1
        self.router = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.num_experts)
                    )
        ##self.voter 通过训练学习不同 token 的相对重要性，从而更智能地对专家分配权重。
        self.voter = nn.Parameter(torch.randn(self.num_tokens, 1)) # (sequence_length, 1)

        self.norm = nn.LayerNorm(self.num_experts)
        # ###通过计算图像特征和文本特征之间的余弦相似度，选取与图像特征最相关的文本特征，用于后续处理。
        # self.text_filter = TextFilterCosine(pad_token_id=config.pad_token_id, text_queries_num=self.text_queries_num, temp=1.0, embed_dim=self.hidden_dim)

        ##将输入序列中的特征进行全局融合，捕获不同位置之间的依赖关系
        self.merger = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, dim_feedforward=config.hidden_size)


        nn.init.kaiming_normal_(self.router[0].weight)
        nn.init.kaiming_normal_(self.router[2].weight)
        nn.init.normal_(self.voter)
    # Hook function to print gradients
    def print_grad(self, module, grad_input, grad_output):
        print("Gradient input: ", grad_input)
        print("Gradient output: ", grad_output)

    def forward(self, hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, label_hidden_states: torch.Tensor, label_mask: torch.Tensor):
        """
        hidden_states shape: (B, 1+n_experts, hidden_dim)
        text_hidden_states: (B, length, 4*hidden_dim)
        """

        text_queries_num = text_hidden_states.size(1)
        image_token_nums = hidden_states.size(1)
        num_tokens = image_token_nums + text_queries_num
        B = hidden_states.shape[0]
        # all cat
        new_hidden_states = torch.zeros((hidden_states.shape[0], num_tokens, self.hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        start = 0
        new_hidden_states[:, start:start+image_token_nums] = hidden_states
        start = start+image_token_nums
        new_hidden_states[:, start:start+text_queries_num] = text_hidden_states

        hidden_states = new_hidden_states.contiguous()

        hidden_states = self.merger(hidden_states)
        router_logits = self.router(hidden_states)  # router_logits: (n, sequence_length, n_experts)
        router_logits = router_logits.permute(0, 2, 1)  # router_logits: (n, n_experts, sequence_length)
        router_logits = torch.einsum('ijk,kl->ijl', router_logits, self.voter)  # router_logits: (n, n_experts, 1)
        router_logits = router_logits.permute(0, 2, 1)  # router_logits: (n, 1, n_experts)

        router_logits = torch.mean(router_logits, dim=0)  # router_logits: (1, n_experts)
        router_logits = self.norm(router_logits)

        if self.training:
            # gumbel softmax
            # router_probs = F.gumbel_softmax(router_logits, tau=1, hard=True) # (1, n_experts)
            router_probs, soft_router_probs = hard_softmax(router_logits, self.topk, tau=1, hard=True)
        else:
            # 推理时选择 top_k 个专家
            top_k_values, top_k_indices = torch.topk(router_logits, k=self.topk, dim=1)  # 获取 top_k 的值和索引
            router_probs = torch.zeros_like(router_logits)  # 初始化为零
            router_probs[0, top_k_indices] = top_k_values  # 将 top_k 的值填入对应位置
            # 对 top_k 的权重进行归一化
            router_probs /= torch.sum(router_probs, dim=1, keepdim=True)

        final_hidden_states = router_probs * hidden_states
        # 检查 dim=1 上哪些行是非零行
        non_zero_mask = torch.any(final_hidden_states != 0, dim=2)  # [B, L]，检查每行是否有非零值

        # 根据布尔索引筛选非零行
        filtered_hidden_states = [final_hidden_states[b, non_zero_mask[b]] for b in range(B)]
        final_hidden_states = torch.stack(filtered_hidden_states)  # 拼接回一个张量

        return final_hidden_states, router_logits


def build_router(config, num_patches):
    return Router(config, num_patches)
