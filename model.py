"""
HSTU (Hierarchical Sequential Transduction Unit) Model for RL-based Recommendation

Adapted from Benchmark-v2 implementation. Provides GBPO-compatible interface
(forward, generate_candidates, predict, apply_layer_mask) so that all 9 RL
algorithms (GRPO, GBPO, GUPO, GSPO, GPPO, DUAL_PPO, SAPO, DAPO, DPO) work
out of the box.

Key design: HSTU is a single-step policy (L=1). The "code" is the item ID itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSTUBlock(nn.Module):
    """
    Single HSTU block with ReLU-activated attention, relative positional bias,
    and gating mechanism. Extracted from Benchmark-v2.
    """

    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super().__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_units, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_units, eps=1e-6)

        # Combined uvqk projection
        self.uvqk_proj = nn.Linear(hidden_units, hidden_units * 4)

        # Output projection
        self.o_proj = nn.Linear(hidden_units, hidden_units)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

        # Relative positional bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * maxlen - 1))
        nn.init.normal_(self.rel_pos_bias, mean=0, std=0.02)

        self.dropout = nn.Dropout(dropout_rate)
        self.maxlen = maxlen

    def forward(self, x, attention_mask, timeline_mask):
        """
        Args:
            x: [B, T, D] - input sequences
            attention_mask: [T, T] - causal mask (True=allowed)
            timeline_mask: [B, T] - padding mask (True=padding)
        Returns:
            x: [B, T, D]
        """
        B, T, D = x.shape

        # Pre-norm
        x_norm = self.ln1(x)

        # Combined projection
        uvqk = self.uvqk_proj(x_norm)  # [B, T, 4D]
        u, v, q, k = uvqk.chunk(4, dim=-1)

        # Multi-head reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]

        # Relative positional bias
        rel_bias = self._get_relative_position_bias(T)
        attn_scores = attn_scores + rel_bias.unsqueeze(0).unsqueeze(0)

        # Scale
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # ReLU activation (not softmax)
        attn_weights = F.relu(attn_scores)

        # Causal mask
        causal_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(~causal_mask, 0.0)

        # Padding mask
        pad_mask = timeline_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(pad_mask, 0.0)

        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D/H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Gating with u
        u = u.contiguous().view(B, T, D)
        attn_output_norm = F.layer_norm(attn_output, normalized_shape=[D], eps=1e-6)
        gated_output = u * attn_output_norm

        # Output projection + residual
        attn_output = self.o_proj(gated_output)
        attn_output = self.dropout(attn_output)
        x = x + attn_output

        # FFN + residual
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output

        return x

    def _get_relative_position_bias(self, seq_len):
        positions = torch.arange(seq_len, device=self.rel_pos_bias.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos = torch.clamp(rel_pos, -self.maxlen + 1, self.maxlen - 1)
        rel_pos = rel_pos + self.maxlen - 1
        bias = torch.index_select(self.rel_pos_bias, 0, rel_pos.flatten())
        return bias.view(seq_len, seq_len)


class HSTUForRL(nn.Module):
    """
    HSTU model with GBPO-compatible interface for SFT and RL training.

    Implements the same public API as TIGER:
    - forward(input_seq, attention_mask, target_item_ids, labels, K)
    - generate_candidates(input_seq, attention_mask, num_candidates, strategy, temperature, use_trie)
    - predict(input_seq, attention_mask, beam_size, topk)
    - apply_layer_mask(logits)

    Since HSTU selects items directly (not via codes), L=1 and "codes" = item IDs.
    """

    def __init__(self, num_items, hidden_size=128, num_heads=2, num_layers=2,
                 dropout=0.2, max_seq_len=20):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        # No trie for HSTU (single-step item selection)
        self.use_trie_rl = "no_trie"

        # Embeddings (item 0 = padding)
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout)

        # HSTU blocks
        self.hstu_blocks = nn.ModuleList([
            HSTUBlock(
                hidden_units=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout,
                maxlen=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Init embeddings
        nn.init.xavier_normal_(self.item_emb.weight.data)
        nn.init.xavier_normal_(self.pos_emb.weight.data)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

    def apply_layer_mask(self, logits, layer_idx=None):
        """Mask padding item (index 0) to -inf. Compatible with RL/utils.py interface."""
        if logits.ndim == 3:  # (N, L, V)
            logits = logits.clone()
            logits[:, :, 0] = float('-inf')
        elif logits.ndim == 2:  # (N, V)
            logits = logits.clone()
            logits[:, 0] = float('-inf')
        return logits

    def encode(self, input_seq, attention_mask):
        """
        Encode user history into a single hidden vector.

        Args:
            input_seq: [B, T] item ID sequences (0-padded)
            attention_mask: [B, T] True = valid, False = padding
        Returns:
            hidden: [B, D]
        """
        batch_size, seq_len = input_seq.size()

        # Truncate if needed
        if seq_len > self.max_seq_len:
            input_seq = input_seq[:, -self.max_seq_len:]
            attention_mask = attention_mask[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        # Item embeddings scaled by sqrt(D)
        seqs = self.item_emb(input_seq) * (self.hidden_size ** 0.5)

        # Positional embeddings
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch_size, -1)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_seq.device))
        # Padding mask (True = padding)
        padding_mask = ~attention_mask

        # HSTU blocks
        for block in self.hstu_blocks:
            seqs = block(seqs, causal_mask, padding_mask)

        # Output projection
        seqs = self.out_proj(seqs)

        # Extract last valid position
        hist_len = attention_mask.sum(dim=1)
        hist_len = torch.where(hist_len > 0, hist_len, torch.ones_like(hist_len))
        last_indices = (hist_len - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=input_seq.device)
        hidden = seqs[batch_indices, last_indices]  # [B, D]

        return hidden

    def forward(self, input_seq, attention_mask, target_item_ids=None, labels=None, K=1):
        """
        Unified forward for SFT and RL.

        Args:
            input_seq: [B, T]
            attention_mask: [B, T]
            target_item_ids: [B] or [B, 1] — target items for SFT
            labels: [B*K, 1] — item IDs as "codes" for RL logit recomputation
            K: number of candidates per sample (1 for SFT)
        Returns:
            logits: [B*K, 1, V] — full vocabulary logits
            loss: CE loss (SFT only) or None
        """
        hidden = self.encode(input_seq, attention_mask)  # [B, D]
        B, D = hidden.shape
        V = self.num_items + 1  # vocab size including padding

        if K > 1:
            # Expand for RL: B -> B*K
            hidden = hidden.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

        # Full vocabulary logits via dot product with item embeddings
        all_item_emb = self.item_emb.weight  # [V, D]
        logits_2d = torch.matmul(hidden, all_item_emb.T)  # [B*K, V]
        logits = logits_2d.unsqueeze(1)  # [B*K, 1, V] — L=1 dimension

        # Compute loss for SFT mode
        loss = None
        if target_item_ids is not None and K == 1:
            if target_item_ids.ndim > 1:
                target_item_ids = target_item_ids.squeeze(-1)
            loss = F.cross_entropy(logits_2d, target_item_ids)

        return logits, loss

    @torch.no_grad()
    def generate_candidates(self, input_seq, attention_mask, num_candidates,
                            strategy="rollout", temperature=1.0, use_trie=None):
        """
        Generate K candidate items per input sequence.

        Returns:
            sampled_items: [B*K] — item IDs
            codes: [B*K, 1] — same as item IDs (L=1)
            seq_log_probs: [B*K] — log prob of selected item
            token_log_probs: [B*K, 1] — same with L dimension
        """
        self.eval()
        hidden = self.encode(input_seq, attention_mask)  # [B, D]
        B, D = hidden.shape
        K = num_candidates

        # Full vocabulary logits
        all_item_emb = self.item_emb.weight  # [V, D]
        logits = torch.matmul(hidden, all_item_emb.T)  # [B, V]
        logits[:, 0] = float('-inf')  # mask padding

        if strategy == "rollout":
            probs = F.softmax(logits / temperature, dim=-1)  # [B, V]
            sampled = torch.multinomial(probs, K, replacement=True)  # [B, K]
        else:  # beam / top-K
            _, sampled = torch.topk(logits, K, dim=-1)  # [B, K]

        sampled_items = sampled.reshape(B * K)  # [B*K]
        codes = sampled_items.unsqueeze(1)  # [B*K, 1]

        # Compute log probs
        actual_temp = temperature if strategy == "rollout" else 1.0
        log_probs_full = F.log_softmax(logits / actual_temp, dim=-1)  # [B, V]
        log_probs_expanded = log_probs_full.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)  # [B*K, V]
        token_log_probs = torch.gather(log_probs_expanded, 1, codes)  # [B*K, 1]
        seq_log_probs = token_log_probs.squeeze(1)  # [B*K]

        return sampled_items, codes, seq_log_probs, token_log_probs

    @torch.no_grad()
    def predict(self, input_seq, attention_mask, beam_size=20, topk=10):
        """
        Predict top-K items for evaluation.

        Returns:
            pred_items: [B, topk]
            validity_rate: 1.0 (all items are valid by construction)
        """
        self.eval()
        hidden = self.encode(input_seq, attention_mask)  # [B, D]

        all_item_emb = self.item_emb.weight  # [V, D]
        logits = torch.matmul(hidden, all_item_emb.T)  # [B, V]
        logits[:, 0] = float('-inf')  # mask padding

        _, pred_items = torch.topk(logits, topk, dim=-1)  # [B, topk]
        return pred_items, 1.0
