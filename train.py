import os
import math
import time
import inspect
from click import Option
from numpy import dtype
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import sys
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training-v1.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Script Started")

# -----------------------------------------------------------------------------
@dataclass
class SmolLM2Config:
    hidden_size: int = 768             # Reduced from 768
    intermediate_size: int = 1536       # Typically 2x hidden_size
    num_hidden_layers: int = 30         # Reduced from 30
    num_attention_heads: int = 12       # Must evenly divide hidden_size (256/8=32)
    num_key_value_heads: int = 3       # Can remain, if used in your design
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048 # Reduced from 2048
    initializer_range: float = 0.02    # You may adjust this
    rms_norm_eps: float = 1e-5
    vocab_size: int = 50257
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = True
    use_cache: bool = True
    compression_ratio: int = 8
    num_experts: int = 4               # Reduced from 8
    num_shared_experts: int = 1
    top_k: int = 2

# -----------------------------------------------------------------------------
def save_model(model, optimizer, loss, output_dir='saved_models'):
    from datetime import datetime
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smollm2-135_{timestamp}.pt"
    filepath = os.path.join(output_dir, filename)
    
    save_dict = {
        'model_state_dict': model.state_dict(),  # Fixed key name
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'loss': loss
    }
    torch.save(save_dict, filepath)
    logging.info(f"Model Saved to {filepath}")
    return filepath

# -----------------------------------------------------------------------------
def apply_rotary_pos_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    sin, cos = rotary_emb[..., :head_dim // 2], rotary_emb[..., head_dim // 2:]
    rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated_x

# -----------------------------------------------------------------------------
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

# -----------------------------------------------------------------------------
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= 0:
            logging.warning(f"Invalid seq_len: {seq_len}, setting to 1")
            seq_len = 1
        if seq_len > self.max_position_embeddings:
            logging.warning(f"seq_len {seq_len} exceeds max_position_embeddings {self.max_position_embeddings}, truncating")
            seq_len = self.max_position_embeddings

        positions = torch.arange(seq_len, device=device)
        sincos = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        # Rearranged so that seq_len is in dimension 2
        return emb[None, None, :, :]

    def apply_rotary_emb(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        # Wrapper to use the standalone function
        return apply_rotary_pos_emb(x, rotary_emb)

# -----------------------------------------------------------------------------
class DeepSeekExpertLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# -----------------------------------------------------------------------------
class DeepSeekMoE(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(config)
            for _ in range(self.num_shared_experts)
        ])
        
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(config)
            for _ in range(self.num_routed_experts)
        ])
        
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts
        
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        combined_output = torch.zeros_like(x)
        
        # Iterate over top-k experts for each token position
        for k in range(self.top_k):
            expert_indices = indices[..., k]  # shape: (batch, seq_len)
            expert_scores = scores[..., k]     # shape: (batch, seq_len)
            
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    # Multiply by corresponding expert score; unsqueeze to match dimensions if needed
                    combined_output[mask] += expert_output * expert_scores[mask].unsqueeze(-1)
        final_output = shared_output + combined_output
        return final_output

    def update_bias_terms(self, expert_load):
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load
        update_rate = 0.1 * torch.abs(load_diff)
        self.routing_bias.data -= update_rate * load_diff

# -----------------------------------------------------------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        # Pass the given config instance instead of the class itself
        self.moe = DeepSeekMoE(config)
        
    def forward(self, x):
        return self.moe(x)

# -----------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=0.0, is_casual=False):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot product

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e9)  # Mask out invalid positions

    if is_casual:
        mask = torch.tril(torch.ones_like(scores)).to(scores.device)
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_probs = F.softmax(scores, dim=-1)  # Apply softmax

    if dropout > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout)

    attn_output = torch.matmul(attn_probs, v)  # Weighted sum of values
    return attn_output

# -----------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads  # Now 768 // 12 = 64
        self.latent_dim = self.hidden_size // config.compression_ratio  # 768 // 8 = 96
        
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        
        # For keys and queries, output dimension: num_heads * (head_dim // 2) = 12 * (64 // 2) = 12 * 32 = 384
        self.k_proj_u = nn.Linear(self.latent_dim, self.num_heads * (self.head_dim // 2), bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.num_heads * (self.head_dim // 2), bias=False)
        
        # For values, output dimension: num_heads * head_dim = 12 * 64 = 768
        self.v_proj_u = nn.Linear(self.latent_dim, self.num_heads * self.head_dim, bias=False)
        
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim // 2)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                pass_key_value: Optional[Tuple[torch.Tensor]] = None,
                rotary_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        kv_d = self.kv_proj_d(hidden_states)
        q_d = self.q_proj_d(hidden_states)
        
        k_proj_2 = self.k_proj_u(kv_d)
        q_proj_2 = self.q_proj_u(q_d)
        v = self.v_proj_u(kv_d)  # Use value projection
        
        k_rope_2 = self.rope_k(hidden_states)
        q_rope_2 = self.rope_q(q_d)
        
        # Reshape projections to split heads.
        # Expected shape: (batch, seq_len, num_heads, head_dim//2)
        k_proj_2 = k_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_proj_2 = q_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        k_rope_2 = k_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_rope_2 = q_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        
        # Transpose to (batch, num_heads, seq_len, head_dim//2) for rotary application.
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        
        # Generate rotary embeddings. This returns a tensor of shape (1, 1, seq_len, head_dim//2).
        rotary_emb_out = self.rotary_emb(seq_len, hidden_states.device)
        
        # Apply rotary embeddings.
        k_rope_2 = self.rotary_emb.apply_rotary_emb(k_rope_2, rotary_emb_out)
        q_rope_2 = self.rotary_emb.apply_rotary_emb(q_rope_2, rotary_emb_out)
        
        # Transpose back to (batch, seq_len, num_heads, head_dim//2)
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        
        # Concatenate the corresponding projections.
        k = torch.cat([k_proj_2, k_rope_2], dim=-1)
        q = torch.cat([q_proj_2, q_rope_2], dim=-1)
        
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # shape: (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout=0.0, is_casual=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

# -----------------------------------------------------------------------------
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                pass_key_value: Optional[Tuple[torch.Tensor]] = None,
                rotary_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            pass_key_value=pass_key_value,
            rotary_emb=rotary_emb
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# -----------------------------------------------------------------------------
class LlamaModel(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,  # Use num_attention_heads for head dimension
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        nn.init.normal_(self.embed_tokens.weight, std=config.initializer_range)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        # Debug logging
        logging.debug(f"Input shape: {input_ids.shape}, Hidden states shape: {hidden_states.shape}")
        logging.debug(f"Max position embeddings: {self.config.max_position_embeddings}")
        
        # Ensure seq_len is valid
        seq_len = hidden_states.shape[1]
        if seq_len > self.config.max_position_embeddings:
            seq_len = self.config.max_position_embeddings
            logging.warning(f"Sequence length {hidden_states.shape[1]} exceeds max_position_embeddings, truncating to {seq_len}")
        
        # Obtain rotary embeddings using the modified forward signature
        rotary_emb_out = self.rotary_emb(seq_len, hidden_states.device)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                positional_ids=positional_ids,
                rotary_emb=rotary_emb_out
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

# -----------------------------------------------------------------------------
class LlamaForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        
        return logits

# -----------------------------------------------------------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # logits: shape (batch_size, vocab_size)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value
    
    return logits

# -----------------------------------------------------------------------------
def generate_text(model, prompt, max_new_tokens=50, temperature=1.2, top_k=50, top_p=0.95):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            # Filter logits with top-k and/or top-p
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = enc.decode(input_ids[0].tolist())
    model.train()
    return generated_text

# -----------------------------------------------------------------------------
# Device selection
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
num_return_sequences = 5
max_length = 30

import tiktoken

# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('Lord-of-the-Rings.txt', 'r', encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        logging.info(f'loaded {len(self.tokens)} tokens')
        logging.info(f'1 epoch = {len(self.tokens) // (B*T)} batches')
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")
model = LlamaForCausalLM(SmolLM2Config())
model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 2000
max_steps = 10000

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine schedule
    return min_lr + coeff * (max_lr - min_lr)

train_loader = DataLoaderLite(B=1, T=768)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps + 1):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    
    # Optional: Update MoE routing bias (currently commented out)
    # with torch.no_grad():
    #     expert_load = torch.zeros(model.model.layers[0].mlp.moe.num_routed_experts, device=device)
    #     for k in range(model.model.layers[0].mlp.moe.top_k):
    #         routing_logits = model.model.layers[0].mlp.moe.router(x) + model.model.layers[0].mlp.moe.routing_bias
    #         routing_probs = torch.sigmoid(routing_logits)
    #         _, indices = torch.topk(routing_probs, model.model.layers[0].mlp.moe.top_k, dim=-1)
    #         for i in range(model.model.layers[0].mlp.moe.num_routed_experts):
    #             expert_load[i] += (indices[..., k] == i).sum()
    #     expert_load = expert_load / (x.size(0) * x.size(1) * model.model.layers[0].mlp.moe.top_k)
    #     model.model.layers[0].mlp.moe.update_bias_terms(expert_load)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(x, labels=y)  # Pass y as labels
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    logging.info(f'step {step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.2f}')
    if step % 1000 == 0:
        saved_path = save_model(model, optimizer, loss)
        logging.info(f"Checkpoint at step {step}")
        prompt = "Through Rohan over fen and field where the long grass grows"
        generated_text = generate_text(model, prompt)
        logging.info(f"Generated text at step {step}:\n{generated_text}")
        
saved_path = save_model(model, optimizer, loss)
logging.info(f"Final Loss: {loss.item()}")
print(loss)
