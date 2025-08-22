#!/usr/bin/env python3
# snapshot_gpt.py
#
# Tiny femtoGPT-like transformer + snapshot-embedding + checkpoint-attention + retro KD
#
# Usage (example):
#   python main.py --datafile /path/to/corpus.txt --device cuda --total_steps 20000
#   python3 -m main --datafile /shakespeare.txt --device cpu --total_steps 20000
#
# Requirements: torch. (If you have GPU, use a CUDA PyTorch build.)

import argparse, math, os, random, time, copy, sys
from collections import deque
import numpy as np  # Added missing import
import torch
import torch.nn as nn
import torch.nn.functional as F

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# -----------------------
# Config / defaults
# -----------------------
DEFAULTS = {
    "vocab_size": 256,       # byte-level
    "block_size": 64,        # context length (reduced from 128)
    "d_model": 256,          # embedding dim (reduced from 384)
    "n_layers": 4,           # reduced from 6
    "n_heads": 4,            # reduced from 6
    "flag_dim": 32,          # reduced from 64
    "max_snapshots": 999999,      # reduced from 5
    "snapshot_interval": 1000,   # steps between snapshotting (reduced from 2000)
    "anchors_per_snapshot": 512, # reduced from 2048
    "batch_size": 32,        # reduced from 64
    "lr": 3e-4,
    "warmup_steps": 500,     # reduced from 1000
    "total_steps": 5000,     # reduced from 20000
    "retro_prob": 0.1,       # probability of doing a retro KD step per step
    "kd_tau": 2.0,
    "kd_weight": 0.7,
    "attn_entropy_reg": 0.0,
    "topk": None,               # top-k snapshots to evaluate per retro/inference
}

# -----------------------
# Utilities: byte tokenizer
# -----------------------
class ByteTextDataset:
    def __init__(self, path, block_size):
        with open(path, "rb") as f:
            data = f.read()

        self.data = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).long()
        self.vocab_size = 256
        self.block_size = block_size

    def get_batch(self, batch_size):
        # random contiguous sequences
        max_start = len(self.data) - self.block_size - 1
        starts = torch.randint(0, max_start+1, (batch_size,))
        xb = torch.stack([self.data[s:s+self.block_size] for s in starts], dim=0)
        yb = torch.stack([self.data[s+1:s+self.block_size+1] for s in starts], dim=0)
        return xb, yb

# -----------------------
# Minimal GPT-like model
# -----------------------
class SimpleBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        att_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + att_out)
        x = self.ln2(x + self.mlp(x))
        return x

class SnapshotGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_layers, n_heads, flag_dim, n_out=None):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([SimpleBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # flag projection (we add to token embeddings)
        self.flag_proj = nn.Linear(flag_dim, d_model)
        # query projection for checkpoint-attention diagnostic (pooled -> query)
        # Fixed: use consistent key dimension
        self.key_dim = d_model // 2
        self.q_proj = nn.Linear(d_model, self.key_dim)

    def forward(self, idx, flag_vec):
        # idx: (B,T) int tokens; flag_vec: (B, flag_dim)
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        flag = self.flag_proj(flag_vec).unsqueeze(1)    # (B,1,d)
        x = x + flag
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        pooled = x.mean(dim=1)   # (B,d)
        logits = self.head(x)    # (B,T,vocab) - we will compute loss with shift
        q = self.q_proj(pooled)  # (B, key_dim)
        return logits, pooled, q

# -----------------------
# Snapshot bank
# -----------------------
class Snapshot:
    def __init__(self, id_str, state_dict, emb, key, anchors, cached_logits=None):
        # state_dict: parameters saved (cpu tensors)
        # emb: CPU tensor (flag embedding)
        # key: CPU tensor (key used for attention)
        self.id_str = id_str
        self.state_dict = {k: v.clone().cpu() for k, v in state_dict.items()}
        self.emb = emb.clone().cpu()
        self.key = key.clone().cpu()
        self.anchors = [a.clone().cpu() for a in anchors]  # list of (1, T) tensors
        self.cached_logits = cached_logits  # optional dict idx->logits (for anchors)

class SnapshotBank:
    def __init__(self, max_keep):
        self.max_keep = max_keep
        self.bank = deque(maxlen=max_keep)

    def add(self, model: SnapshotGPT, flag_embedding: torch.Tensor, key_vec: torch.Tensor, anchors):
        sid = f"snap{len(self.bank)+1}_{int(time.time())}"
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        snap = Snapshot(sid, sd, flag_embedding.detach().cpu(), key_vec.detach().cpu(), anchors)
        self.bank.append(snap)
        return snap

    def list_keys(self, device):
        if len(self.bank) == 0:
            return None
        return torch.stack([s.key.to(device) for s in self.bank], dim=0)  # (S, key_dim)

    def list_embs(self, device):
        if len(self.bank) == 0:
            return None
        return torch.stack([s.emb.to(device) for s in self.bank], dim=0)  # (S, flag_dim)

    def sample(self):
        return random.choice(list(self.bank)) if len(self.bank) else None

# -----------------------
# Checkpoint-attention module
# -----------------------
class CheckpointAttention(nn.Module):
    def __init__(self, query_dim, key_dim, flag_dim):
        super().__init__()
        # Fixed: ensure query_dim matches the actual query dimension from model
        self.q_proj = nn.Linear(query_dim, key_dim)
        self.beta_proj = nn.Linear(query_dim, 1)  # gating scalar
        self.fallback = nn.Parameter(torch.zeros(flag_dim))

    def forward(self, pooled_h, snapshot_keys, snapshot_embs, topk=2):
        # pooled_h: (B, qdim)
        q = self.q_proj(pooled_h)  # (B, key_dim)
        if snapshot_keys is None or snapshot_embs is None or snapshot_keys.shape[0] == 0:
            B = pooled_h.shape[0]
            return self.fallback.unsqueeze(0).expand(B, -1), torch.zeros(B,0, device=pooled_h.device)
        # scores
        scores = torch.matmul(q, snapshot_keys.t()) / math.sqrt(q.size(-1))  # (B, S)
        if topk is None or topk >= scores.shape[-1]:
            alpha = F.softmax(scores, dim=-1)  # (B,S)
            emb_mix = alpha @ snapshot_embs  # (B, flag_dim)
        else:
            # top-k selection (differentiable soft selection by masked softmax)
            vals, idx = scores.topk(topk, dim=-1)
            # create masked -inf for others
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, idx, vals)
            alpha = F.softmax(mask, dim=-1)
            emb_mix = alpha @ snapshot_embs
        beta = torch.sigmoid(self.beta_proj(pooled_h))  # (B,1)
        return emb_mix * beta, alpha

# -----------------------
# Reservoir sampler for anchors
# -----------------------
class ReservoirAnchors:
    def __init__(self, k):
        self.k = k
        self.pool = []
        self.n_seen = 0

    def add_batch(self, batch_x):  # batch_x: (B,T) cpu tensor
        for i in range(batch_x.shape[0]):
            self.n_seen += 1
            if len(self.pool) < self.k:
                self.pool.append(batch_x[i:i+1].cpu().clone())
            else:
                j = random.randint(0, self.n_seen-1)
                if j < self.k:
                    self.pool[j] = batch_x[i:i+1].cpu().clone()

    def get_anchors(self):
        return list(self.pool)

# -----------------------
# KD helper
# -----------------------
def kd_loss(student_logits, teacher_logits, tau):
    # student_logits, teacher_logits: (B, T, V)
    # KL(soft_teacher || soft_student) averaged over tokens
    s = F.log_softmax(student_logits / tau, dim=-1)
    t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (tau * tau)

# -----------------------
# Training orchestration
# -----------------------
def run_training(args):
    # config
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))

    # Set up logging
    log_filename = f"training_log_{int(time.time())}.txt"
    logger = TeeLogger(log_filename)
    sys.stdout = logger
    
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logging to: {log_filename}")
    print(f"Config: {cfg}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print("Using device:", device)

    # dataset
    ds = ByteTextDataset(args.datafile, cfg["block_size"])
    model = SnapshotGPT(ds.vocab_size, cfg["block_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["flag_dim"]).to(device)

    # flag table: slot 0 reserved for "current"
    max_slots = 256
    flag_table = nn.Embedding(max_slots, cfg["flag_dim"]).to(device)
    nn.init.normal_(flag_table.weight, std=0.02)
    with torch.no_grad():
        flag_table.weight[0].zero_()  # current flag near-zero

    # Fixed: use correct dimensions for checkpoint attention
    cp_att = CheckpointAttention(query_dim=cfg["d_model"], key_dim=model.key_dim, flag_dim=cfg["flag_dim"]).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(flag_table.parameters()) + list(cp_att.parameters()), lr=cfg["lr"], weight_decay=1e-2)

    snap_bank = SnapshotBank(cfg["max_snapshots"])
    anchor_reservoir = ReservoirAnchors(cfg["anchors_per_snapshot"])

    step = 0
    lr = cfg["lr"]

    def snapshot_and_store():
        # create new snapshot:
        # - pick a free flag slot index (simple circular assignment)
        slot_idx = (len(snap_bank.bank) % (max_slots-1)) + 1
        emb_vec = flag_table.weight[slot_idx].detach().cpu().clone()
        # compute key by passing some anchors through current model pooled (use reservoir pool)
        anchors = anchor_reservoir.get_anchors()
        # if anchors empty, create a few by sampling the dataset
        if len(anchors) == 0:
            xb, _ = ds.get_batch(min(128, cfg["batch_size"]))
            anchors = [xb[i:i+1].cpu() for i in range(min(128, xb.shape[0]))]
        # Fixed: compute key using the projected query dimension, not raw pooled
        model.eval()
        with torch.no_grad():
            q_list = []
            for a in anchors:
                a = a.to(device)
                _, pooled, q = model(a, torch.zeros(1, cfg["flag_dim"], device=device))
                q_list.append(q.cpu())  # Use q instead of pooled
            key_vec = torch.cat(q_list, dim=0).mean(dim=0).cpu()
        snap = snap_bank.add(model, emb_vec, key_vec, anchors)
        print(f"[snap] added {snap.id_str}; bank_size={len(snap_bank.bank)}")

    # optionally pre-seed with initial snapshot
    snapshot_and_store()

    # training loop
    while step < cfg["total_steps"]:
        model.train()
        xb, yb = ds.get_batch(cfg["batch_size"])
        xb = xb.to(device); yb = yb.to(device)

        # current-mode (CE)
        flag_vec = flag_table(torch.zeros(xb.size(0), dtype=torch.long, device=device))
        logits, pooled, q = model(xb, flag_vec)  # logits: (B,T,V)
        # next-token CE: logits[:, :-1, :] vs targets[:, :-1]
        loss_cur = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        loss = loss_cur

        # add this batch into reservoir for future snapshots
        anchor_reservoir.add_batch(xb.detach().cpu())

        do_retro = (len(snap_bank.bank) > 0) and (random.random() < cfg["retro_prob"])
        if do_retro:
            snap = snap_bank.sample()
            # sample an anchor from snap.anchors
            if len(snap.anchors) > 0:
                a_idx = random.randrange(len(snap.anchors))
                anchor_x = snap.anchors[a_idx].to(device)  # shape (1,T)
                # teacher model: instantiate from state_dict and eval
                teacher = SnapshotGPT(ds.vocab_size, cfg["block_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["flag_dim"]).to(device)
                teacher.load_state_dict({k: v.to(device) for k,v in snap.state_dict.items()})
                teacher.eval()
                with torch.no_grad():
                    # teacher uses its snapshot embedding
                    t_emb = snap.emb.to(device).unsqueeze(0).expand(anchor_x.size(0), -1)
                    t_logits, t_pooled, _ = teacher(anchor_x, t_emb)  # (1,T,V)

                # student: compute mixed embedding using cp_att (we'd like alpha to focus on this snapshot)
                with torch.no_grad():
                    # get keys/embs tensor
                    keys = snap_bank.list_keys(device)
                    embs = snap_bank.list_embs(device)
                # produce pooled from student with zero-flag to get a query
                _, pooled_s, _ = model(anchor_x, torch.zeros(anchor_x.size(0), cfg["flag_dim"], device=device))
                mix_emb, alpha = cp_att(pooled_s, keys, embs, topk=cfg["topk"])  # (B,flag_dim), (B,S)
                # student logits conditioned on mix_emb
                s_logits, _, _ = model(anchor_x, mix_emb)
                loss_kd = kd_loss(s_logits, t_logits, cfg["kd_tau"])
                # optionally nudge alpha to point to the sampled snapshot (supervision)
                # build alpha target: one-hot pointing to snap index
                snap_index = list(snap_bank.bank).index(snap)
                alpha_target = torch.zeros_like(alpha); alpha_target[:, snap_index] = 1.0
                loss_att = F.mse_loss(alpha, alpha_target) * 0.2
                loss = loss + cfg["kd_weight"] * loss_kd + loss_att

        # backprop step
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # simple LR warmup
        step += 1
        if step < cfg["warmup_steps"]:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * (step / cfg["warmup_steps"])
        opt.step()

        if step % 100 == 0:
            print(f"step {step} loss_cur {loss_cur.item():.4f} total_loss {loss.item():.4f} bank_size {len(snap_bank.bank)}")
            
        # Generate samples every 250 steps
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                # Sample with current flag (zero flag)
                seed_text = "To be or not to be"
                seed_bytes = torch.tensor([ord(c) for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
                flag_curr = flag_table(torch.zeros(1, dtype=torch.long, device=device))
                
                generated_curr = seed_bytes.clone()
                for _ in range(100):  # Generate 100 characters
                    if generated_curr.shape[1] >= cfg["block_size"]:
                        # Use last block_size tokens
                        input_seq = generated_curr[:, -cfg["block_size"]:]
                    else:
                        input_seq = generated_curr
                    
                    logits, _, _ = model(input_seq, flag_curr)
                    # Sample from the distribution (temperature = 0.8)
                    probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    generated_curr = torch.cat([generated_curr, next_token], dim=1)
                
                # Convert to text (handle non-printable bytes)
                generated_bytes = generated_curr[0].cpu().numpy()
                try:
                    generated_text = ''.join([chr(b) if 32 <= b <= 126 else f'[{b}]' for b in generated_bytes])
                    print(f"[CURRENT] Generated: {generated_text}")
                except:
                    print(f"[CURRENT] Generated bytes: {generated_bytes.tolist()[:50]}...")
                
                # If we have snapshots, also try generating with mixed flag
                if len(snap_bank.bank) > 0:
                    # Get mixed embedding using checkpoint attention
                    keys = snap_bank.list_keys(device)
                    embs = snap_bank.list_embs(device)
                    if keys is not None and embs is not None:
                        _, pooled_curr, _ = model(seed_bytes, torch.zeros(1, cfg["flag_dim"], device=device))
                        mix_emb, alpha = cp_att(pooled_curr, keys, embs, topk=cfg["topk"])
                        
                        generated_mix = seed_bytes.clone()
                        for _ in range(100):
                            if generated_mix.shape[1] >= cfg["block_size"]:
                                input_seq = generated_mix[:, -cfg["block_size"]:]
                            else:
                                input_seq = generated_mix
                            
                            logits, _, _ = model(input_seq, mix_emb)
                            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                            generated_mix = torch.cat([generated_mix, next_token], dim=1)
                        
                        generated_bytes_mix = generated_mix[0].cpu().numpy()
                        try:
                            generated_text_mix = ''.join([chr(b) if 32 <= b <= 126 else f'[{b}]' for b in generated_bytes_mix])
                            print(f"[MIXED] Generated: {generated_text_mix}")
                            print(f"[MIXED] Attention weights: {alpha[0].cpu().numpy()}")
                        except:
                            print(f"[MIXED] Generated bytes: {generated_bytes_mix.tolist()[:50]}...")
            
            model.train()

        # snapshot creation at interval
        if step % cfg["snapshot_interval"] == 0:
            snapshot_and_store()

    # save final model
    torch.save({
        "model": model.state_dict(),
        "flag_table": flag_table.weight.detach().cpu(),
        "snapshots": len(snap_bank.bank),
    }, args.out_model)
    print("Training finished. Saved:", args.out_model)
    print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Close logger
    sys.stdout = logger.terminal
    logger.close()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, required=True, help="path to plaintext corpus (.txt or raw bytes)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--total_steps", type=int, default=DEFAULTS["total_steps"])
    parser.add_argument("--snapshot_interval", type=int, default=DEFAULTS["snapshot_interval"])
    parser.add_argument("--anchors_per_snapshot", type=int, default=DEFAULTS["anchors_per_snapshot"])
    parser.add_argument("--max_snapshots", type=int, default=DEFAULTS["max_snapshots"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--out_model", type=str, default="snapshot_gpt_final.pt")
    args = parser.parse_args()
    run_training(args)


