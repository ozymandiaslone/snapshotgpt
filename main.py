
#!/usr/bin/env python3
"""
Snapshot-GPT (hash flags + per-chunk HARD overfit + learned SOFT attention over past chunks)

Key ideas added on top of `snapshot_gpt_hash_overfit.py`:
- Stage 1 (HARD): overfit a chunk using its own hash flag (one-hot selection). While doing so,
  we update a per-chunk key vector (EMA) derived from the model's pooled state (q-projected).
- Stage 2 (SOFT): for the same chunk, we DO NOT hard-set the flag. Instead we route via a
  learned attention over the keys of RECENT chunks, producing a soft mixture of flag embeddings.
  We supervise the router with a target that should put most mass on the current chunk.

This realizes "attention to past selves" while still allowing strong per-chunk memorization.

Usage example:
  python snapshot_gpt_hash_overfit_attn.py \
    --datafile ./shakespeare.txt \
    --device cpu \
    --total_cycles 3 \
    --steps_per_chunk_hard 300 \
    --steps_per_chunk_soft 120 \
    --target_loss 0.15 \
    --max_attn_chunks 256 \
    --attn_ce_weight 0.2

Notes:
- We cap the attention set to the last `max_attn_chunks` chunks for tractability.
- Hash flags provide infinite IDs; collisions possible by design.
- We keep byte-level modeling.
"""

import argparse, math, os, random, re, time, sys, hashlib
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Tee logger
# -----------------------
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()

# -----------------------
# Defaults
# -----------------------
DEFAULTS = {
    "vocab_size": 256,
    "block_size": 128,
    "d_model": 256,
    "n_layers": 3, 
    "n_heads": 4,  # and this
    "flag_dim": 32,
    "batch_size": 32,
    "lr": 3e-4,
    "warmup_steps": 500,
    # chunking
    "max_sentences_per_chunk": 5,
    "min_sentences_per_chunk": 1,
    # cycles
    "total_cycles": 3,
    # per-chunk stages
    "steps_per_chunk_hard": 300,
    "steps_per_chunk_soft": 120,
    "target_loss": 0.15,
    "gen_every": 400,
    # router / memory
    "max_attn_chunks": 512,   # number of recent chunks to include in router's attention set
    "ema_alpha": 0.9,         # EMA factor for per-chunk key vectors
    "attn_ce_weight": 0.2,    # weight for attention supervision (push mass to current chunk)
}

# -----------------------
# Byte utilities
# -----------------------

def to_long_bytes(s: bytes) -> torch.Tensor:
    arr = np.frombuffer(s, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(arr)

# -----------------------
# Sentence chunking (simple, deterministic)
# -----------------------

def read_and_chunk(path: str, min_sentences: int, max_sentences: int) -> List[bytes]:
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1')
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]

    chunks: List[str] = []
    i = 0
    rng = random.Random(0)
    while i < len(sentences):
        span = rng.randint(min_sentences, max_sentences)
        piece = " ".join(sentences[i:i+span]).strip()
        if piece:
            chunks.append(piece)
        i += span
    return [c.encode('utf-8', errors='replace') for c in chunks]

# -----------------------
# Hash-based flag embedding (infinite IDs)
# -----------------------
class HashFlagEmbedding(nn.Module):
    def __init__(self, flag_dim: int):
        super().__init__()
        if flag_dim > 64:
            raise ValueError("flag_dim > 64 not supported by this simple hasher; lower it or extend.")
        self.flag_dim = flag_dim
    @torch.no_grad()
    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        out = []
        for i in ids.tolist():
            h = hashlib.blake2b(str(i).encode('utf-8'), digest_size=self.flag_dim).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            v = (v / 127.5) - 1.0
            v = np.tanh(v * 1.5)
            n = np.linalg.norm(v) + 1e-6
            out.append(v / n)
        return torch.tensor(np.stack(out, axis=0), dtype=torch.float32, device=ids.device)

# -----------------------
# Model
# -----------------------
class SimpleBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.mlp(x))
        return x

class SnapshotGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_layers, n_heads, flag_dim):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([SimpleBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.flag_proj = nn.Linear(flag_dim, d_model)
        self.key_dim = d_model // 2
        self.q_proj = nn.Linear(d_model, self.key_dim)
    def forward(self, idx, flag_vec):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        x = x + self.flag_proj(flag_vec).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        pooled = x.mean(dim=1)
        logits = self.head(x)
        q = self.q_proj(pooled)
        return logits, pooled, q

# -----------------------
# Per-chunk batcher
# -----------------------
class ChunkBatcher:
    def __init__(self, chunk_bytes: bytes):
        self.data = to_long_bytes(chunk_bytes)
    def sample(self, batch_size: int, block_size: int):
        arr = self.data
        if len(arr) < 2:
            arr = torch.tensor([32,32], dtype=torch.long)
        max_start = max(0, len(arr) - 2)
        starts = torch.randint(0, max_start+1, (batch_size,))
        xs, ys = [], []
        for s in starts.tolist():
            T = min(block_size, len(arr) - s - 1)
            x = arr[s:s+T]; y = arr[s+1:s+T+1]
            xs.append(x); ys.append(y)
        maxT = max(x.size(0) for x in xs)
        xb = torch.zeros((batch_size, maxT), dtype=torch.long)
        yb = torch.zeros((batch_size, maxT), dtype=torch.long)
        mask = torch.zeros((batch_size, maxT), dtype=torch.bool)
        for i,(x,y) in enumerate(zip(xs,ys)):
            xb[i,:x.size(0)] = x
            yb[i,:y.size(0)] = y
            mask[i,:y.size(0)] = 1
        return xb, yb, mask

# -----------------------
# Chunk key memory (EMA over q)
# -----------------------
class ChunkKeyMemory:
    def __init__(self, n_chunks: int, key_dim: int, device: torch.device, ema_alpha: float):
        self.keys = torch.zeros((n_chunks, key_dim), dtype=torch.float32, device=device)
        self.filled = torch.zeros((n_chunks,), dtype=torch.bool, device=device)
        self.alpha = ema_alpha
        self.eps = 1e-6
    @torch.no_grad()
    def update_with_batch(self, cid: int, q_batch: torch.Tensor):
        # q_batch: (B, key_dim)
        q_mean = q_batch.mean(dim=0)
        if not self.filled[cid]:
            self.keys[cid] = q_mean
            self.filled[cid] = True
        else:
            self.keys[cid] = self.alpha * self.keys[cid] + (1 - self.alpha) * q_mean
    def get_recent_set(self, last_seen_cid: int, max_attn_chunks: int):
        start = max(0, last_seen_cid - max_attn_chunks + 1)
        idx = torch.arange(start, last_seen_cid+1, dtype=torch.long, device=self.keys.device)
        # keep only filled ones
        mask = self.filled[idx]
        return idx[mask], self.keys[idx[mask]]

# -----------------------
# Router: learned attention over chunk keys -> mix flag embeddings
# -----------------------
class LearnedChunkRouter(nn.Module):
    def __init__(self, query_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, query_dim, bias=True)
        self.scale = math.sqrt(query_dim)
    def forward(self, pooled_h: torch.Tensor, attn_keys: torch.Tensor, flag_embs_for_keys: torch.Tensor):
        # pooled_h: (B, d) ; attn_keys: (S, d) ; flag_embs_for_keys: (S, F)
        q = self.q_proj(pooled_h)  # (B,d)
        scores = q @ attn_keys.T / self.scale  # (B,S)
        alpha = scores.softmax(dim=-1)  # (B,S)
        mix = alpha @ flag_embs_for_keys  # (B,F)
        return mix, alpha

# -----------------------
# Loss helpers
# -----------------------

def ce_loss_with_mask(logits, targets, mask):
    V = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none')
    loss = loss * mask.reshape(-1).float()
    denom = mask.sum().clamp_min(1).float()
    return loss.sum() / denom

# -----------------------
# Training orchestration
# -----------------------

def run_training(args):
    cfg = DEFAULTS.copy(); cfg.update(vars(args))

    log_filename = f"training_log_{int(time.time())}.txt"
    logger = TeeLogger(log_filename); sys.stdout = logger
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {cfg}")

    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    chunks = read_and_chunk(cfg["datafile"], cfg["min_sentences_per_chunk"], cfg["max_sentences_per_chunk"])
    n_chunks = len(chunks)
    print(f"Prepared {n_chunks} chunks.")

    model = SnapshotGPT(cfg["vocab_size"], cfg["block_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["flag_dim"]).to(device)
    hasher = HashFlagEmbedding(cfg["flag_dim"]).to(device)
    router = LearnedChunkRouter(model.key_dim).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(router.parameters()), lr=cfg["lr"], weight_decay=1e-2)

    mem = ChunkKeyMemory(n_chunks=n_chunks, key_dim=model.key_dim, device=device, ema_alpha=cfg["ema_alpha"])

    global_step = 0

    for cycle in range(cfg["total_cycles"]):
        print(f"==== Cycle {cycle+1}/{cfg['total_cycles']} ====")

        for cid, chunk_bytes in enumerate(chunks):
            batcher = ChunkBatcher(chunk_bytes)

            # ---------- Stage 1: HARD overfit (one-hot hash flag) + update key memory ----------
            running = []
            for step_in_chunk in range(cfg["steps_per_chunk_hard"]):
                model.train()
                xb, yb, mask = batcher.sample(cfg["batch_size"], cfg["block_size"])
                xb = xb.to(device); yb = yb.to(device); mask = mask.to(device)
                eid = torch.full((cfg["batch_size"],), cid, dtype=torch.long, device=device)
                flag_vec = hasher(eid)

                logits, pooled, q = model(xb, flag_vec)
                loss = ce_loss_with_mask(logits, yb, mask)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                global_step += 1
                if global_step < cfg["warmup_steps"]:
                    for g in opt.param_groups:
                        g["lr"] = cfg["lr"] * (global_step / cfg["warmup_steps"]) 
                opt.step()

                # update memory with this chunk's q
                mem.update_with_batch(cid, q.detach())

                running.append(float(loss.item()))
                if (step_in_chunk+1) % 50 == 0:
                    avg50 = sum(running[-50:]) / min(50, len(running))
                    print(f"cycle {cycle+1} | chunk {cid+1}/{n_chunks} [HARD] | step {step_in_chunk+1}/{cfg['steps_per_chunk_hard']} | loss {loss.item():.3f} | avg50 {avg50:.3f}")

                # early stop for this chunk (hard phase)
                if len(running) >= 30 and (sum(running[-30:]) / 30.0) <= cfg["target_loss"]:
                    print(f"  -> HARD early stop: avg last 30 = {sum(running[-30:]) / 30.0:.3f} <= target {cfg['target_loss']}")
                    break

                if cfg["gen_every"] and (global_step % cfg["gen_every"] == 0):
                    model.eval()
                    with torch.no_grad():
                        seed = to_long_bytes(chunk_bytes)[:min(32, len(chunk_bytes))].unsqueeze(0).to(device)
                        gen = seed.clone()
                        for _ in range(120):
                            inp = gen[:, -cfg["block_size"]:]
                            logits, _, _ = model(inp, hasher(torch.tensor([cid], device=device)))
                            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                            nxt = torch.multinomial(probs, 1)
                            gen = torch.cat([gen, nxt], dim=1)
                        arr = gen[0].detach().cpu().numpy().tolist()
                        preview = bytes(arr).decode('utf-8', errors='replace')
                        print("[GEN HARD] chunk", cid, preview[:200].replace('\n',' '))

            # ---------- Stage 2: SOFT routing (learned attention over recent chunk keys) ----------
            # build attention candidate set (recent filled keys)
            idx_set, key_set = mem.get_recent_set(last_seen_cid=cid, max_attn_chunks=cfg["max_attn_chunks"])
            if idx_set.numel() == 0:
                continue  # nothing to route over yet

            # precompute flag embeddings for candidate set
            flag_embs_for_keys = hasher(idx_set)

            running_soft = []
            for step_in_chunk in range(cfg["steps_per_chunk_soft"]):
                model.train()
                xb, yb, mask = batcher.sample(cfg["batch_size"], cfg["block_size"])
                xb = xb.to(device); yb = yb.to(device); mask = mask.to(device)

                # query from zero-flag pass (or a tiny neutral vector)
                neutral = torch.zeros((cfg["batch_size"], cfg["flag_dim"]), device=device)
                _, pooled, q = model(xb, neutral)

                # learned routing over past chunk keys -> mixed flag
                mix_emb, alpha = router(q, key_set, flag_embs_for_keys)

                # run the model conditioned on mixed flag to predict the chunk
                logits, _, _ = model(xb, mix_emb)
                loss_lm = ce_loss_with_mask(logits, yb, mask)

                # supervise attention to put mass on the true chunk within the candidate set
                # build target indices
                # find where cid sits in idx_set
                pos = (idx_set == cid).nonzero(as_tuple=False).flatten()
                if pos.numel() > 0:
                    target = pos[0].item()
                    attn_loss = F.cross_entropy(alpha, torch.full((alpha.size(0),), target, dtype=torch.long, device=alpha.device))
                else:
                    attn_loss = torch.tensor(0.0, device=alpha.device)

                loss = loss_lm + cfg["attn_ce_weight"] * attn_loss

                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

                running_soft.append(float(loss.item()))
                if (step_in_chunk+1) % 50 == 0:
                    avg50 = sum(running_soft[-50:]) / min(50, len(running_soft))
                    print(f"cycle {cycle+1} | chunk {cid+1}/{n_chunks} [SOFT] | step {step_in_chunk+1}/{cfg['steps_per_chunk_soft']} | loss {loss.item():.3f} | avg50 {avg50:.3f}")

    # Save
    torch.save({
        "model": model.state_dict(),
        "router": router.state_dict(),
        "config": cfg,
    }, cfg["out_model"])
    print("Saved:", cfg["out_model"])
    sys.stdout = logger.terminal; logger.close()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datafile", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--block_size", type=int, default=DEFAULTS["block_size"])
    p.add_argument("--d_model", type=int, default=DEFAULTS["d_model"])
    p.add_argument("--n_layers", type=int, default=DEFAULTS["n_layers"])
    p.add_argument("--n_heads", type=int, default=DEFAULTS["n_heads"])
    p.add_argument("--flag_dim", type=int, default=DEFAULTS["flag_dim"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
    # chunking
    p.add_argument("--min_sentences_per_chunk", type=int, default=DEFAULTS["min_sentences_per_chunk"])
    p.add_argument("--max_sentences_per_chunk", type=int, default=DEFAULTS["max_sentences_per_chunk"])
    # cycles
    p.add_argument("--total_cycles", type=int, default=DEFAULTS["total_cycles"])
    # per-chunk stages
    p.add_argument("--steps_per_chunk_hard", type=int, default=DEFAULTS["steps_per_chunk_hard"])
    p.add_argument("--steps_per_chunk_soft", type=int, default=DEFAULTS["steps_per_chunk_soft"])
    p.add_argument("--target_loss", type=float, default=DEFAULTS["target_loss"])
    p.add_argument("--gen_every", type=int, default=DEFAULTS["gen_every"])
    # router / memory
    p.add_argument("--max_attn_chunks", type=int, default=DEFAULTS["max_attn_chunks"])
    p.add_argument("--ema_alpha", type=float, default=DEFAULTS["ema_alpha"])
    p.add_argument("--attn_ce_weight", type=float, default=DEFAULTS["attn_ce_weight"])
    p.add_argument("--out_model", type=str, default="snapshot_gpt_hash_overfit_attn.pt")
    args = p.parse_args()
    run_training(args)
