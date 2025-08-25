
#!/usr/bin/env python3
"""
Snapshot-GPT (hash flags + chunk overfitting)

What's new vs your original file:
- Replaces the finite nn.Embedding flag table with a HASH-BASED flag embedding generator
  (infinite ID space; collisions possible by design).
- Adds a deterministic sentence chunker and an "overfit-per-chunk" training loop that
  cycles chunks and rehearses earlier ones to mitigate forgetting.
- Keeps your SimpleBlock/SnapshotGPT structure; uses flag_proj inside the model as before.

Usage example:
  python snapshot_gpt_hash_overfit.py \
    --datafile ./shakespeare.txt \
    --device cuda \
    --total_cycles 5 \
    --max_sentences_per_chunk 6 \
    --steps_per_chunk 300 \
    --target_loss 0.2

Notes:
- Byte-level modeling is preserved. We overfit each chunk by sampling windows from the
  chunk and minimizing next-token CE until the target loss or steps_per_chunk reached.
- "Infinite" flags: every chunk i uses a deterministic hash -> flag vector of size flag_dim.
  The model still has a learned Linear(flag_dim -> d_model) (flag_proj) to map into model space.
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
    "block_size": 128,       # a bit larger; chunks can exceed this; we sample windows
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "flag_dim": 32,
    "batch_size": 32,
    "lr": 3e-4,
    "warmup_steps": 500,
    # overfitting scheme controls
    "max_sentences_per_chunk": 6,
    "min_sentences_per_chunk": 1,
    "total_cycles": 3,       # how many full passes over all chunks
    "steps_per_chunk": 300,  # cap steps devoted to one chunk per visit
    "rehearsal_steps": 60,   # after a full pass, quick rehearsal per chunk
    "target_loss": 0.25,     # early-stop threshold for a chunk (cross-entropy)
    "gen_every": 400,        # generate samples every N steps
}

# -----------------------
# Byte utilities
# -----------------------

def to_long_bytes(s: bytes) -> torch.Tensor:
    arr = np.frombuffer(s, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(arr)

# -----------------------
# Sentence chunking (very simple, deterministic)
# -----------------------
SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")

def read_and_chunk(path: str, min_sentences: int, max_sentences: int) -> List[bytes]:
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        # fall back: treat as latin-1 to keep bytes stable
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
    # encode back to bytes so downstream stays byte-level
    return [c.encode('utf-8', errors='replace') for c in chunks]

# -----------------------
# Hash-based flag embedding (infinite IDs)
# -----------------------
class HashFlagEmbedding(nn.Module):
    """Deterministic hash(ID) -> R^{flag_dim} in [-1,1], then L2-normalize.
    Uses BLAKE2b with digest_size=flag_dim (<=64). Collisions are possible by design.
    """
    def __init__(self, flag_dim: int):
        super().__init__()
        if flag_dim > 64:
            raise ValueError("flag_dim > 64 not supported by this simple hasher; lower it or extend implementation.")
        self.flag_dim = flag_dim

    @torch.no_grad()
    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        # ids: (B,) of non-negative integers
        out = []
        for i in ids.tolist():
            h = hashlib.blake2b(str(i).encode('utf-8'), digest_size=self.flag_dim).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            v = (v / 127.5) - 1.0  # map 0..255 -> -1..1
            # small nonlinearity to spread
            v = np.tanh(v * 1.5)
            # normalize
            n = np.linalg.norm(v) + 1e-6
            v = v / n
            out.append(v)
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
        att_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + att_out)
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
# Batching over a single chunk
# -----------------------
class ChunkBatcher:
    def __init__(self, chunk_bytes: bytes):
        self.data = to_long_bytes(chunk_bytes)  # (L,)
    def sample(self, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L = len(self.data)
        # If the chunk is very short, repeat it to reach a minimum length
        if L < 2:
            arr = torch.tensor([32, 32], dtype=torch.long)  # minimal space-space
        else:
            arr = self.data
        max_start = max(0, len(arr) - 2)
        starts = torch.randint(0, max_start + 1, (batch_size,))
        xs = []
        ys = []
        for s in starts:
            s = int(s)
            # window cannot exceed available tokens
            T = min(block_size, len(arr) - s - 1)
            x = arr[s:s+T]
            y = arr[s+1:s+T+1]
            xs.append(x)
            ys.append(y)
        # pad to same length within batch (left pad with 0 which is a valid byte; masking via loss)
        maxT = max(x.size(0) for x in xs)
        xb = torch.zeros((batch_size, maxT), dtype=torch.long)
        yb = torch.zeros((batch_size, maxT), dtype=torch.long)
        mask = torch.zeros((batch_size, maxT), dtype=torch.bool)
        for i,(x,y) in enumerate(zip(xs,ys)):
            xb[i, :x.size(0)] = x
            yb[i, :y.size(0)] = y
            mask[i, :y.size(0)] = 1
        return xb, yb, mask

# -----------------------
# Training orchestration (overfit-per-chunk)
# -----------------------

def run_training(args):
    cfg = DEFAULTS.copy(); cfg.update(vars(args))

    log_filename = f"training_log_{int(time.time())}.txt"
    logger = TeeLogger(log_filename); sys.stdout = logger
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {cfg}")

    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # Prepare chunks
    chunks = read_and_chunk(cfg["datafile"], cfg["min_sentences_per_chunk"], cfg["max_sentences_per_chunk"])
    print(f"Prepared {len(chunks)} chunks.")

    # Model + hash flags
    model = SnapshotGPT(cfg["vocab_size"], cfg["block_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["flag_dim"]).to(device)
    hash_flags = HashFlagEmbedding(cfg["flag_dim"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)

    global_step = 0
    def ce_loss_with_mask(logits, targets, mask):
        # logits: (B,T,V), targets: (B,T), mask: (B,T)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none')
        loss = loss * mask.reshape(-1).float()
        denom = mask.sum().clamp_min(1).float()
        return loss.sum() / denom

    for cycle in range(cfg["total_cycles"]):
        print(f"==== Cycle {cycle+1}/{cfg['total_cycles']} ====")
        # visit each chunk and overfit
        for cid, chunk_bytes in enumerate(chunks):
            batcher = ChunkBatcher(chunk_bytes)
            eid = torch.full((cfg["batch_size"],), cid, dtype=torch.long, device=device)

            # inner loop for a single chunk
            running = []
            for step_in_chunk in range(cfg["steps_per_chunk"]):
                model.train()
                xb, yb, mask = batcher.sample(cfg["batch_size"], cfg["block_size"])  # on CPU
                xb = xb.to(device); yb = yb.to(device); mask = mask.to(device)
                flag_vec = hash_flags(eid)

                logits, _, _ = model(xb, flag_vec)
                loss = ce_loss_with_mask(logits, yb, mask)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                # warmup
                global_step += 1
                if global_step < cfg["warmup_steps"]:
                    for g in opt.param_groups:
                        g["lr"] = cfg["lr"] * (global_step / cfg["warmup_steps"])
                opt.step()

                running.append(float(loss.item()))
                if (step_in_chunk+1) % 50 == 0:
                    avg = sum(running[-50:]) / min(50, len(running))
                    print(f"cycle {cycle+1} | chunk {cid+1}/{len(chunks)} | step {step_in_chunk+1}/{cfg['steps_per_chunk']} | loss {loss.item():.3f} | avg50 {avg:.3f}")

                # early stop for this chunk
                if len(running) >= 30 and (sum(running[-30:]) / 30.0) <= cfg["target_loss"]:
                    print(f"  -> early stop: avg last 30 = {sum(running[-30:]) / 30.0:.3f} <= target {cfg['target_loss']}")
                    break

                if cfg["gen_every"] and (global_step % cfg["gen_every"] == 0):
                    model.eval()
                    with torch.no_grad():
                        # generate 120 bytes from the first 32 bytes of the chunk under this flag
                        seed = to_long_bytes(chunk_bytes)[:min(32, len(chunk_bytes))].unsqueeze(0).to(device)
                        gen = seed.clone()
                        for _ in range(120):
                            inp = gen[:, -cfg["block_size"]:]
                            logits, _, _ = model(inp, hash_flags(torch.tensor([cid], device=device)))
                            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                            nxt = torch.multinomial(probs, 1)
                            gen = torch.cat([gen, nxt], dim=1)
                        arr = gen[0].detach().cpu().numpy().tolist()
                        preview = bytes(arr).decode('utf-8', errors='replace')
                        print("[GEN] chunk", cid, preview[:200].replace('\n',' '))

        # quick rehearsal pass across all chunks (few steps each)
        if cfg["rehearsal_steps"] > 0:
            print("---- Rehearsal pass ----")
            for cid, chunk_bytes in enumerate(chunks):
                batcher = ChunkBatcher(chunk_bytes)
                eid = torch.full((cfg["batch_size"],), cid, dtype=torch.long, device=device)
                for _ in range(cfg["rehearsal_steps"]):
                    model.train()
                    xb, yb, mask = batcher.sample(cfg["batch_size"], cfg["block_size"])  # cpu -> device
                    xb = xb.to(device); yb = yb.to(device); mask = mask.to(device)
                    flag_vec = hash_flags(eid)
                    logits, _, _ = model(xb, flag_vec)
                    loss = ce_loss_with_mask(logits, yb, mask)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    # save
    torch.save({
        "model": model.state_dict(),
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
    # overfit scheme controls
    p.add_argument("--min_sentences_per_chunk", type=int, default=DEFAULTS["min_sentences_per_chunk"])
    p.add_argument("--max_sentences_per_chunk", type=int, default=DEFAULTS["max_sentences_per_chunk"])
    p.add_argument("--total_cycles", type=int, default=DEFAULTS["total_cycles"])
    p.add_argument("--steps_per_chunk", type=int, default=DEFAULTS["steps_per_chunk"])
    p.add_argument("--rehearsal_steps", type=int, default=DEFAULTS["rehearsal_steps"])
    p.add_argument("--target_loss", type=float, default=DEFAULTS["target_loss"])
    p.add_argument("--gen_every", type=int, default=DEFAULTS["gen_every"])
    p.add_argument("--out_model", type=str, default="snapshot_gpt_hash_overfit.pt")
    args = p.parse_args()
    run_training(args)
