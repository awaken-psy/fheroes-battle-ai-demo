"""T3 — Self-play opponent pool to prevent strategy collapse.

Maintains a pool of recent model checkpoints that are used as opponents
during training.  Each rollout randomly picks between pure self-play
(current policy vs itself) and pool-play (current policy vs a sampled
older checkpoint), keeping the learning agent exposed to diverse
strategies.

Key design:
  - FIFO eviction: oldest checkpoint removed when capacity exceeded.
  - Disk persistence: every ``add`` writes a file; ``load_from_disk``
    restores the pool on resume.
  - Lazy loading: ``sample`` reads from disk on demand (keeps memory
    footprint low — only one opponent model in VRAM at a time).
"""

import os
import random as _random
from typing import List, Optional, Tuple

import torch


class OpponentPool:
    """Save recent model checkpoints as opponents for self-play training.

    Parameters
    ----------
    capacity : int
        Maximum number of opponent checkpoints to retain.
    save_dir : str
        Directory for persisting checkpoint files.
    """

    def __init__(self, capacity: int = 5, save_dir: str = "checkpoints/opponent_pool"):
        self._capacity = max(1, capacity)
        self._save_dir = save_dir
        # Internal list of (step, file_path) tuples, ordered oldest-first.
        self._entries: List[Tuple[int, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────

    def add(self, model_state_dict: dict, step: int) -> None:
        """Add a model checkpoint to the pool.

        Persists to disk immediately.  Evicts the oldest entry if the
        pool exceeds capacity.

        Parameters
        ----------
        model_state_dict : dict
            ``model.state_dict()`` of the checkpoint to save.
        step : int
            Training step at which this checkpoint was created.
        """
        path = os.path.join(self._save_dir, f"pool_{step}.pt")
        torch.save({"step": step, "model": model_state_dict}, path)
        self._entries.append((step, path))

        # Evict oldest if over capacity
        while len(self._entries) > self._capacity:
            _, old_path = self._entries.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def sample(self) -> Optional[dict]:
        """Randomly sample an opponent's model state_dict.

        Returns ``None`` if the pool is empty.
        """
        if not self._entries:
            return None
        _, path = _random.choice(self._entries)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt["model"]

    def load_from_disk(self) -> None:
        """Restore pool state from the save directory.

        Scans ``save_dir`` for ``pool_*.pt`` files, loads their step
        counts, and rebuilds the internal entry list sorted by step
        (oldest first, matching FIFO eviction order).
        """
        self._entries.clear()
        if not os.path.isdir(self._save_dir):
            return

        candidates: List[Tuple[int, str]] = []
        for fname in os.listdir(self._save_dir):
            if not fname.startswith("pool_") or not fname.endswith(".pt"):
                continue
            path = os.path.join(self._save_dir, fname)
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                step = int(ckpt["step"])
                candidates.append((step, path))
            except Exception:
                # Corrupted file — skip
                continue

        # Sort by step (oldest first) and trim to capacity
        candidates.sort(key=lambda x: x[0])
        self._entries = candidates[-self._capacity:]

    def __len__(self) -> int:
        """Current number of opponents in the pool."""
        return len(self._entries)

    # ── Representation ──────────────────────────────────────────

    def __repr__(self) -> str:
        steps = [s for s, _ in self._entries]
        return (f"OpponentPool(capacity={self._capacity}, "
                f"size={len(self._entries)}, steps={steps})")
