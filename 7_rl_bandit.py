"""
7_rl_bandit.py
Reinforcement Learning feedback loop.
The bandit learns the best retrieval strategy from user clicks.
Integrates with the full pipeline — import and use after launch.
"""

import json
import datetime
import numpy as np
import os
import importlib.util

# ── config ────────────────────────────────────────────────────────────────────
BANDIT_STATE_FILE = "models/bandit.json"
FEEDBACK_LOG      = "data/feedback.jsonl"


# ── retrieval strategy arms ───────────────────────────────────────────────────
ARM_CONFIGS = [
    {"name": "bm25_heavy",     "w_bm25": 0.50, "w_faiss": 0.30, "w_fuzzy": 0.20},
    {"name": "faiss_heavy",    "w_bm25": 0.20, "w_faiss": 0.60, "w_fuzzy": 0.20},
    {"name": "balanced",       "w_bm25": 0.30, "w_faiss": 0.50, "w_fuzzy": 0.20},
    {"name": "semantic_focus", "w_bm25": 0.15, "w_faiss": 0.65, "w_fuzzy": 0.20},
]
N_ARMS = len(ARM_CONFIGS)


_PIPELINE_MOD = None


def _load_pipeline_module():
    """Load 5_full_pipeline.py despite its numeric filename."""
    global _PIPELINE_MOD
    if _PIPELINE_MOD is not None:
        return _PIPELINE_MOD

    spec = importlib.util.spec_from_file_location(
        "pipeline5", "5_full_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _PIPELINE_MOD = mod
    return _PIPELINE_MOD


class AddressBandit:
    """
    Epsilon-greedy multi-arm bandit.
    Each arm = a different weighting of retrieval signals.
    Learns which weighting works best for your specific data.

    Usage:
        bandit = AddressBandit()
        bandit.load()

        # choose arm before search
        arm, weights = bandit.choose()

        # after user accepts/rejects suggestion
        bandit.update(arm, reward=1.0)   # accepted
        bandit.update(arm, reward=0.0)   # rejected
    """

    def __init__(self, n_arms: int = N_ARMS, epsilon: float = 0.10):
        self.n_arms  = n_arms
        self.epsilon = epsilon
        # optimistic initialisation: start at 0.5, not 0.0
        self.q       = np.ones(n_arms) * 0.50
        self.cnt     = np.zeros(n_arms)

    def choose(self) -> tuple:
        """
        Epsilon-greedy selection.
        Returns (arm_index, weight_config_dict).
        """
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.n_arms)   # explore
        else:
            arm = int(np.argmax(self.q))            # exploit best known
        return arm, ARM_CONFIGS[arm]

    def update(self, arm: int, reward: float):
        """
        Update estimated value for the chosen arm.
        reward = 1.0 if user accepted suggestion, 0.0 if rejected.
        Incremental mean update (no replay buffer needed).
        """
        self.cnt[arm] += 1
        self.q[arm]   += (reward - self.q[arm]) / self.cnt[arm]
        self._save()
        self._log_feedback(arm, reward)

    def stats(self):
        """Print current bandit state."""
        print("\nBandit arm estimates:")
        print(f"  {'Arm':<20} {'Value':>8} {'Pulls':>8}")
        print("  " + "-" * 38)
        for i, cfg in enumerate(ARM_CONFIGS):
            marker = " ← best" if i == int(np.argmax(self.q)) else ""
            print(f"  {cfg['name']:<20} {self.q[i]:>8.3f} "
                  f"{int(self.cnt[i]):>8}{marker}")
        print(f"\n  Total pulls: {int(self.cnt.sum()):,}")
        print(f"  Epsilon    : {self.epsilon:.0%} exploration")

    def decay_epsilon(self, min_epsilon: float = 0.02):
        """Reduce exploration as the bandit gains confidence."""
        self.epsilon = max(min_epsilon, self.epsilon * 0.995)

    def _save(self):
        os.makedirs("models", exist_ok=True)
        with open(BANDIT_STATE_FILE, "w") as f:
            json.dump({
                "q":       self.q.tolist(),
                "cnt":     self.cnt.tolist(),
                "epsilon": self.epsilon,
            }, f, indent=2)

    def load(self):
        """Load saved bandit state. Safe to call even if file doesn't exist."""
        try:
            with open(BANDIT_STATE_FILE) as f:
                d = json.load(f)
            self.q       = np.array(d["q"])
            self.cnt     = np.array(d["cnt"])
            self.epsilon = d.get("epsilon", self.epsilon)
            print(f"Bandit loaded - {int(self.cnt.sum()):,} total pulls so far")
        except FileNotFoundError:
            print("No saved bandit state - starting fresh")

    def _log_feedback(self, arm: int, reward: float):
        os.makedirs("data", exist_ok=True)
        entry = {
            "ts":      str(datetime.datetime.now()),
            "arm":     arm,
            "arm_name":ARM_CONFIGS[arm]["name"],
            "reward":  reward,
            "q_values":self.q.tolist(),
        }
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── integration with pipeline ─────────────────────────────────────────────────
def search_with_bandit(raw_input: str, bandit: AddressBandit,
                       models: tuple, top_n: int = 5) -> dict:
    """
    Drop-in replacement for correct_address() that uses bandit weights.
    Returns result dict + arm index for feedback recording.
    """
    pipeline = _load_pipeline_module()
    t5_correct = pipeline.t5_correct
    rerank_candidates = pipeline.rerank_candidates
    normalize = pipeline.normalize
    t5_beams = pipeline.T5_BEAMS
    retrieval_top_k = pipeline.RETRIEVAL_TOP_K

    # Supports both tuple shapes:
    #  - 5_full_pipeline.load_models() -> 8 elements
    #  - 5_full_pipeline_sql.load_models() -> 11 elements
    if len(models) >= 11:
        t5_tok, t5, embedder, faiss_idx, addresses, _, bm25, ranker, addr_to_idx, _, _ = models
    else:
        t5_tok, t5, embedder, faiss_idx, addresses, bm25, ranker, addr_to_idx = models

    # choose arm
    arm, weights = bandit.choose()
    bandit.decay_epsilon()

    # step 1: T5 correction
    corrected = t5_correct(raw_input, t5_tok, t5, num_beams=t5_beams)

    # step 2: retrieve with bandit-chosen weights
    from collections import defaultdict
    from rapidfuzz import fuzz as rfuzz

    scores   = defaultdict(float)
    q        = normalize(corrected)
    bm25_raw = bm25.get_scores(q.split())
    bm25_max = float(bm25_raw.max()) + 1e-9

    for i in np.argsort(bm25_raw)[::-1][:retrieval_top_k]:
        scores[i] += (float(bm25_raw[i]) / bm25_max) * weights["w_bm25"]

    q_vec = embedder.encode([q], normalize_embeddings=True).astype("float32")
    D, I  = faiss_idx.search(q_vec, retrieval_top_k)
    for j, idx in enumerate(I[0]):
        scores[idx] += float(D[0][j]) * weights["w_faiss"]

    for idx, addr in enumerate(addresses):
        fs = rfuzz.token_sort_ratio(q, addr) / 100.0
        if fs > 0.55:
            scores[idx] += fs * weights["w_fuzzy"]

    ranked     = sorted(scores.items(), key=lambda x: -x[1])
    candidates = [(addresses[i], s) for i, s in ranked[:retrieval_top_k]]

    # step 3: re-rank
    final = rerank_candidates(
        corrected, candidates,
        embedder, faiss_idx, addresses, bm25, ranker, addr_to_idx,
        top_n=top_n,
    )

    return {
        "original":    raw_input,
        "corrected":   corrected,
        "top_matches": final,
        "best_match":  final[0][0] if final else "",
        "confidence":  round(final[0][1], 4) if final else 0.0,
        "arm":         arm,          # pass this back to record_feedback()
    }


def record_feedback(bandit: AddressBandit, arm: int, accepted: bool):
    """Call this when user clicks a suggestion or dismisses results."""
    bandit.update(arm, reward=1.0 if accepted else 0.0)


# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bandit = AddressBandit()
    bandit.load()

    # simulate 50 rounds of user feedback
    print("Simulating 50 rounds of user feedback...\n")
    for _ in range(50):
        arm, weights = bandit.choose()
        # simulate: semantic-heavy arm tends to work better
        reward = 1.0 if weights["w_faiss"] >= 0.50 else \
                 float(np.random.random() > 0.35)
        bandit.update(arm, reward)

    bandit.stats()
    print(f"\nFeedback log saved -> {FEEDBACK_LOG}")
    print("Bandit state saved -> models/bandit.json")
    print("\nAfter ~1000 real user interactions, the bandit converges")
    print("to the best strategy for your specific dataset.")
