"""
Faithful approximation of T-BoostNet
Hassan, Saleh, Al-Sameai, de Moura, Alomayri & Zhang (2026), Composite Structures 378, 119906, Section 4.6, Eqs. (4)-(10).

Stage 1  XGBoost (n_estimators=200, max_depth=6, learning_rate=0.1, reg_lambda=10; Section 4.7) gives one leaf index
         per tree for every sample (Eq. 4).
Stage 2  leaf one-hot -> linear projection to e=32 (Eq. 5); inputs -> linear projection to p (Eq. 6); concatenation
         (Eq. 7); two multi-head self-attention blocks with residual + LayerNorm (Eqs. 8-9), n_heads=2; residual fusion
         H2 + H0 followed by a feed-forward network (Eq. 10). Transformer learning rate 0.001 (Section 4.7).

The paper does not say how the attention input is tokenised, so two readings are implemented:
  variant="literal"  H0 is one vector per sample, exactly as Eqs. (6)-(7) define it, so attention runs over a
                     sequence of length one (softmax over a single key is 1: the block reduces to linear maps).
  variant="tokens"   the generous reading: every input feature becomes a token (FT-Transformer style) and the leaf
                     embedding is one more token, so attention can mix information across features.

Details the paper does not report, fixed here and stated in the report: p=64 (the paper's own Transformer baseline
proj_dim, Table 3), FFN 64-32 with ReLU, Adam, batch 64, up to 400 epochs, early stopping (patience 30) on an inner 10 %
validation split drawn the same way as the outer split (random, by mix or by study), no dropout. The XGBoost stage is
fitted first and frozen: a tree ensemble cannot be trained jointly with a network by gradient descent, so the paper's
"joint optimization" is read as joint hyperparameter tuning.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

torch.set_num_threads(2)


class _Net(nn.Module):
    def __init__(self, d, n_leaf_ids, variant, p=64, e=32, heads=2, ffn=(64, 32)):
        super().__init__()
        self.variant = variant
        self.leaf = nn.EmbeddingBag(n_leaf_ids, e, mode="sum")          # W_leaf . OneHot(l), Eq. (5)
        bound = 1.0 / np.sqrt(n_leaf_ids)                                 # initialised like nn.Linear on the one-hot input
        nn.init.uniform_(self.leaf.weight, -bound, bound)
        if variant == "literal":
            self.inp = nn.Linear(d, p)                                   # Eq. (6)
            D, flat = p + e, p + e                                       # Eq. (7): one vector per sample
        elif variant == "tokens":
            D, flat = e, e * (d + 1)
            self.tok_w = nn.Parameter(torch.randn(d, D) * 0.02)          # per-feature token embedding
            self.tok_b = nn.Parameter(torch.zeros(d, D))
        else:
            raise ValueError(variant)
        self.mha1, self.ln1 = nn.MultiheadAttention(D, heads, batch_first=True), nn.LayerNorm(D)   # Eq. (8)
        self.mha2, self.ln2 = nn.MultiheadAttention(D, heads, batch_first=True), nn.LayerNorm(D)   # Eq. (9)
        layers, w = [], flat
        for h in ffn:
            layers += [nn.Linear(w, h), nn.ReLU()]
            w = h
        self.ffn = nn.Sequential(*layers, nn.Linear(w, 1))              # Eq. (10)

    def forward(self, x, leaf_ids):
        E = self.leaf(leaf_ids)                                          # (B, e)
        if self.variant == "literal":
            H0 = torch.cat([self.inp(x), E], dim=1).unsqueeze(1)         # (B, 1, p+e)
        else:
            T = x.unsqueeze(-1) * self.tok_w + self.tok_b                # (B, d, e)
            H0 = torch.cat([T, E.unsqueeze(1)], dim=1)                   # (B, d+1, e)
        a, _ = self.mha1(H0, H0, H0)
        H1 = self.ln1(H0 + a)
        a, _ = self.mha2(H1, H1, H1)
        H2 = self.ln2(H1 + a)
        return self.ffn((H2 + H0).flatten(1)).squeeze(-1)


class TBoostNet(BaseEstimator, RegressorMixin):
    def __init__(self, variant="literal", n_estimators=200, max_depth=6, learning_rate=0.1, reg_lambda=10.0,
                 n_heads=2, leaf_embed_dim=32, proj_dim=64, lr=1e-3, epochs=400, patience=30, batch=64,
                 val_frac=0.1, seed=42):
        self.variant, self.n_estimators, self.max_depth = variant, n_estimators, max_depth
        self.learning_rate, self.reg_lambda, self.n_heads = learning_rate, reg_lambda, n_heads
        self.leaf_embed_dim, self.proj_dim, self.lr, self.epochs = leaf_embed_dim, proj_dim, lr, epochs
        self.patience, self.batch, self.val_frac, self.seed = patience, batch, val_frac, seed

    def _leaf_ids(self, X):
        leaves = self.xgb_.apply(X).astype(np.int64)                     # (n, m) node index per tree, Eq. (4)
        return torch.from_numpy(leaves + self._offsets)

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)
        # inner validation split for early stopping, drawn the same way as the outer split
        if groups is None:
            idx = rng.permutation(len(y))
            n_val = max(1, int(round(self.val_frac * len(y))))
            va, tr = idx[:n_val], idx[n_val:]
        else:
            tr, va = next(GroupShuffleSplit(1, test_size=self.val_frac, random_state=self.seed)
                          .split(X, y, groups=np.asarray(groups)))
        # stage 1: XGBoost on the inner training rows, then frozen
        self.xgb_ = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                 learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                 random_state=self.seed, n_jobs=2)
        self.xgb_.fit(X[tr], y[tr])
        stride = 2 ** (self.max_depth + 1)
        self._offsets = (np.arange(self.n_estimators, dtype=np.int64) * stride)[None, :]
        n_leaf_ids = self.n_estimators * stride
        # stage 2: network on standardised target
        self.y_mu_, self.y_sd_ = y[tr].mean(), y[tr].std() + 1e-9
        Xt, Lt = torch.from_numpy(X), self._leaf_ids(X)
        yt = torch.from_numpy(((y - self.y_mu_) / self.y_sd_).astype(np.float32))
        self.net_ = _Net(X.shape[1], n_leaf_ids, self.variant, p=self.proj_dim, e=self.leaf_embed_dim,
                         heads=self.n_heads)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        tr_t, va_t = torch.from_numpy(tr), torch.from_numpy(va)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            perm = tr_t[torch.randperm(len(tr_t))]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[b], Lt[b]), yt[b])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = loss_fn(self.net_(Xt[va_t], Lt[va_t]), yt[va_t]).item()
            if v < best - 1e-6:
                best, best_state, wait = v, copy.deepcopy(self.net_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.epochs_run_ = epoch + 1
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters()))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.from_numpy(X), self._leaf_ids(X)).numpy()
        return out * self.y_sd_ + self.y_mu_
