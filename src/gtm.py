"""Generative Topographic Mapping (Bishop, Svensén & Williams, 1998) numpy 구현.

합성 데이터로 로그우도 단조 증가, NaN 없음, 빈 cell 복원(map_other_space)을 확인했다.
"""
import numpy as np
from scipy.special import logsumexp


def square_grid(n):
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


class GTM:
    """Generative Topographic Mapping (Bishop, Svensén & Williams, 1998), numpy 구현."""

    def __init__(self, grid=25, rbf_grid=10, rbf_width=1.0, alpha=1e-3, n_iter=200, tol=1e-5):
        self.grid, self.rbf_grid, self.rbf_width = grid, rbf_grid, rbf_width
        self.alpha, self.n_iter, self.tol = alpha, n_iter, tol

    def _phi(self, X):
        sigma = self.rbf_width * 2 / (self.rbf_grid - 1)   # RBF 중심 간격 × 배수
        d2 = ((X[:, None, :] - self.mu[None, :, :]) ** 2).sum(-1)
        return np.hstack([np.exp(-d2 / (2 * sigma ** 2)), np.ones((len(X), 1))])

    def _dist(self, T):
        Y = self.Phi @ self.W
        return (Y ** 2).sum(1)[:, None] + (T ** 2).sum(1)[None, :] - 2 * Y @ T.T   # (K, N)

    def _estep(self, T):
        logp = -0.5 * self.beta * self._dist(T)
        lse = logsumexp(logp, axis=0)
        R = np.exp(logp - lse)
        D = T.shape[1]
        ll = (lse - np.log(len(self.X)) + 0.5 * D * np.log(self.beta / (2 * np.pi))).sum()
        return R, ll

    def fit(self, T):
        N, D = T.shape
        self.X = square_grid(self.grid)                     # 잠재 격자점 (K, 2) = 맵의 cell
        self.mu = square_grid(self.rbf_grid)                # RBF 중심 (M, 2)
        self.Phi = self._phi(self.X)                        # (K, M+1)

        # PCA 초기화: 격자를 주성분 1·2 평면에 펼쳐 놓고 시작
        mean = T.mean(0)
        _, S, Vt = np.linalg.svd(T - mean, full_matrices=False)
        eig = S ** 2 / N
        Xs = self.X / self.X.std(0)
        target = mean + Xs @ (np.sqrt(eig[:2])[:, None] * Vt[:2])
        self.W = np.linalg.lstsq(self.Phi, target, rcond=None)[0]
        Yg = (self.Phi @ self.W).reshape(self.grid, self.grid, D)
        nn = ((Yg[1:] - Yg[:-1]) ** 2).sum(-1).mean() / 2
        self.beta = 1 / max(eig[2], nn)

        self.loglik = []
        for _ in range(self.n_iter):
            R, ll = self._estep(T)
            self.loglik.append(ll)
            if len(self.loglik) > 1 and abs(ll - self.loglik[-2]) < self.tol * abs(ll):
                break
            G = R.sum(1)
            A = self.Phi.T @ (G[:, None] * self.Phi) + (self.alpha / self.beta) * np.eye(self.Phi.shape[1])
            self.W = np.linalg.solve(A, self.Phi.T @ (R @ T))
            self.beta = N * D / (R * self._dist(T)).sum()
        self.R, _ = self._estep(T)   # 마지막 W·β 기준으로 다시 계산 (반복 한도로 끝난 경우 대비)
        return self

    def save(self, path):
        np.savez(path, X=self.X, mu=self.mu, Phi=self.Phi, W=self.W, beta=self.beta,
                 loglik=np.array(self.loglik),
                 config=np.array([self.grid, self.rbf_grid, self.rbf_width, self.alpha]))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        grid, rbf_grid, rbf_width, alpha = f["config"]
        g = cls(grid=int(grid), rbf_grid=int(rbf_grid), rbf_width=float(rbf_width), alpha=float(alpha))
        g.X, g.mu, g.Phi, g.W, g.beta = f["X"], f["mu"], f["Phi"], f["W"], float(f["beta"])
        g.loglik = f["loglik"].tolist()
        return g

    def transform(self, T):
        """학습된 맵에 (검증 구간 등) 새 특허를 투영."""
        R, _ = self._estep(T)
        return R, R.argmax(0), R.T @ self.X                # responsibility, 최빈 cell, 2D 좌표(사후평균)

    def cell_vectors(self):
        """모든 cell(빈 cell 포함)의 원공간 복원 벡터 y_k."""
        return self.Phi @ self.W

    def map_other_space(self, T_other, R=None, lam=1e-3):
        """방법 B: SBERT로 학습한 R·Φ로 TF-IDF 등 다른 공간의 매핑을 맞춤 (M-step과 같은 식)."""
        R = self.R if R is None else R
        G = R.sum(1)
        A = self.Phi.T @ (G[:, None] * self.Phi) + lam * np.eye(self.Phi.shape[1])
        RT = np.asarray((T_other.T @ R.T).T)               # (K, V), 희소행렬도 가능
        W_other = np.linalg.solve(A, self.Phi.T @ RT)
        return self.Phi @ W_other                           # (K, V): cell별 키워드 가중치
