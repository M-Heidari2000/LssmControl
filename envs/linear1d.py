import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class Linear1D(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 1
    y_dim = 1
    u_dim = 1

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Ns: Optional[np.ndarray] = None,
        No: Optional[np.ndarray] = None,
        render_mode: str = None,
        horizon: int = 1000,
        periodic: Optional[bool] = True,
    ):

        super().__init__()

        self.A = np.asarray(A, dtype=np.float32)
        self.B = np.asarray(B, dtype=np.float32)
        self.C = np.asarray(C, dtype=np.float32)
        if self.C.ndim == 0:
            self.C = self.C.reshape(1, 1)
        elif self.C.ndim == 1:
            self.C = self.C.reshape(1, -1)
        self.Ns = np.asarray(Ns, dtype=np.float32) if Ns is not None else None
        self.No = np.asarray(No, dtype=np.float32) if No is not None else None

        self._verify_parameters()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon
        self.periodic = periodic

        self.state_space = spaces.Box(
            low=np.array([-np.pi]),
            high=np.array([np.pi]),
            shape=(1,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

    def _verify_parameters(self):
        assert self.A.shape == (self.x_dim, self.x_dim)
        assert self.B.shape == (self.x_dim, self.u_dim)
        assert self.C.shape == (self.y_dim, self.x_dim)
        if self.Ns is not None:
            assert self.Ns.shape == (self.x_dim, self.x_dim)
        if self.No is not None:
            assert self.No.shape == () or self.No.shape == (self.y_dim, self.y_dim)
        observability = np.vstack([self.C, self.C @ self.A])
        assert np.linalg.matrix_rank(observability) == self.x_dim, "System is not observable"

    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        return s @ self.C.T

    def _get_obs(self):
        obs = self.manifold(self._state)
        if self.No is not None:
            cov = self.No if self.No.shape != () else np.array([[float(self.No)]], dtype=np.float32)
            no = self.np_random.multivariate_normal(
                mean=np.zeros(self.observation_space.shape),
                cov=cov,
            ).astype(np.float32).reshape(1, -1)
            obs = obs + no
        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)
        options = options or {}
        initial_state = options.get("initial_state")
        target_state = options.get("target_state")

        if initial_state is not None:
            assert initial_state.shape == self.state_space.shape
            self._state = initial_state.astype(np.float32).reshape(1, -1)
        else:
            self._state = self.state_space.sample().reshape(1, -1)

        if target_state is not None:
            assert target_state.shape == self.state_space.shape
            self._target = target_state.astype(np.float32).reshape(1, -1)
        else:
            self._target = self.state_space.sample().reshape(1, -1)

        self._step = 0
        observation = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ):
        assert action.shape == self.action_space.shape
        action = action.astype(np.float32).reshape(1, -1)
        action = np.clip(
            action,
            a_min=self.action_space.low,
            a_max=self.action_space.high,
        )

        self._state = self._state @ self.A.T + action @ self.B.T
        if self.Ns is not None:
            ns = self.np_random.multivariate_normal(
                mean=np.zeros(self.state_space.shape),
                cov=self.Ns,
            ).astype(np.float32).reshape(1, -1)
            self._state = self._state + ns

        self._step += 1
        truncated = bool(self._step >= self.horizon)
        terminated = False
        reward = 0.0

        if self.periodic:
            rng = self.state_space.high - self.state_space.low
            self._state = ((self._state - self.state_space.low) % rng) + self.state_space.low
        else:
            is_valid = (
                np.all(self.state_space.low < self._state.flatten()) and np.all(self._state.flatten() < self.state_space.high)
            )
            if not is_valid:
                terminated = True

        obs = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        fig, ax = plt.subplots(figsize=(8, 2.8), dpi=180)

        obs_cur = float(self._get_obs().reshape(-1)[0])
        obs_tgt = float(self.manifold(self._target).reshape(-1)[0])

        lo = float(self.state_space.low.reshape(-1)[0])
        hi = float(self.state_space.high.reshape(-1)[0])
        state_grid = np.linspace(lo, hi, 800, dtype=np.float32).reshape(-1, 1)
        curve = self.manifold(state_grid).reshape(-1)

        ax.plot(state_grid[:, 0], curve, linewidth=3, alpha=0.6, color="steelblue")
        ax.scatter(float(self._state.reshape(-1)[0]), obs_cur, s=90, c="black", marker="o", label="current")
        ax.scatter(float(self._target.reshape(-1)[0]), obs_tgt, s=120, c="red", marker="X", label="target")

        ax.grid(True, alpha=0.2)
        ax.set_xlabel("state x")
        ax.set_ylabel("observation y")
        ax.legend()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img