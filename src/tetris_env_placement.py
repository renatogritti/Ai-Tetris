"""
Ambiente Tetris de alto nível com ações de posicionamento de peça.

Cada passo escolhe uma rotação e uma coluna de destino para a peça atual.
Isso simplifica o aprendizado e torna o problema muito mais apropriado para RL,
pois o agente decide diretamente onde colocar a peça, em vez de controlar movimentos atômicos.
"""

from typing import Dict, Any, Optional, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.config import GRID_ALTURA, GRID_LARGURA
from src.tetris_engine import TetrisEngine
from src.tetris_env import PEÇA_PARA_INT
from src.tetris_gui import TetrisGUI


def rotate_shape(shape: List[List[int]]) -> List[List[int]]:
    """Rotaciona a matriz da peça 90 graus no sentido horário."""
    return [list(row) for row in zip(*shape[::-1])]


class TetrisPlacementEnv(gym.Env):
    """
    Ambiente Gymnasium com ações de localização final para a peça atual.

    A composição das ações é:
        action = rotation_index * GRID_LARGURA + target_column
    onde rotation_index é 0..3 e target_column é 0..9.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    MAX_PLACEMENTS: int = 250

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.engine = TetrisEngine()
        self.gui: Optional[TetrisGUI] = None
        self._placement_count = 0

        self.action_space = spaces.Discrete(4 * GRID_LARGURA)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2, GRID_ALTURA, GRID_LARGURA),
                dtype=np.float32,
            ),
            "current_piece": spaces.Discrete(8),
            "next_piece": spaces.Discrete(8),
        })

    def _get_obs(self) -> Dict[str, Any]:
        grid_binaria = np.zeros((GRID_ALTURA, GRID_LARGURA), dtype=np.float32)
        for r in range(GRID_ALTURA):
            for c in range(GRID_LARGURA):
                if self.engine.grid[r][c] != "":
                    grid_binaria[r, c] = 1.0

        current_piece_mask = np.zeros((GRID_ALTURA, GRID_LARGURA), dtype=np.float32)
        for r, row in enumerate(self.engine.current_piece_shape):
            for c, val in enumerate(row):
                if val:
                    gy = self.engine.current_y + r
                    gx = self.engine.current_x + c
                    if 0 <= gy < GRID_ALTURA and 0 <= gx < GRID_LARGURA:
                        current_piece_mask[gy, gx] = 1.0

        return {
            "grid": np.stack([grid_binaria, current_piece_mask], axis=0),
            "current_piece": PEÇA_PARA_INT.get(self.engine.current_piece_type, 0),
            "next_piece": PEÇA_PARA_INT.get(self.engine.next_piece_type, 0),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "score": self.engine.score,
            "level": self.engine.level,
            "lines_cleared": self.engine.lines_cleared_total,
            "placement_count": self._placement_count,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)

        if self.engine.game_over and self.render_mode == "human" and self.gui is not None:
            self.gui.renderizar()
            pygame.time.wait(500)

        self.engine.reset()
        self._placement_count = 0

        if self.gui is not None:
            self.gui.jogo_iniciado = True
            self.gui.pausado = False

        return self._get_obs(), self._get_info()

    def _decode_action(self, action: int):
        rotation = action // GRID_LARGURA
        column = action % GRID_LARGURA
        return rotation, column

    def _apply_placement(self, rotation: int, target_x: int) -> bool:
        shape = self.engine.current_piece_shape
        for _ in range(rotation):
            shape = rotate_shape(shape)

        if self.engine.check_collision(shape, target_x, self.engine.current_y):
            return False

        self.engine.current_piece_shape = shape
        self.engine.current_x = target_x
        return True

    def _compute_reward(self, lines_cleared: int) -> float:
        heights = self.engine.get_column_heights()
        max_height = max(heights) if heights else 0
        holes = self.engine.get_holes_count()
        bumpiness = self.engine.get_bumpiness()

        reward = 0.0
        if lines_cleared > 0:
            reward += (2 ** lines_cleared - 1) * 4.0
        reward += 0.05
        reward += -holes * 1.0
        reward += -max_height * 0.1
        reward += -bumpiness * 0.2

        if self.engine.game_over:
            reward += -15.0

        return float(reward)

    def step(self, action: int):
        if self.engine.game_over:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self._placement_count += 1
        rotation, target_x = self._decode_action(int(action))

        valid = self._apply_placement(rotation, target_x)
        if not valid:
            self.engine.game_over = True
            reward = -8.0
            return self._get_obs(), reward, True, False, self._get_info()

        lines_cleared = self.engine.drop_hard()
        terminated = self.engine.game_over
        truncated = self._placement_count >= self.MAX_PLACEMENTS
        reward = self._compute_reward(lines_cleared)

        if self.render_mode is not None:
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = TetrisGUI(self.engine)
                self.gui.jogo_iniciado = True

            pygame.event.pump()
            self.gui.renderizar()
            self.gui.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            if self.gui is None:
                self.gui = TetrisGUI(self.engine)
                self.gui.jogo_iniciado = True

            self.gui.renderizar()
            return pygame.surfarray.array3d(self.gui.screen)
        return None

    def close(self) -> None:
        if self.gui is not None:
            pygame.quit()
            self.gui = None
