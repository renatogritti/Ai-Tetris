"""
Ambiente Tetris com Features Heurísticas (tetris_env_features.py).

Implementa o paradigma de avaliação de estados para Tetris:
Para cada peça, o ambiente enumera TODAS as posições finais válidas
(combinação de rotação × coluna), simula cada placement em uma cópia
do tabuleiro, e extrai 4 features heurísticas do board resultante.

Features extraídas por estado:
    1. lines_cleared: Linhas eliminadas por este placement
    2. holes: Número de buracos no tabuleiro resultante
    3. bumpiness: Irregularidade da superfície (soma de |Δh| entre colunas)
    4. aggregate_height: Soma das alturas de todas as colunas

O agente DQN recebe esses vetores de features e escolhe o placement
que maximiza o Q-value, em vez de aprender a partir da grid bruta.
"""

import copy
import random
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from src.config import GRID_ALTURA, GRID_LARGURA, FORMATOS_PEÇAS
from src.tetris_engine import TetrisEngine


def _rotate_shape(shape: List[List[int]]) -> List[List[int]]:
    """
    Rotaciona a matriz da peça 90 graus no sentido horário.

    Parâmetros:
        shape (List[List[int]]): Matriz da forma da peça.

    Retorna:
        List[List[int]]: Matriz rotacionada.
    """
    return [list(row) for row in zip(*shape[::-1])]


def _get_unique_rotations(piece_type: str) -> List[List[List[int]]]:
    """
    Retorna todas as rotações únicas de uma peça.

    Peça 'O' tem apenas 1 rotação. Peças 'I', 'S', 'Z' têm 2.
    Demais ('T', 'J', 'L') têm 4.

    Parâmetros:
        piece_type (str): Tipo da peça ('I', 'O', 'T', etc.).

    Retorna:
        List[List[List[int]]]: Lista de matrizes de forma, uma por rotação.
    """
    base_shape = [list(row) for row in FORMATOS_PEÇAS[piece_type]]

    if piece_type == "O":
        return [base_shape]

    rotations = [base_shape]
    current = base_shape
    num_rotations = 2 if piece_type in ("I", "S", "Z") else 4

    for _ in range(num_rotations - 1):
        current = _rotate_shape(current)
        rotations.append(current)

    return rotations


class TetrisFeatureEnv:
    """
    Ambiente de Tetris baseado em features heurísticas.

    Para cada peça, gera todos os placements possíveis e suas features.
    Não implementa a interface Gymnasium completa pois o paradigma DQN
    customizado não precisa dela — o agente interage diretamente.

    Atributos:
        engine (TetrisEngine): Motor lógico do Tetris.
        score (int): Pontuação acumulada do episódio.
        pieces_placed (int): Número de peças colocadas no episódio.
        lines_cleared (int): Total de linhas eliminadas no episódio.
        game_over (bool): Se o jogo terminou.
    """

    def __init__(self) -> None:
        """Inicializa o ambiente com um novo motor do Tetris."""
        self.engine = TetrisEngine()
        self.score: int = 0
        self.pieces_placed: int = 0
        self.lines_cleared: int = 0
        self.game_over: bool = False

    def reset(self) -> List[np.ndarray]:
        """
        Reinicia o ambiente para um novo episódio.

        Retorna:
            List[np.ndarray]: Lista de vetores de features para cada
                posição válida da primeira peça.
        """
        self.engine.reset()
        self.score = 0
        self.pieces_placed = 0
        self.lines_cleared = 0
        self.game_over = False
        return self.get_next_states()

    def _compute_features(self, grid: List[List[str]]) -> np.ndarray:
        """
        Calcula as 4 features heurísticas de um tabuleiro.

        Features:
            - lines_cleared: Contadas separadamente antes de limpar
            - holes: Células vazias abaixo de blocos preenchidos
            - bumpiness: Soma |Δh| entre colunas adjacentes
            - aggregate_height: Soma das alturas de todas as colunas

        Parâmetros:
            grid (List[List[str]]): Estado do tabuleiro (matriz 20×10).

        Retorna:
            np.ndarray: Vetor [lines_cleared, holes, bumpiness, aggregate_height].
        """
        # Contar linhas completas
        lines = 0
        for row in grid:
            if "" not in row:
                lines += 1

        # Limpar as linhas completas para calcular as demais features
        cleaned_grid = [row for row in grid if "" in row]
        while len(cleaned_grid) < GRID_ALTURA:
            cleaned_grid.insert(0, ["" for _ in range(GRID_LARGURA)])

        # Calcular alturas por coluna
        heights = [0] * GRID_LARGURA
        for col in range(GRID_LARGURA):
            for row in range(GRID_ALTURA):
                if cleaned_grid[row][col] != "":
                    heights[col] = GRID_ALTURA - row
                    break

        # Aggregate height (soma total)
        aggregate_height = sum(heights)

        # Holes (buracos)
        holes = 0
        for col in range(GRID_LARGURA):
            block_found = False
            for row in range(GRID_ALTURA):
                if cleaned_grid[row][col] != "":
                    block_found = True
                elif block_found:
                    holes += 1

        # Bumpiness (irregularidade)
        bumpiness = 0
        for col in range(GRID_LARGURA - 1):
            bumpiness += abs(heights[col] - heights[col + 1])

        return np.array(
            [lines, holes, bumpiness, aggregate_height], dtype=np.float32
        )

    def _try_placement(
        self, shape: List[List[int]], target_x: int, grid: List[List[str]], piece_type: str
    ) -> Optional[List[List[str]]]:
        """
        Tenta colocar uma peça em uma posição específica do tabuleiro.

        Simula o hard drop: coloca a peça na coluna target_x e deixa
        cair até colidir. Se a posição for inválida, retorna None.

        Parâmetros:
            shape (List[List[int]]): Matriz da forma da peça.
            target_x (int): Coluna de destino (canto esquerdo da matriz).
            grid (List[List[str]]): Estado atual do tabuleiro (será copiado).
            piece_type (str): Tipo da peça para marcar no grid.

        Retorna:
            Optional[List[List[str]]]: Grid resultante após o placement,
                ou None se a posição for inválida.
        """
        # Verificar se a peça cabe na coluna
        shape_width = len(shape[0])
        if target_x < 0 or target_x + shape_width > GRID_LARGURA:
            return None

        # Verificar se cabe no topo (posição inicial)
        start_y = -len(shape)
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    gx = target_x + c
                    gy = start_y + r
                    if gx < 0 or gx >= GRID_LARGURA:
                        return None

        # Hard drop: descer até colidir
        test_y = start_y
        while True:
            collision = False
            for r, row in enumerate(shape):
                for c, val in enumerate(row):
                    if val:
                        gx = target_x + c
                        gy = test_y + 1 + r
                        if gy >= GRID_ALTURA:
                            collision = True
                            break
                        if gy >= 0 and grid[gy][gx] != "":
                            collision = True
                            break
                if collision:
                    break
            if collision:
                break
            test_y += 1

        # Verificar se a peça ficou acima do tabuleiro (game over)
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    gy = test_y + r
                    if gy < 0:
                        return None

        # Criar cópia do grid e colocar a peça
        new_grid = [row[:] for row in grid]
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    gy = test_y + r
                    gx = target_x + c
                    if 0 <= gy < GRID_ALTURA and 0 <= gx < GRID_LARGURA:
                        new_grid[gy][gx] = piece_type

        return new_grid

    def get_next_states(self) -> List[np.ndarray]:
        """
        Gera as features de todos os estados futuros possíveis.

        Para a peça atual, testa todas as combinações válidas de
        (rotação × coluna) e extrai as features do board resultante.

        Retorna:
            List[np.ndarray]: Lista de vetores de features, um por
                placement válido. Cada vetor tem shape (4,).
        """
        if self.engine.game_over:
            return []

        piece_type = self.engine.current_piece_type
        rotations = _get_unique_rotations(piece_type)
        current_grid = self.engine.grid

        states = []
        self._valid_placements = []  # Armazena (shape, target_x) para cada estado

        for shape in rotations:
            shape_width = len(shape[0])
            for target_x in range(GRID_LARGURA - shape_width + 1):
                result_grid = self._try_placement(shape, target_x, current_grid, piece_type)
                if result_grid is not None:
                    features = self._compute_features(result_grid)
                    states.append(features)
                    self._valid_placements.append((shape, target_x))

        return states

    def step(self, action_index: int) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
        """
        Executa um placement no tabuleiro.

        Aplica a ação escolhida pelo agente (índice na lista de placements
        válidos), atualiza o estado do jogo e retorna as features dos
        próximos estados possíveis.

        Parâmetros:
            action_index (int): Índice do placement escolhido na lista
                retornada por get_next_states().

        Retorna:
            Tuple contendo:
                - next_states (List[np.ndarray]): Features dos próximos estados.
                - reward (float): Recompensa (linhas eliminadas ao quadrado).
                - done (bool): Se o jogo terminou.
                - info (Dict): Estatísticas do episódio.
        """
        if not self._valid_placements or action_index >= len(self._valid_placements):
            self.game_over = True
            return [], 0.0, True, self._get_info()

        shape, target_x = self._valid_placements[action_index]

        # Aplicar o placement no engine real
        self.engine.current_piece_shape = [row[:] for row in shape]
        self.engine.current_x = target_x
        lines = self.engine.drop_hard()

        self.pieces_placed += 1
        self.lines_cleared += lines
        self.score += self.engine.score - self.score  # Delta score

        # Recompensa: linhas ao quadrado para incentivar Tetris (4 linhas)
        # 0 linhas → 0, 1 linha → 1, 2 → 4, 3 → 9, 4 → 16
        reward = lines ** 2

        # Verificar game over
        done = self.engine.game_over
        if done:
            self.game_over = True

        # Gerar próximos estados
        next_states = [] if done else self.get_next_states()
        if not next_states and not done:
            # Nenhuma posição válida para a próxima peça = game over
            done = True
            self.game_over = True

        return next_states, float(reward), done, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        """
        Retorna informações auxiliares do episódio.

        Retorna:
            Dict[str, Any]: Estatísticas contendo score, linhas e peças.
        """
        return {
            "score": self.engine.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "level": self.engine.level,
        }

    def get_current_board(self) -> List[List[str]]:
        """
        Retorna o estado atual do tabuleiro (para renderização).

        Retorna:
            List[List[str]]: Grid atual do engine.
        """
        return self.engine.grid
