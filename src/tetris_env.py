"""
Módulo do Ambiente de Reinforcement Learning (TetrisEnv).

Este módulo implementa um ambiente compatível com a biblioteca Gymnasium (padrão de mercado)
para permitir o treinamento de agentes de Aprendizado por Reforço no jogo Tetris.
Suporta renderização visual ("human") usando o Pygame e execução headless rápida para treinamento.
"""

from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.config import (
    GRID_LARGURA,
    GRID_ALTURA,
    RL_PESO_LINHAS_LIMPAS,
    RL_PESO_ALTURA,
    RL_PESO_BURACOS,
    RL_PESO_BUMPINESS,
    RL_PESO_PERDA,
    RL_RECOMPENSA_PASSO,
    RL_PESO_DENSIDADE_LINHAS
)
from src.tetris_engine import TetrisEngine
from src.tetris_gui import TetrisGUI

# Mapeamento clássico de tipos de peça para inteiros
PEÇA_PARA_INT: Dict[str, int] = {
    "": 0,
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7
}

class TetrisEnv(gym.Env):
    """
    Ambiente Gymnasium personalizado para treinar IAs no jogo Tetris.
    
    Ações (Discretas):
        0: Nenhuma ação (Apenas deixa cair)
        1: Mover para a esquerda
        2: Mover para a direita
        3: Rotacionar (sentido horário)
        4: Queda suave (Soft Drop - desce 1 bloco)
        5: Queda rápida (Hard Drop - desce tudo e trava)
        
    Observações (Dict):
        - "grid" (Box 20x10): Tabuleiro binário (1.0 para ocupado, 0.0 para vazio).
        - "current_piece" (Discrete): Código numérico da peça atual (1 a 7).
        - "next_piece" (Discrete): Código numérico da próxima peça (1 a 7).

    Truncamento:
        O episódio é encerrado (truncado) ao atingir MAX_PASSOS_POR_EPISODIO passos,
        mesmo sem Game Over, para garantir diversidade de experiências no treinamento.
        
    Metadata:
        render_modes: ["human", "rgb_array"]
        render_fps: 60
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    # Limite máximo de passos por episódio (importado do módulo de configuração de treino)
    MAX_PASSOS: int = 3_000
    
    def __init__(self, render_mode: Optional[str] = None) -> None:
        """
        Inicializa o ambiente Gymnasium do Tetris.
        
        Parâmetros:
            render_mode (Optional[str]): Modo de renderização ("human" para assistir, "rgb_array" para capturar pixels, None para headless).
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.engine = TetrisEngine()
        self.gui: Optional[TetrisGUI] = None
        
        # Contador de passos no episódio atual para controle de truncamento
        self._passos_no_episodio: int = 0
        
        # Define o espaço de ações: 6 ações discretas
        self.action_space = spaces.Discrete(6)
        
        # Define o espaço de observações como um dicionário estruturado
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=1.0, shape=(GRID_ALTURA, GRID_LARGURA), dtype=np.float32
            ),
            "current_piece": spaces.Discrete(8),  # 0 a 7
            "next_piece": spaces.Discrete(8)     # 0 a 7
        })

    def _get_obs(self) -> Dict[str, Any]:
        """
        Gera a observação atual no formato definido pelo observation_space.
        
        Retorna:
            Dict[str, Any]: O estado atualizado do jogo formatado.
        """
        # Converte a grade de string do motor lógico para uma matriz binária float32
        grid_binaria = np.zeros((GRID_ALTURA, GRID_LARGURA), dtype=np.float32)
        for r in range(GRID_ALTURA):
            for c in range(GRID_LARGURA):
                if self.engine.grid[r][c] != "":
                    grid_binaria[r, c] = 1.0

        return {
            "grid": grid_binaria,
            "current_piece": PEÇA_PARA_INT.get(self.engine.current_piece_type, 0),
            "next_piece": PEÇA_PARA_INT.get(self.engine.next_piece_type, 0)
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Gera informações auxiliares que não são observadas diretamente pelo agente.
        
        Retorna:
            Dict[str, Any]: Informações extras (pontuação, nível, etc.).
        """
        return {
            "score": self.engine.score,
            "level": self.engine.level,
            "lines_cleared": self.engine.lines_cleared_total
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reseta o jogo para o estado inicial.
        
        Parâmetros:
            seed (Optional[int]): Semente aleatória para reprodutibilidade.
            options (Optional[dict]): Parâmetros adicionais para reset.
            
        Retorna:
            Tuple[Dict[str, Any], Dict[str, Any]]: Observação inicial e informações extras.
        """
        # Inicializa a semente aleatória herdada da classe pai
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
            
        # Se o jogo estava em Game Over e estamos em modo visual,
        # mostra o overlay de Game Over por 1 segundo para fins educacionais antes de limpar o tabuleiro.
        if self.engine.game_over and self.render_mode == "human" and self.gui is not None:
            self.gui.renderizar()
            pygame.time.wait(1000)  # Pausa de 1 segundo do Pygame
            
        self.engine.reset()
        
        # Zera o contador de passos do episódio
        self._passos_no_episodio = 0
        
        # Se houver interface ativa e estiver rodando, reseta as flags dela
        if self.gui is not None:
            self.gui.jogo_iniciado = True
            self.gui.pausado = False
            
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Avança um passo de tempo no ambiente executando a ação do agente.
        
        Parâmetros:
            action (int): Ação discreta a ser tomada (0 a 5).
            
        Retorna:
            Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
                - observacao: Estado atualizado.
                - recompensa: Valor numérico da recompensa da ação.
                - terminado: Indica se houve Game Over.
                - truncado: Indica se o limite de tempo/passos foi atingido.
                - info: Estatísticas de depuração.
        """
        if self.engine.game_over:
            return self._get_obs(), 0.0, True, False, self._get_info()

        # Incrementa o contador de passos do episódio
        self._passos_no_episodio += 1

        # Armazena estado anterior para cálculo de recompensa baseada em delta
        linhas_antes = self.engine.lines_cleared_total
        
        alturas_antes = self.engine.get_column_heights()
        altura_max_antes = max(alturas_antes) if alturas_antes else 0
        buracos_antes = self.engine.get_holes_count()
        bumpiness_antes = self.engine.get_bumpiness()
        densidade_antes = self.engine.get_row_density_score()

        # Executa a ação recebida no motor lógico
        peca_travou = False
        if action == 1:
            self.engine.move(-1, 0)
        elif action == 2:
            self.engine.move(1, 0)
        elif action == 3:
            self.engine.rotate()
        elif action == 4:
            # Drop suave retorna False se a peça travou na base
            peca_travou = not self.engine.drop_soft()
        elif action == 5:
            # Queda rápida sempre trava a peça
            self.engine.drop_hard()
            peca_travou = True
            
        # Se o agente não executou uma ação de descida direta (4 ou 5),
        # aplica a gravidade/queda clássica de 1 bloco a cada ação para avançar o jogo.
        if action not in [4, 5]:
            peca_travou = not self.engine.drop_soft()

        # Calcula a diferença de linhas limpas nessa jogada
        linhas_limpas_turno = self.engine.lines_cleared_total - linhas_antes

        # Calcula heurísticas depois do passo
        alturas_depois = self.engine.get_column_heights()
        altura_max_depois = max(alturas_depois) if alturas_depois else 0
        buracos_depois = self.engine.get_holes_count()
        bumpiness_depois = self.engine.get_bumpiness()
        densidade_depois = self.engine.get_row_density_score()

        # Calcula os deltas (depois - antes)
        delta_altura = altura_max_depois - altura_max_antes
        delta_buracos = buracos_depois - buracos_antes
        delta_bumpiness = bumpiness_depois - bumpiness_antes
        delta_densidade = densidade_depois - densidade_antes

        # Função de Recompensa Heurística baseada em DELTA
        # - Penaliza ações que aumentam altura, criam buracos ou aumentam irregularidade.
        # - Recompensa ações que aproximam linhas da completude (densidade quadrática).
        # - Evita o problema de "suicídio antecipado" da IA (penalidade contínua de altura absoluta).
        recompensa = (
            (linhas_limpas_turno * RL_PESO_LINHAS_LIMPAS) +
            (delta_altura * RL_PESO_ALTURA) +
            (delta_buracos * RL_PESO_BURACOS) +
            (delta_bumpiness * RL_PESO_BUMPINESS) +
            (delta_densidade * RL_PESO_DENSIDADE_LINHAS) +
            RL_RECOMPENSA_PASSO
        )

        # Se perdeu o jogo, aplica uma penalidade severa
        terminado = self.engine.game_over
        if terminado:
            recompensa += RL_PESO_PERDA

        # Verifica se o episódio atingiu o limite máximo de passos (truncamento)
        truncado = self._passos_no_episodio >= self.MAX_PASSOS

        # Renderização visual se habilitada
        if self.render_mode is not None:
            self.render()

        return self._get_obs(), float(recompensa), terminado, truncado, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        """
        Gera a saída visual do jogo.
        
        Retorna:
            Optional[np.ndarray]: Matriz de pixels 3D se render_mode for 'rgb_array', None caso contrário.
        """
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = TetrisGUI(self.engine)
                self.gui.jogo_iniciado = True
            
            # Processa a fila de eventos do Pygame para manter a janela responsiva
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
        """Encerra a interface do Pygame de forma segura ao encerrar o ambiente."""
        if self.gui is not None:
            pygame.quit()
            self.gui = None
