"""
Módulo de Configuração do Jogo Tetris.

Este arquivo contém todas as constantes, definições de cores,
formatos das peças (Tetrominos), parâmetros de jogabilidade e pesos
de recompensa para o treinamento de aprendizado por reforço (RL).
Todos os valores podem ser ajustados diretamente aqui.
"""

from typing import Dict, List, Tuple

# ==============================================================================
# CONFIGURAÇÕES DA GRADE DO JOGO (GRID)
# ==============================================================================
GRID_LARGURA: int = 10
"""Número de colunas no tabuleiro de Tetris."""

GRID_ALTURA: int = 20
"""Número de linhas no tabuleiro de Tetris."""

TAMANHO_BLOCO: int = 35
"""Tamanho em pixels de cada bloco individual (quadrado)."""

# ==============================================================================
# CONFIGURAÇÕES DA JANELA E INTERFACE (GUI)
# ==============================================================================
LARGURA_JANELA: int = 650
"""Largura total da janela do jogo em pixels."""

ALTURA_JANELA: int = 780
"""Altura total da janela do jogo em pixels."""

FPS: int = 60
"""Quadros por segundo da interface Pygame."""

FONTE_FAMILIA: str = "Outfit"
"""Família da fonte utilizada no jogo (será carregada do Google Fonts ou fallback)."""

# ==============================================================================
# CORES (PALETA PREMIUM NEON/DARK)
# ==============================================================================
# Cores da interface geral
COR_FUNDO: Tuple[int, int, int] = (15, 15, 20)        # Azul escuro profundo
COR_GRID: Tuple[int, int, int] = (25, 25, 35)         # Linhas internas do grid
COR_PAREDE: Tuple[int, int, int] = (45, 45, 60)       # Borda externa do tabuleiro
COR_PAINEL: Tuple[int, int, int] = (25, 25, 35)       # Painéis de pontuação e próxima peça
COR_TEXTO: Tuple[int, int, int] = (240, 240, 245)     # Branco acinzentado suave
COR_TEXTO_MUTED: Tuple[int, int, int] = (130, 130, 150) # Cinza para textos secundários
COR_SOMBRA: Tuple[int, int, int] = (5, 5, 10)         # Efeito de sombra projetada

# Cores vibrantes neon para cada tipo de Tetromino
CORES_PEÇAS: Dict[str, Tuple[int, int, int]] = {
    "I": (0, 240, 240),     # Ciano Neon
    "O": (240, 240, 0),     # Amarelo Neon
    "T": (160, 0, 240),     # Roxo Neon
    "S": (0, 240, 0),       # Verde Neon
    "Z": (240, 0, 0),       # Vermelho Neon
    "J": (0, 0, 240),       # Azul Neon
    "L": (240, 120, 0)      # Laranja Neon
}

# ==============================================================================
# FORMATOS E MATRIZES DOS TETROMINOS
# ==============================================================================
# Os formatos são matrizes binárias. 1 representa bloco, 0 representa vazio.
# Usamos matrizes quadradas para facilitar o algoritmo de rotação por transposição.
FORMATOS_PEÇAS: Dict[str, List[List[int]]] = {
    "I": [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ],
    "O": [
        [1, 1],
        [1, 1]
    ],
    "T": [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ],
    "S": [
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ],
    "Z": [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ],
    "J": [
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ],
    "L": [
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ]
}

# ==============================================================================
# PARÂMETROS DE JOGABILIDADE
# ==============================================================================
VELOCIDADE_INICIAL: float = 0.8  # Segundos por queda natural
DIFICULDADE_DECRESCIMO_VELOCIDADE: float = 0.05  # Redução de tempo por nível
VELOCIDADE_MINIMA: float = 0.1  # Velocidade máxima real (menor tempo por queda)
LINHAS_POR_NIVEL: int = 10  # Quantas linhas limpar para subir o nível

# Pontuação clássica do NES/Sega Tetris multiplicada
PONTOS_POR_LINHAS: Dict[int, int] = {
    0: 0,
    1: 40,
    2: 100,
    3: 300,
    4: 1200
}

# ==============================================================================
# PARÂMETROS E PESOS PARA REINFORCEMENT LEARNING (IA)
# ==============================================================================
# Estes pesos determinam o cálculo da recompensa no wrapper do Gymnasium.
# A recompensa é calculada com base nos DELTAS (mudanças após cada ação), não em valores absolutos.
# Isso evita que a IA "prefira morrer" para escapar de penalidades contínuas de altura.
RL_PESO_LINHAS_LIMPAS: float = 10.0    # Recompensa alta por limpar linhas (comportamento desejado)
RL_PESO_ALTURA: float = -1.0           # Penalidade por AUMENTAR a altura máxima
RL_PESO_BURACOS: float = -3.0          # Penalidade por CRIAR novos buracos (inacessíveis)
RL_PESO_BUMPINESS: float = -0.5        # Penalidade por AUMENTAR a irregularidade entre colunas
RL_PESO_PERDA: float = -5.0            # Penalidade ao perder o jogo (Game Over)
RL_RECOMPENSA_PASSO: float = 0.001     # Recompensa pequena por sobreviver a cada passo

RL_PESO_DENSIDADE_LINHAS: float = 0.3
"""
Peso para a recompensa de densidade das linhas.

Esta recompensa é calculada como o DELTA (variação) da pontuação de densidade
de todas as linhas entre o estado antes e depois da ação. Isso cria um sinal
de aprendizado gradual que orienta a IA a construir estruturas densas e planas,
mesmo antes de conseguir limpar uma linha completa.

Funcionamento do score por linha:
  - Fração preenchida ao quadrado → linhas quase completas valem muito mais!
  - 5/10 preenchidas → 0.25 pontos
  - 9/10 preenchidas → 0.81 pontos (3x mais!)
  - 10/10 (linha completa) → 1.00 pontos (antes de ser eliminada)
"""
