# ==============================================================================
#  AI Tetris - Bootstrap do modo manual
#
#  Author: Renato Gritti
#  Descrição: Ponto de entrada principal para o modo manual do jogo Tetris.
# ==============================================================================
"""
Ponto de entrada principal do projeto AI Tetris.

Este módulo inicializa o modo manual do jogo, permitindo que o usuário
controle as peças com teclado. O fluxo principal do projeto é o treinamento
e a demonstração do agente DQN com features heurísticas.
"""

from src.tetris_engine import TetrisEngine
from src.tetris_gui import TetrisGUI


def rodar_modo_manual() -> None:
    """Inicializa e executa o jogo de Tetris controlado pelo teclado."""
    print("Iniciando o Tetris Clássico no modo manual...")
    print("Controles:")
    print("  - Setas Esquerda/Direita: Mover peça")
    print("  - Seta Cima: Rotacionar peça")
    print("  - Seta Baixo: Acelerar descida (Soft Drop)")
    print("  - Espaço: Queda instantânea (Hard Drop)")
    print("  - P: Pausar o jogo")
    print("  - Enter: Iniciar/Reiniciar após Game Over")

    engine: TetrisEngine = TetrisEngine()
    gui: TetrisGUI = TetrisGUI(engine)
    gui.rodar_jogo_manual()


if __name__ == "__main__":
    rodar_modo_manual()
