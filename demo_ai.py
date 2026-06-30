"""
Demonstração Visual da IA Tetris (demo_ai.py).

Carrega um modelo DQN treinado e exibe a IA jogando Tetris em tempo real
usando a interface Pygame. O programa roda em loop contínuo, reiniciando
automaticamente após cada game over.

Uso:
    python demo_ai.py                                              # Modelo padrão
    python demo_ai.py --modelo ./saved_models/tetris_dqn_best.pt   # Modelo específico
    python demo_ai.py --velocidade 10                              # Mais rápido
    python demo_ai.py --velocidade 2                               # Mais lento
"""

import argparse
import os
import sys
import time

import numpy as np
import pygame

from src.tetris_engine import TetrisEngine
from src.tetris_gui import TetrisGUI
from src.tetris_env_features import TetrisFeatureEnv
from src.dqn_agent import DQNAgent
import config_train


def rodar_demo(
    caminho_modelo: str,
    velocidade: int = 5,
    episodios: int = 0,
) -> None:
    """
    Executa a demonstração visual da IA jogando Tetris.

    O programa:
        1. Carrega o modelo DQN treinado
        2. Inicializa o ambiente e a GUI
        3. Para cada peça, gera os próximos estados e escolhe o melhor
        4. Renderiza a ação em tempo real com Pygame
        5. Exibe estatísticas no console a cada game over
        6. Reinicia automaticamente (ou para após N episódios)

    Parâmetros:
        caminho_modelo (str): Caminho para o arquivo .pt do modelo.
        velocidade (int): Peças por segundo (1-30, default: 5).
        episodios (int): Número de episódios para rodar (0 = infinito).
    """
    # Verificar arquivo do modelo
    if not os.path.exists(caminho_modelo):
        print(f"Erro: Modelo não encontrado em '{caminho_modelo}'")
        print("Treine um modelo primeiro com: python train_dqn.py")
        sys.exit(1)

    # Carregar agente (sem exploração — epsilon = 0)
    agent = DQNAgent(**config_train.HIPERPARAMETROS_DQN)
    agent.load(caminho_modelo)
    agent.epsilon = 0.0  # Modo determinístico — sempre escolhe o melhor

    print("=" * 60)
    print("  DEMONSTRAÇÃO - TETRIS AI (DQN)")
    print("=" * 60)
    print(f"  Modelo: {caminho_modelo}")
    print(f"  Velocidade: {velocidade} peças/segundo")
    print(f"  Episódios: {'infinito' if episodios == 0 else episodios}")
    print(f"  Controles: ESC para sair, +/- para velocidade")
    print("=" * 60)
    print()

    # Inicializar ambiente
    env = TetrisFeatureEnv()

    # Inicializar GUI
    pygame.init()
    gui = TetrisGUI(env.engine)
    gui.jogo_iniciado = True

    ep_count = 0
    total_scores = []
    total_lines = []

    try:
        while True:
            # Reset
            next_states = env.reset()
            gui.engine = env.engine  # Atualizar referência do engine na GUI
            gui.jogo_iniciado = True

            if not next_states:
                continue

            ep_count += 1

            while True:
                # Processar eventos Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            velocidade = min(30, velocidade + 1)
                            print(f"  Velocidade: {velocidade} peças/s")
                        elif event.key == pygame.K_MINUS:
                            velocidade = max(1, velocidade - 1)
                            print(f"  Velocidade: {velocidade} peças/s")

                # Agente escolhe ação
                action = agent.choose_action(next_states)

                # Executar ação
                next_states, reward, done, info = env.step(action)

                # Renderizar
                gui.renderizar()
                gui.clock.tick(velocidade)

                if done:
                    # Exibir overlay de game over
                    gui.renderizar()
                    pygame.time.wait(800)
                    break

            # Estatísticas do episódio
            score = info["score"]
            lines = info["lines_cleared"]
            pieces = info["pieces_placed"]
            total_scores.append(score)
            total_lines.append(lines)

            print(
                f"  Episódio {ep_count:>3} │ "
                f"Score: {score:>6} │ "
                f"Linhas: {lines:>4} │ "
                f"Peças: {pieces:>4} │ "
                f"Score Médio: {np.mean(total_scores):>8.1f}"
            )

            # Parar após N episódios (se especificado)
            if episodios > 0 and ep_count >= episodios:
                break

    except KeyboardInterrupt:
        print("\n  Demonstração encerrada pelo usuário.")

    finally:
        # Relatório final
        if total_scores:
            print()
            print("=" * 60)
            print("  RELATÓRIO FINAL DA DEMONSTRAÇÃO")
            print("=" * 60)
            print(f"  Episódios jogados: {ep_count}")
            print(f"  Score médio: {np.mean(total_scores):.1f}")
            print(f"  Score máximo: {np.max(total_scores)}")
            print(f"  Linhas médias: {np.mean(total_lines):.1f}")
            print(f"  Linhas máximas: {np.max(total_lines)}")
            print("=" * 60)

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstração Visual da IA Tetris (DQN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python demo_ai.py                                              # Modelo padrão
  python demo_ai.py --modelo ./saved_models/tetris_dqn_best.pt   # Modelo específico
  python demo_ai.py --velocidade 10                              # Mais rápido
  python demo_ai.py --episodios 5                                # Parar após 5 jogos
        """
    )
    parser.add_argument(
        "--modelo", type=str,
        default=os.path.join(config_train.DIRETORIO_MODELOS, "tetris_dqn_best.pt"),
        help="Caminho para o modelo .pt treinado"
    )
    parser.add_argument(
        "--velocidade", type=int, default=5,
        help="Peças por segundo na visualização (1-30, default: 5)"
    )
    parser.add_argument(
        "--episodios", type=int, default=0,
        help="Número de episódios para rodar (0 = infinito, default: 0)"
    )
    args = parser.parse_args()

    rodar_demo(
        caminho_modelo=args.modelo,
        velocidade=args.velocidade,
        episodios=args.episodios,
    )
