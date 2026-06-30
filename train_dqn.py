"""
Script de Treinamento DQN para Tetris (train_dqn.py).

Treina um agente DQN a jogar Tetris usando features heurísticas do tabuleiro.
O treinamento exibe métricas detalhadas no console para acompanhamento em
tempo real da convergência do modelo.

Métricas reportadas a cada N episódios:
    - Score médio, máximo e mínimo
    - Linhas eliminadas (média e máximo)
    - Peças colocadas (média)
    - Epsilon atual
    - Loss média do DQN

Uso:
    python train_dqn.py                     # Treinar com defaults (3000 episódios)
    python train_dqn.py --episodes 5000     # Treinar com mais episódios
    python train_dqn.py --render            # Treinar com visualização (lento)
    python train_dqn.py --continue-from ./saved_models/tetris_dqn_best.pt
"""

import os
import csv
import time
import argparse
from typing import List, Optional, Dict

import numpy as np

from src.tetris_env_features import TetrisFeatureEnv
from src.dqn_agent import DQNAgent
import config_train


def criar_diretorios() -> None:
    """Cria os diretórios de saída se não existirem."""
    os.makedirs(config_train.DIRETORIO_MODELOS, exist_ok=True)
    os.makedirs(config_train.DIRETORIO_LOGS, exist_ok=True)


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho formatado do treinamento."""
    print("=" * 90)
    print("  TREINAMENTO DQN - TETRIS AI (Features Heurísticas)")
    print("=" * 90)
    print(f"  Episódios: {config_train.TOTAL_EPISODIOS}")
    print(f"  Batch size: {config_train.HIPERPARAMETROS_DQN['batch_size']}")
    print(f"  Learning rate: {config_train.HIPERPARAMETROS_DQN['learning_rate']}")
    print(f"  Epsilon: {config_train.HIPERPARAMETROS_DQN['epsilon_start']}"
          f" → {config_train.HIPERPARAMETROS_DQN['epsilon_end']}"
          f" ({config_train.HIPERPARAMETROS_DQN['epsilon_decay_episodes']} eps)")
    print(f"  Replay buffer: {config_train.HIPERPARAMETROS_DQN['buffer_capacity']}")
    print(f"  Gamma: {config_train.HIPERPARAMETROS_DQN['gamma']}")
    print("=" * 90)
    print()
    print(f"{'Ep':>6} │ {'Score Méd':>10} │ {'Score Máx':>10} │ {'Linhas Méd':>11} │"
          f" {'Peças Méd':>10} │ {'Epsilon':>8} │ {'Loss':>8} │ {'Tempo':>6}")
    print("─" * 90)


def imprimir_metricas(
    ep: int,
    scores: List[int],
    lines: List[int],
    pieces: List[int],
    epsilon: float,
    losses: List[float],
    tempo_bloco: float,
) -> None:
    """
    Imprime as métricas de um bloco de episódios.

    Parâmetros:
        ep (int): Episódio atual.
        scores (List[int]): Scores dos episódios do bloco.
        lines (List[int]): Linhas eliminadas por episódio.
        pieces (List[int]): Peças colocadas por episódio.
        epsilon (float): Epsilon atual.
        losses (List[float]): Losses do treinamento no bloco.
        tempo_bloco (float): Tempo em segundos do bloco.
    """
    score_med = np.mean(scores)
    score_max = np.max(scores)
    lines_med = np.mean(lines)
    pieces_med = np.mean(pieces)
    loss_med = np.mean(losses) if losses else 0.0

    print(
        f"{ep:>6} │ {score_med:>10.1f} │ {score_max:>10} │ {lines_med:>11.1f} │"
        f" {pieces_med:>10.1f} │ {epsilon:>8.4f} │ {loss_med:>8.4f} │ {tempo_bloco:>5.1f}s"
    )


def salvar_log_csv(
    caminho: str,
    ep: int,
    score_med: float,
    score_max: int,
    lines_med: float,
    pieces_med: float,
    epsilon: float,
    loss_med: float,
) -> None:
    """
    Salva uma linha de métricas no arquivo CSV de log.

    Parâmetros:
        caminho (str): Caminho do arquivo CSV.
        ep (int): Episódio atual.
        score_med (float): Score médio do bloco.
        score_max (int): Score máximo do bloco.
        lines_med (float): Linhas médias do bloco.
        pieces_med (float): Peças médias do bloco.
        epsilon (float): Epsilon atual.
        loss_med (float): Loss média do bloco.
    """
    file_exists = os.path.exists(caminho)
    with open(caminho, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episodio", "score_med", "score_max", "lines_med",
                             "pieces_med", "epsilon", "loss_med"])
        writer.writerow([ep, f"{score_med:.1f}", score_max, f"{lines_med:.1f}",
                         f"{pieces_med:.1f}", f"{epsilon:.4f}", f"{loss_med:.6f}"])


def treinar(
    total_episodios: int,
    caminho_continuar: Optional[str] = None,
    render: bool = False,
) -> None:
    """
    Executa o loop principal de treinamento do agente DQN.

    O treinamento segue este fluxo para cada episódio:
        1. Reset do ambiente → receber estados iniciais possíveis
        2. Agente escolhe ação (epsilon-greedy)
        3. Ambiente executa placement → retorna próximos estados
        4. Armazena transição no replay buffer
        5. Treina a rede com minibatch do buffer
        6. Repete até game over
        7. Atualiza target network e decai epsilon

    Parâmetros:
        total_episodios (int): Número total de episódios de treinamento.
        caminho_continuar (Optional[str]): Caminho para modelo pré-treinado.
        render (bool): Se True, renderiza o jogo durante o treino (lento!).
    """
    criar_diretorios()

    # Inicializar agente
    agent = DQNAgent(**config_train.HIPERPARAMETROS_DQN)

    if caminho_continuar:
        print(f"Carregando modelo pré-treinado: {caminho_continuar}")
        agent.load(caminho_continuar)

    # Inicializar ambiente
    env = TetrisFeatureEnv()

    # GUI opcional para renderização durante treinamento
    gui = None
    if render:
        from src.tetris_engine import TetrisEngine
        from src.tetris_gui import TetrisGUI
        import pygame

    # Tracking de métricas
    melhor_score_medio = -float("inf")
    bloco_scores: List[int] = []
    bloco_lines: List[int] = []
    bloco_pieces: List[int] = []
    bloco_losses: List[float] = []

    csv_path = os.path.join(config_train.DIRETORIO_LOGS, "training_log.csv")
    imprimir_cabecalho()
    tempo_inicio_bloco = time.time()

    for episodio in range(1, total_episodios + 1):
        # Reset do ambiente
        next_states = env.reset()

        if not next_states:
            continue

        # Estado atual = features do board vazio (sem placement ainda)
        current_features = np.zeros(agent.num_features, dtype=np.float32)
        total_reward = 0.0

        # Renderização (se habilitada)
        if render and gui is None:
            gui = TetrisGUI(env.engine)
            gui.jogo_iniciado = True

        while True:
            # Agente escolhe ação
            action = agent.choose_action(next_states)

            # Features do estado escolhido
            chosen_state = next_states[action]

            # Executar ação no ambiente
            next_states, reward, done, info = env.step(action)
            total_reward += reward

            # Próximo estado para o buffer
            if done:
                next_state_for_buffer = chosen_state  # Terminal
            else:
                # O "next_state" para DQN é o melhor estado possível
                # da próxima jogada (avaliado pela target net)
                if next_states:
                    import torch
                    with torch.no_grad():
                        ns_tensor = torch.FloatTensor(np.array(next_states)).to(agent.device)
                        q_vals = agent.target_net(ns_tensor)
                        best_idx = int(torch.argmax(q_vals).item())
                    next_state_for_buffer = next_states[best_idx]
                else:
                    next_state_for_buffer = chosen_state
                    done = True

            # Armazenar transição
            agent.store(chosen_state, reward, next_state_for_buffer, done)

            # Treinar rede
            loss = agent.train_step()
            if loss is not None:
                bloco_losses.append(loss)

            # Renderizar (se habilitado)
            if render and gui is not None:
                import pygame
                pygame.event.pump()
                gui.renderizar()
                gui.clock.tick(30)

            if done:
                break

        # Atualizar target network a cada episódio
        agent.sync_target_network()

        # Decair epsilon
        agent.decay_epsilon(episodio)

        # Registrar métricas
        bloco_scores.append(info["score"])
        bloco_lines.append(info["lines_cleared"])
        bloco_pieces.append(info["pieces_placed"])

        # Reportar métricas a cada N episódios
        if episodio % config_train.FREQUENCIA_REPORT == 0:
            tempo_bloco = time.time() - tempo_inicio_bloco

            imprimir_metricas(
                episodio, bloco_scores, bloco_lines, bloco_pieces,
                agent.epsilon, bloco_losses, tempo_bloco
            )

            # Salvar log CSV
            salvar_log_csv(
                csv_path, episodio,
                float(np.mean(bloco_scores)), int(np.max(bloco_scores)),
                float(np.mean(bloco_lines)), float(np.mean(bloco_pieces)),
                agent.epsilon, float(np.mean(bloco_losses)) if bloco_losses else 0.0
            )

            # Salvar melhor modelo
            score_medio_atual = float(np.mean(bloco_scores))
            if score_medio_atual > melhor_score_medio:
                melhor_score_medio = score_medio_atual
                caminho_melhor = os.path.join(
                    config_train.DIRETORIO_MODELOS, "tetris_dqn_best.pt"
                )
                agent.save(caminho_melhor)

            # Reset dos contadores do bloco
            bloco_scores = []
            bloco_lines = []
            bloco_pieces = []
            bloco_losses = []
            tempo_inicio_bloco = time.time()

        # Checkpoint periódico
        if episodio % config_train.SALVAR_CHECKPOINT_FREQUENCIA == 0:
            caminho_ckpt = os.path.join(
                config_train.DIRETORIO_MODELOS,
                "checkpoints",
                f"tetris_dqn_ep{episodio}.pt"
            )
            os.makedirs(os.path.dirname(caminho_ckpt), exist_ok=True)
            agent.save(caminho_ckpt)

    # Salvar modelo final
    caminho_final = os.path.join(config_train.DIRETORIO_MODELOS, "tetris_dqn_final.pt")
    agent.save(caminho_final)

    print()
    print("=" * 90)
    print(f"  Treinamento concluído! ({total_episodios} episódios)")
    print(f"  Melhor score médio: {melhor_score_medio:.1f}")
    print(f"  Modelo final: {caminho_final}")
    print(f"  Melhor modelo: {os.path.join(config_train.DIRETORIO_MODELOS, 'tetris_dqn_best.pt')}")
    print(f"  Log CSV: {csv_path}")
    print("=" * 90)

    if render and gui is not None:
        import pygame
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treinamento DQN para Tetris AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python train_dqn.py                          # Treinar com defaults
  python train_dqn.py --episodes 5000          # Mais episódios
  python train_dqn.py --render                 # Visualizar treinamento
  python train_dqn.py --continue-from modelo.pt  # Continuar treino
        """
    )
    parser.add_argument(
        "--episodes", type=int, default=config_train.TOTAL_EPISODIOS,
        help=f"Número de episódios de treinamento (default: {config_train.TOTAL_EPISODIOS})"
    )
    parser.add_argument(
        "--continue-from", type=str, default=None,
        help="Caminho para modelo .pt para continuar treinamento"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Renderizar o jogo durante o treinamento (muito lento!)"
    )
    args = parser.parse_args()

    treinar(
        total_episodios=args.episodes,
        caminho_continuar=args.continue_from,
        render=args.render,
    )
