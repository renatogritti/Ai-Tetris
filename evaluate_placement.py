"""
Avaliação do modelo Tetris Placement RL.

Este script testa um modelo treinado com `train_placement.py` usando o ambiente
`TetrisPlacementEnv` e reporta métricas por episódio.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from src.tetris_env_placement import TetrisPlacementEnv


def avaliar_modelo(caminho_modelo: str, episodios: int, render: bool, random_policy: bool):
    if not os.path.exists(caminho_modelo):
        if not caminho_modelo.endswith('.zip'):
            caminho_modelo = caminho_modelo + '.zip'
        if not os.path.exists(caminho_modelo):
            raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")

    model = PPO.load(caminho_modelo)
    env = TetrisPlacementEnv(render_mode='human' if render else None)

    stats: Dict[str, List[float]] = {
        'score': [],
        'lines': [],
        'steps': [],
        'reward': [],
        'placements': []
    }

    for ep in range(episodios):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        stats['score'].append(info['score'])
        stats['lines'].append(info['lines_cleared'])
        stats['steps'].append(steps)
        stats['reward'].append(total_reward)
        stats['placements'].append(info['placement_count'])

        print(
            f"Episódio {ep+1}/{episodios} | "
            f"Score: {info['score']:5d} | "
            f"Linhas: {info['lines_cleared']:3d} | "
            f"Passos: {steps:4d} | "
            f"Placements: {info['placement_count']:3d} | "
            f"Reward: {total_reward:.2f}"
        )

    env.close()
    return stats


def gerar_relatorio(stats: Dict[str, List[float]]):
    score = np.array(stats['score'])
    lines = np.array(stats['lines'])
    steps = np.array(stats['steps'])
    reward = np.array(stats['reward'])
    placements = np.array(stats['placements'])

    print('\n' + '=' * 60)
    print('RELATÓRIO DE AVALIAÇÃO - Tetris Placement')
    print('=' * 60)
    print(f"Score médio: {score.mean():.1f}")
    print(f"Score máximo: {score.max():.1f}")
    print(f"Linhas médias: {lines.mean():.1f}")
    print(f"Linhas totais: {lines.sum():.0f}")
    print(f"Steps médios: {steps.mean():.1f}")
    print(f"Reward médio: {reward.mean():.2f}")
    print(f"Placement médio: {placements.mean():.1f}")
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Avaliação do modelo Tetris Placement RL')
    parser.add_argument('--modelo', type=str, default='./saved_models/tetris_ppo_model_placement.zip', help='Caminho para o modelo salvo')
    parser.add_argument('--episodios', type=int, default=10, help='Número de episódios para avaliação')
    parser.add_argument('--render', action='store_true', help='Renderizar o jogo durante a avaliação')
    parser.add_argument('--random', action='store_true', help='Usar política aleatória em vez do modelo')
    args = parser.parse_args()

    stats = avaliar_modelo(args.modelo, args.episodios, args.render, args.random)
    gerar_relatorio(stats)
