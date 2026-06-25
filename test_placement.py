"""
Testa visualmente o modelo de posicionamento do Tetris.

Este arquivo carrega um modelo salvo treinado com `train_placement.py`
e executa o agente em modo `human` usando o ambiente `TetrisPlacementEnv`.
"""

import argparse
import os

from stable_baselines3 import PPO

from src.tetris_env_placement import TetrisPlacementEnv


def testar_modelo(caminho_modelo: str, deterministic: bool = True) -> None:
    if not caminho_modelo.endswith('.zip'):
        caminho_modelo = caminho_modelo + '.zip'

    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")

    print(f"Carregando modelo: {caminho_modelo}")
    model = PPO.load(caminho_modelo)

    env = TetrisPlacementEnv(render_mode='human')
    obs, info = env.reset()

    print("Executando modelo visualmente. Pressione Ctrl+C para encerrar.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                print("\nEpisódio finalizado")
                print(f"Score: {info['score']}")
                print(f"Linhas limpas: {info['lines_cleared']}")
                print(f"Placements: {info.get('placement_count', 'N/A')}")
                break
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualiza o modelo Tetris Placement')
    parser.add_argument('--modelo', type=str, default='./saved_models/tetris_ppo_model_placement.zip', help='Caminho para o modelo salvo')
    parser.add_argument('--random', action='store_true', help='Usar uma política aleatória em vez do modelo')
    args = parser.parse_args()

    if args.random:
        raise NotImplementedError('Política aleatória ainda não está disponível para visualização.')

    testar_modelo(args.modelo)
