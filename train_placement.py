"""
Treinamento de RL para Tetris usando ações de posicionamento final.

Este arquivo implementa um fluxo de treino do zero que usa o ambiente
`TetrisPlacementEnv`, onde cada ação escolhe a rotação e a coluna final da peça.
"""

import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.tetris_env_placement import TetrisPlacementEnv
import config_train


def criar_diretorios() -> None:
    os.makedirs(config_train.DIRETORIO_MODELOS, exist_ok=True)
    os.makedirs(config_train.DIRETORIO_LOGS, exist_ok=True)


def make_env(render_mode: str = None):
    def _init():
        env = TetrisPlacementEnv(render_mode=render_mode)
        return Monitor(env)
    return _init


def treinar(caminho_continuar: str = None) -> None:
    criar_diretorios()
    print("Preparando ambiente de treinamento de posicionamento...")

    env = DummyVecEnv([make_env(render_mode=None)])
    eval_env = DummyVecEnv([make_env(render_mode=None)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config_train.DIRETORIO_MODELOS,
        log_path=config_train.DIRETORIO_LOGS,
        eval_freq=config_train.AVALIAR_FREQUENCIA,
        n_eval_episodes=config_train.EPISODIOS_AVALIACAO,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config_train.SALVAR_CHECKPOINT_FREQUENCIA,
        save_path=os.path.join(config_train.DIRETORIO_MODELOS, "checkpoints"),
        name_prefix="tetris_placement_ppo"
    )

    callbacks = [eval_callback, checkpoint_callback]

    if caminho_continuar:
        print(f"Carregando modelo existente de '{caminho_continuar}'...")
        model = PPO.load(caminho_continuar, env=env)
    else:
        print("Criando novo modelo PPO para posicionamento...")
        model = PPO(
            policy=config_train.TIPO_POLITICA,
            env=env,
            tensorboard_log=config_train.DIRETORIO_LOGS,
            **config_train.HIPERPARAMETROS_PPO
        )

    print(f"Treinamento iniciado: {config_train.TOTAL_TIMESTEPS} timesteps")
    model.learn(total_timesteps=config_train.TOTAL_TIMESTEPS, callback=callbacks, tb_log_name="PPO_Tetris_Placement")

    caminho_final = os.path.join(config_train.DIRETORIO_MODELOS, f"{config_train.NOME_MODELO}_placement")
    model.save(caminho_final)
    print(f"Treinamento finalizado. Modelo salvo em: {caminho_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento Tetris Placement RL")
    parser.add_argument("--train", action="store_true", help="Treinar do zero")
    parser.add_argument("--continue-train", type=str, help="Continuar treino de um modelo salvo")
    args = parser.parse_args()

    if args.train:
        treinar()
    elif args.continue_train:
        treinar(caminho_continuar=args.continue_train)
    else:
        parser.print_help()
