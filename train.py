"""
Programa Principal de Treinamento de RL (train.py).

Este arquivo é responsável por instanciar o ambiente Gymnasium de Tetris,
criar ou carregar o agente inteligente usando o algoritmo PPO (Stable-Baselines3),
configurar callbacks de monitoramento/avaliação e salvar o modelo treinado.

Suporta três modos de execução via linha de comando:
  1. Treinar um novo modelo do zero (`--train`)
  2. Continuar o treinamento de um checkpoint existente (`--continue <caminho>`)
  3. Assistir o modelo treinado jogando visualmente (`--test <caminho>`)
"""

import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Importação dos módulos locais
from src.tetris_env import TetrisEnv
import config_train


def criar_diretorios() -> None:
    """Cria os diretórios necessários para armazenar os logs e modelos salvos."""
    os.makedirs(config_train.DIRETORIO_MODELOS, exist_ok=True)
    os.makedirs(config_train.DIRETORIO_LOGS, exist_ok=True)


def treinar(caminho_continuar: str = None) -> None:
    """
    Instancia o ambiente, configura logs/callbacks e executa o treinamento do modelo.
    
    Parâmetros:
        caminho_continuar (str): Caminho opcional para carregar um modelo e continuar o treino.
    """
    print("Preparando ambiente de treinamento...")
    criar_diretorios()
    
    # 1. Função auxiliar para instanciar o ambiente com monitoramento (essencial para SB3 logs)
    def make_env():
        render_mode = "human" if config_train.RENDERIZAR_TREINAMENTO else None
        env = TetrisEnv(render_mode=render_mode)
        env = Monitor(env)               # Captura estatísticas de recompensa/tempo
        return env

    # Vectorização do ambiente (SB3 exige ou recomenda Vectorized Environments)
    env = DummyVecEnv([make_env])
    
    # Ambiente separado apenas para as rodadas periódicas de avaliação
    eval_env = DummyVecEnv([lambda: Monitor(TetrisEnv(
        render_mode="human" if config_train.RENDERIZAR_AVALIACAO else None
    ))])

    # 2. Configuração de Callbacks
    # Callback de Avaliação: Avalia o agente periodicamente e salva o melhor modelo obtido
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config_train.DIRETORIO_MODELOS,
        log_path=config_train.DIRETORIO_LOGS,
        eval_freq=config_train.AVALIAR_FREQUENCIA,
        n_eval_episodes=config_train.EPISODIOS_AVALIACAO,
        deterministic=True,
        render=config_train.RENDERIZAR_AVALIACAO
    )
    
    # Callback de Checkpoint: Salva o estado do modelo periodicamente para prevenção de falhas
    checkpoint_callback = CheckpointCallback(
        save_freq=config_train.SALVAR_CHECKPOINT_FREQUENCIA,
        save_path=os.path.join(config_train.DIRETORIO_MODELOS, "checkpoints"),
        name_prefix="tetris_ppo"
    )
    
    callbacks = [eval_callback, checkpoint_callback]

    # 3. Inicialização ou Carregamento do Modelo
    if caminho_continuar:
        print(f"Carregando modelo existente de '{caminho_continuar}' para continuar o treinamento...")
        model = PPO.load(caminho_continuar, env=env)
    else:
        print(f"Criando novo modelo PPO do zero com os parâmetros de 'config_train.py'...")
        model = PPO(
            policy=config_train.TIPO_POLITICA,
            env=env,
            tensorboard_log=config_train.DIRETORIO_LOGS,
            **config_train.HIPERPARAMETROS_PPO
        )

    # 4. Início do Treinamento
    print(f"Treinamento iniciado! Total de passos configurado: {config_train.TOTAL_TIMESTEPS}")
    print(f"Logs do TensorBoard serão gravados em: {config_train.DIRETORIO_LOGS}")
    print("Para monitorar em tempo real, execute: tensorboard --logdir=" + config_train.DIRETORIO_LOGS)
    
    try:
        model.learn(
            total_timesteps=config_train.TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name="PPO_Tetris"
        )
        
        # Salva o modelo final treinado
        caminho_final = os.path.join(config_train.DIRETORIO_MODELOS, config_train.NOME_MODELO)
        model.save(caminho_final)
        print(f"\nTreinamento concluído com sucesso! Modelo final salvo em: {caminho_final}")
        
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário. Salvando estado atual do modelo...")
        caminho_interrompido = os.path.join(config_train.DIRETORIO_MODELOS, f"{config_train.NOME_MODELO}_interrompido")
        model.save(caminho_interrompido)
        print(f"Modelo de salvamento de emergência salvo em: {caminho_interrompido}")


def testar(caminho_modelo: str) -> None:
    """
    Carrega um modelo treinado e o executa em modo interativo visual ("human")
    para que o usuário possa assistir a IA jogando.
    
    Parâmetros:
        caminho_modelo (str): Caminho para o arquivo do modelo .zip treinado.
    """
    print(f"Carregando modelo treinado de '{caminho_modelo}'...")
    if not os.path.exists(caminho_modelo) and not caminho_modelo.endswith(".zip"):
        # Adiciona extensão se necessário
        caminho_modelo += ".zip"
        
    if not os.path.exists(caminho_modelo):
        print(f"Erro: Arquivo do modelo '{caminho_modelo}' não foi encontrado.")
        return

    # Inicializa o ambiente com exibição em tempo real
    env = TetrisEnv(render_mode="human")
    
    # Carrega a rede neural treinada
    model = PPO.load(caminho_modelo)
    
    obs, info = env.reset()
    terminado = False
    truncado = False
    score_total = 0
    linhas_totais = 0
    
    print("IA jogando! Pressione Ctrl+C no console para fechar.")
    
    try:
        while not (terminado or truncado):
            # A IA prevê a melhor ação de forma determinística
            acao, _ = model.predict(obs, deterministic=True)
            
            # Avança o estado do jogo
            # O ambiente cuida da renderização visual interna caso o render_mode seja 'human'
            obs, recompensa, terminado, truncado, info = env.step(int(acao))
            
            score_total = info["score"]
            linhas_totais = info["lines_cleared"]
            
        print("\nFim do episódio!")
        print(f"Pontuação máxima: {score_total}")
        print(f"Total de linhas limpas pela IA: {linhas_totais}")
        
    except KeyboardInterrupt:
        print("\nVisualização encerrada pelo usuário.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de Reinforcement Learning para AI Tetris.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Inicia um novo treinamento do zero.")
    group.add_argument("--continue-train", type=str, metavar="CAMINHO_MODELO", help="Continua o treinamento de um modelo salvo.")
    group.add_argument("--test", type=str, metavar="CAMINHO_MODELO", help="Assiste um modelo treinado jogando na tela.")
    
    args = parser.parse_args()
    
    if args.train:
        treinar()
    elif args.continue_train:
        treinar(caminho_continuar=args.continue_train)
    elif args.test:
        testar(caminho_modelo=args.test)
