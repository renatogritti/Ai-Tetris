"""
Módulo de Configuração para o Treinamento de IA (config_train.py).

Este arquivo reúne todos os parâmetros e hiperparâmetros necessários para o
treinamento de Reinforcement Learning usando a biblioteca Stable-Baselines3.
Permite ajustar facilmente políticas, taxas de aprendizado, diretórios de salvamento e logs.
"""

from typing import Dict, Any

# ==============================================================================
# ALGORITMO E ARQUITETURA DE REDE
# ==============================================================================
ALGORITMO: str = "PPO"
"""Algoritmo de RL a ser utilizado. PPO (Proximal Policy Optimization) é recomendado por sua estabilidade."""

TIPO_POLITICA: str = "MultiInputPolicy"
"""
Tipo de política do Stable-Baselines3. 
Usamos 'MultiInputPolicy' pois nosso ambiente expõe uma observação do tipo dicionário (Dict),
contendo a grade bidimensional (Box) e as peças (Discrete).
"""

# ==============================================================================
# HIPERPARÂMETROS DO MODELO (PPO)
# ==============================================================================
HIPERPARAMETROS_PPO: Dict[str, Any] = {
    "learning_rate": 0.0005,       # Taxa de aprendizado inicial (aumentado para melhor convergência)
    "n_steps": 2048,               # Número de passos para rodar por vetor de ambiente antes de atualizar
    "batch_size": 128,             # Tamanho do lote aumentado para 128 (melhor estabilidade)
    "n_epochs": 15,                # Aumentado para 15 épocas (mais refinamento)
    "gamma": 0.99,                 # Fator de desconto para recompensas futuras
    "gae_lambda": 0.95,            # Fator para trade-off de viés-variância na estimativa do GAE
    "clip_range": 0.2,             # Parâmetro de clipagem do PPO
    "ent_coef": 0.08,              # AUMENTADO para 0.08 (mais exploração, menos convergência precoce)
    "verbose": 1                   # Nível de log detalhado no console (1 = estatísticas básicas)
}

# ==============================================================================
# PARÂMETROS DE EXECUÇÃO E DURABILIDADE
# ==============================================================================
TOTAL_TIMESTEPS: int = 500_000
"""Número total de interações (passos) com o ambiente durante todo o treinamento."""

MAX_PASSOS_POR_EPISODIO: int = 3_000
"""
Número máximo de passos por episódio. Ao atingir este limite, o episódio é truncado
e o agente recomeça em um novo tabuleiro. Isso evita episódios infinitamente longos
e garante que a IA seja exposta a diferentes estados iniciais durante o treinamento.
"""

DIRETORIO_MODELOS: str = "./saved_models"
"""Diretório onde os modelos treinados e checkpoints serão armazenados."""

DIRETORIO_LOGS: str = "./tb_logs"
"""Diretório para salvar arquivos de log do TensorBoard."""

NOME_MODELO: str = "tetris_ppo_model"
"""Nome do arquivo de saída para o modelo final treinado."""

# ==============================================================================
# CONFIGURAÇÕES DE MONITORAMENTO E AVALIAÇÃO
# ==============================================================================
SALVAR_CHECKPOINT_FREQUENCIA: int = 50_000
"""Intervalo de passos para salvar um checkpoint intermediário do modelo."""

AVALIAR_FREQUENCIA: int = 25_000
"""Intervalo de passos para rodar rodadas de avaliação no agente em treinamento."""

EPISODIOS_AVALIACAO: int = 5
"""Número de episódios de teste para tirar a média de pontuação na avaliação do modelo."""

RENDERIZAR_AVALIACAO: bool = False
"""Se verdadeiro, exibe visualmente o jogo durante a fase de avaliação do agente."""

RENDERIZAR_TREINAMENTO: bool = False
"""
Se verdadeiro, exibe visualmente o jogo em tempo real durante todo o treinamento.
Ideal para demonstrações educacionais. DEIXAR FALSE PARA MODO HEADLESS RÁPIDO!
→ Renderização ativa reduz velocidade em 5-10x!
"""
