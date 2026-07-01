# ==============================================================================
#  AI Tetris - Hiperparâmetros do treinamento DQN
#
#  Author: Renato Gritti
#  Descrição: Centraliza os hiperparâmetros e diretórios de saída do treinamento.
# ==============================================================================
"""
Configuração do treinamento do agente DQN para Tetris.

Este módulo reúne todos os parâmetros necessários para ajustar a arquitetura da
rede, a taxa de aprendizado, o decaimento do epsilon e os diretórios de saída
do treinamento. A estratégia atual usa features heurísticas do tabuleiro para
avaliar cada placement possível.
"""

from typing import Any, Dict

# ==============================================================================
# HIPERPARÂMETROS DO AGENTE DQN
# ==============================================================================
HIPERPARAMETROS_DQN: Dict[str, Any] = {
    "num_features": 4,              # Número de features de entrada (lines, holes, bumpiness, height)
    "buffer_capacity": 20_000,      # Capacidade do Replay Buffer
    "batch_size": 512,              # Tamanho do minibatch para treino
    "gamma": 0.99,                  # Fator de desconto para recompensas futuras
    "learning_rate": 1e-3,          # Taxa de aprendizado do Adam optimizer
    "epsilon_start": 1.0,           # Epsilon inicial (100% exploração)
    "epsilon_end": 1e-3,            # Epsilon final (quase 0% exploração)
    "epsilon_decay_episodes": 2000, # Episódios para decair de start até end
}
"""
Hiperparâmetros do agente DQN.

Notas sobre ajustes:
    - Se o modelo não converge: aumente epsilon_decay_episodes (mais exploração)
    - Se o modelo fica instável: diminua learning_rate (ex: 5e-4)
    - Se o modelo é muito lento: diminua buffer_capacity e batch_size
    - Se o modelo "esquece": aumente buffer_capacity
"""

# ==============================================================================
# PARÂMETROS DE EXECUÇÃO
# ==============================================================================
TOTAL_EPISODIOS: int = 3000
"""
Número total de episódios (jogos completos) de treinamento.

Um episódio = um jogo do início até game over.
Convergência típica ocorre entre 1500-2500 episódios.
Para resultados robustos, use 3000-5000 episódios.
"""

# ==============================================================================
# DIRETÓRIOS DE SAÍDA
# ==============================================================================
DIRETORIO_MODELOS: str = "./saved_models"
"""Diretório onde os modelos treinados (.pt) e checkpoints serão armazenados."""

DIRETORIO_LOGS: str = "./tb_logs"
"""Diretório para arquivos de log (CSV de métricas de treinamento)."""

# ==============================================================================
# CONFIGURAÇÕES DE MONITORAMENTO
# ==============================================================================
FREQUENCIA_REPORT: int = 50
"""
Intervalo de episódios para exibir métricas de treinamento no console.

A cada FREQUENCIA_REPORT episódios, exibe:
    - Score médio, máximo e mínimo do bloco
    - Linhas eliminadas médias
    - Peças colocadas médias
    - Epsilon atual
    - Loss média do DQN
    - Tempo do bloco
"""

SALVAR_CHECKPOINT_FREQUENCIA: int = 500
"""Intervalo de episódios para salvar um checkpoint intermediário do modelo."""
