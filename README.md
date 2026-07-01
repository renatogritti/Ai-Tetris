<!--
===============================================================================
 AI Tetris - Documentação principal do projeto

 Author: Renato Gritti
 Descrição: Visão geral do projeto, fluxo de uso e estrutura de arquivos.
===============================================================================
-->

# 🎮 AI Tetris — Deep Q-Network com Features Heurísticas

Este projeto implementa um agente de Inteligência Artificial que aprende a jogar Tetris usando Deep Reinforcement Learning com DQN e features heurísticas do tabuleiro. Em vez de processar pixels ou a grade bruta, o agente avalia métricas estratégicas do estado para decidir o melhor placement de cada peça.

## 🧠 Arquitetura da IA

### Abordagem: DQN com Avaliação de Estados

Para cada peça, o sistema:
1. **Enumera** todas as posições finais válidas (rotação × coluna)
2. **Simula** cada placement em uma cópia do tabuleiro
3. **Extrai** 4 features heurísticas do board resultante
4. **Avalia** cada estado com a rede neural (Q-value)
5. **Escolhe** a posição com maior Q-value

### Features Heurísticas

| Feature | Descrição | Impacto |
|---------|-----------|---------|
| `lines_cleared` | Linhas eliminadas pelo placement | Objetivo principal |
| `holes` | Células vazias abaixo de blocos | Principal causa de game over |
| `bumpiness` | Irregularidade da superfície | Superfícies planas = mais Tetris |
| `aggregate_height` | Soma das alturas das colunas | Manter board baixo = sobrevivência |

### Rede Neural

```
Input (4 features) → Dense(64, ReLU) → Dense(64, ReLU) → Output(1, Q-value)
```

A rede é propositalmente compacta — as features já são altamente informativas.

## 📁 Estrutura do Projeto

```
├── train_dqn.py                  # 🏋️ Script principal de treinamento
├── demo_ai.py                    # 🎬 Demonstração visual da IA jogando
├── main.py                       # 🎮 Jogo manual (teclado)
├── config_train.py               # ⚙️ Hiperparâmetros do DQN
├── requirements.txt              # 📦 Dependências
├── README.md                     # 📖 Esta documentação
├── GUIA_OTIMIZACAO.md            # 🔧 Guia de otimização detalhado
│
├── src/
│   ├── dqn_agent.py              # 🤖 Agente DQN (PyTorch)
│   ├── tetris_env_features.py    # 🌍 Ambiente com features heurísticas
│   ├── tetris_engine.py          # ⚡ Motor lógico do Tetris
│   ├── tetris_gui.py             # 🖼️ Interface gráfica (Pygame)
│   └── config.py                 # 🎨 Configurações do jogo
│
├── saved_models/                 # 💾 Modelos treinados
│   ├── tetris_dqn_best.pt        #     Melhor modelo
│   ├── tetris_dqn_final.pt       #     Modelo final
│   └── checkpoints/              #     Checkpoints intermediários
│
└── tb_logs/                      # 📊 Logs de treinamento
    └── training_log.csv          #     Métricas por episódio
```

## 🚀 Início Rápido

### 1. Instalar Dependências

```bash
cd /home/gritti/Documentos/python/Ai/Ai-Tetris
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o Modelo

```bash
# Treinamento padrão (3000 episódios, ~15-30 min)
python train_dqn.py

# Treinamento customizado
python train_dqn.py --episodes 5000

# Continuar treinamento de modelo existente
python train_dqn.py --continue-from ./saved_models/tetris_dqn_best.pt

# Treinar com visualização (lento, para debug)
python train_dqn.py --render
```

**Saída do treinamento:**
```
==========================================================================================
  TREINAMENTO DQN - TETRIS AI (Features Heurísticas)
==========================================================================================
  Episódios: 3000
  Batch size: 512
  Learning rate: 0.001
  Epsilon: 1.0 → 0.001 (2000 eps)
==========================================================================================

    Ep │  Score Méd │  Score Máx │  Linhas Méd │  Peças Méd │  Epsilon │     Loss │  Tempo
──────────────────────────────────────────────────────────────────────────────────────────────
    50 │       40.0 │        120 │         0.2 │        8.5 │   0.9750 │   0.0234 │   2.1s
   100 │       80.0 │        300 │         0.8 │       12.3 │   0.9500 │   0.0187 │   2.3s
   ...
  2500 │     2400.0 │       8500 │        18.5 │       85.2 │   0.0010 │   0.0021 │   4.5s
  3000 │     3200.0 │      12000 │        24.1 │      102.7 │   0.0010 │   0.0018 │   4.8s
```

### 3. Demonstrar o Modelo

```bash
# Demonstração visual (velocidade padrão)
python demo_ai.py

# Mais rápido
python demo_ai.py --velocidade 15

# Modelo específico
python demo_ai.py --modelo ./saved_models/tetris_dqn_best.pt

# Parar após 5 jogos
python demo_ai.py --episodios 5
```

### 4. Jogar Manualmente

```bash
python main.py
```

**Controles:**
- ← / → : Mover peça
- ↑ : Rotacionar
- ↓ : Descida lenta
- Espaço : Queda rápida
- P : Pausar

## ⚙️ Configuração de Hiperparâmetros

Edite `config_train.py` para ajustar:

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `batch_size` | 512 | Tamanho do minibatch |
| `gamma` | 0.99 | Fator de desconto |
| `learning_rate` | 1e-3 | Taxa de aprendizado |
| `epsilon_start` | 1.0 | Exploração inicial |
| `epsilon_end` | 1e-3 | Exploração final |
| `epsilon_decay_episodes` | 2000 | Episódios de decay |
| `buffer_capacity` | 20000 | Tamanho do replay buffer |
| `TOTAL_EPISODIOS` | 3000 | Episódios de treinamento |

## 📊 Monitoramento do Treinamento

O treinamento gera um arquivo CSV (`tb_logs/training_log.csv`) com todas as métricas. Você pode plotar os resultados com:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tb_logs/training_log.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df["episodio"], df["score_med"])
axes[0,0].set_title("Score Médio")

axes[0,1].plot(df["episodio"], df["lines_med"])
axes[0,1].set_title("Linhas Médias")

axes[1,0].plot(df["episodio"], df["epsilon"])
axes[1,0].set_title("Epsilon Decay")

axes[1,1].plot(df["episodio"], df["loss_med"])
axes[1,1].set_title("Loss Média")

plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()
```

## 🔬 Detalhes Técnicos

### Por que DQN e não PPO?

| Aspecto | PPO (antigo) | DQN (novo) |
|---------|-------------|------------|
| Tipo | On-policy | Off-policy |
| Replay de experiências | ❌ Descarta após uso | ✅ Reutiliza milhares de vezes |
| Eficiência de dados | Baixa | Alta |
| Espaço de ações | Todas as ações possíveis | Avalia cada estado futuro |
| Convergência em Tetris | ~1M+ timesteps, instável | ~2000 episódios, estável |

### Por que Features e não Grid Bruta?

| Entrada | Dimensão | Convergência |
|---------|----------|-------------|
| Grid bruta 20×10 | 200 valores | Muito lenta ou nunca |
| Grid + peça (2 canais) | 400 valores | Lenta |
| 4 features heurísticas | 4 valores | Rápida (~2000 eps) |

As features condensam a informação estratégica relevante, eliminando ruído.

## 📦 Dependências Principais

- **PyTorch** — Rede neural e treinamento do DQN
- **Pygame** — Interface gráfica do jogo
- **NumPy** — Computação numérica
- **Gymnasium** — Interface padrão de ambientes (legado)
- **Matplotlib** — Plotagem de métricas de treinamento

## 🧹 Arquivos Legados

Os módulos antigos relacionados ao fluxo PPO/Gymnasium foram removidos do fluxo principal, pois a implementação atual prioriza a abordagem DQN com features heurísticas e o motor de jogo direto.

---

Para otimização avançada, consulte o [Guia de Otimização](GUIA_OTIMIZACAO.md).
