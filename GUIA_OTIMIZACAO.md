# 🔧 Guia de Otimização — Tetris AI (DQN)

## 📋 Visão Geral

Este guia explica como otimizar o treinamento do agente DQN para Tetris.
A abordagem usa **features heurísticas** (aggregate_height, holes, bumpiness, lines_cleared)
em vez da grid bruta, resultando em convergência rápida e estável.

## 🚀 Fluxo Recomendado

1. **Treinar** com parâmetros padrão (`python train_dqn.py`)
2. **Monitorar** as métricas no console durante o treinamento
3. **Avaliar** visualmente com `python demo_ai.py`
4. **Ajustar** hiperparâmetros se necessário
5. **Retreinar** com parâmetros otimizados

## 📊 Como Interpretar as Métricas

### Métricas do Console

```
    Ep │  Score Méd │  Score Máx │  Linhas Méd │  Peças Méd │  Epsilon │     Loss │  Tempo
──────────────────────────────────────────────────────────────────────────────────────────────
    50 │       40.0 │        120 │         0.2 │        8.5 │   0.9750 │   0.0234 │   2.1s
```

| Métrica | O que significa | Valor bom |
|---------|----------------|-----------|
| Score Méd | Pontuação NES média | Crescendo ao longo do treino |
| Score Máx | Melhor jogo do bloco | > 1000 após 1000 eps |
| Linhas Méd | Linhas eliminadas por jogo | > 10 após 2000 eps |
| Peças Méd | Peças colocadas por jogo | > 50 após 2000 eps |
| Epsilon | Taxa de exploração | Decai de 1.0 até 0.001 |
| Loss | Erro do DQN | Tendência de queda |

### Sinais de Problema

| Sintoma | Causa Provável | Solução |
|---------|---------------|---------|
| Score não sobe após 500 eps | Exploração insuficiente | Aumente `epsilon_decay_episodes` |
| Score sobe e depois cai | Overfitting | Aumente `buffer_capacity` |
| Loss oscila muito | Learning rate muito alto | Diminua `learning_rate` para 5e-4 |
| Peças média < 10 | Rede não aprendeu a evitar buracos | Treine mais episódios |
| Score máximo bom mas média baixa | Variância alta | Normal no início; espere mais eps |

## ⚙️ Ajuste de Hiperparâmetros

### Parâmetros Principais (em `config_train.py`)

#### Learning Rate
```python
"learning_rate": 1e-3,  # Padrão
```
- **Muito alto** (> 5e-3): Treinamento instável, loss oscila
- **Muito baixo** (< 1e-4): Convergência muito lenta
- **Recomendado**: 1e-3 para início, 5e-4 se instável

#### Epsilon Decay
```python
"epsilon_start": 1.0,
"epsilon_end": 1e-3,
"epsilon_decay_episodes": 2000,
```
- **Decay rápido** (500 eps): Aprende rápido mas pode ficar preso em ótimos locais
- **Decay lento** (5000 eps): Explora mais mas converge devagar
- **Recomendado**: 2000 para 3000 episódios totais

#### Batch Size
```python
"batch_size": 512,
```
- **Menor** (128): Mais ruído no gradiente, pode escapar de mínimos locais
- **Maior** (1024): Gradiente mais estável, mais lento por step
- **Recomendado**: 512 (bom equilíbrio)

#### Replay Buffer
```python
"buffer_capacity": 20_000,
```
- **Menor** (5000): Memória curta, esquece experiências antigas
- **Maior** (50000): Mais diversidade, mais RAM
- **Recomendado**: 20000 (cabe na RAM sem problemas)

#### Gamma (Fator de Desconto)
```python
"gamma": 0.99,
```
- **Menor** (0.9): Foco em recompensas imediatas
- **Maior** (0.999): Planeja mais a longo prazo
- **Recomendado**: 0.99 (bom equilíbrio para Tetris)

## 🔄 Estratégias de Treinamento

### Treinamento Básico
```bash
python train_dqn.py --episodes 3000
```
Suficiente para a maioria dos casos. O agente aprende a evitar buracos e limpar linhas.

### Treinamento Longo
```bash
python train_dqn.py --episodes 5000
```
Para resultados mais consistentes e scores mais altos.

### Continuar Treinamento
```bash
# Se o modelo está melhorando mas precisa de mais tempo:
python train_dqn.py --continue-from ./saved_models/tetris_dqn_best.pt --episodes 2000
```

### Treinamento com Debug Visual
```bash
# Para verificar se o agente está aprendendo visualmente:
python train_dqn.py --episodes 200 --render
```

## 📈 Plotando Resultados

O treinamento gera `tb_logs/training_log.csv` automaticamente. Para plotar:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tb_logs/training_log.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df["episodio"], df["score_med"], color="#00E5FF")
axes[0,0].set_title("Score Médio por Bloco")
axes[0,0].set_xlabel("Episódio")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df["episodio"], df["lines_med"], color="#76FF03")
axes[0,1].set_title("Linhas Eliminadas Médias")
axes[0,1].set_xlabel("Episódio")
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(df["episodio"], df["epsilon"], color="#FF9100")
axes[1,0].set_title("Epsilon Decay")
axes[1,0].set_xlabel("Episódio")
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(df["episodio"], df["loss_med"], color="#FF1744")
axes[1,1].set_title("Loss Média do DQN")
axes[1,1].set_xlabel("Episódio")
axes[1,1].grid(True, alpha=0.3)

plt.suptitle("Progresso do Treinamento — Tetris DQN", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_progress.png", dpi=150, bbox_inches="tight")
plt.show()
```

## 📁 Arquivos do Projeto

| Arquivo | Função |
|---------|--------|
| `train_dqn.py` | Script de treinamento DQN |
| `demo_ai.py` | Demonstração visual da IA |
| `main.py` | Jogo manual com teclado |
| `config_train.py` | Hiperparâmetros do DQN |
| `src/dqn_agent.py` | Agente DQN (PyTorch) |
| `src/tetris_env_features.py` | Ambiente com features heurísticas |
| `src/tetris_engine.py` | Motor lógico do Tetris |
| `src/tetris_gui.py` | Interface gráfica (Pygame) |
| `src/config.py` | Configurações do jogo |

## 💡 Dicas Finais

- **Sempre treine em modo headless** (sem `--render`) para velocidade máxima
- **Monitore o Score Médio** — é a métrica mais importante
- **O modelo salva automaticamente** o melhor por score médio
- **Use o CSV** para análise pós-treinamento detalhada
- **Não interrompa** o treinamento antes de 1500 episódios (epsilon ainda alto)
- **Aumente episódios** se o score ainda estiver subindo no final do treino
