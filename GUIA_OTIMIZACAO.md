# 🎮 Guia de Otimização - Tetris Placement RL

## 📋 Visão Geral

Este projeto agora usa um fluxo de treinamento de Tetris baseado em ações de posicionamento final.
Em vez de controlar movimentos atômicos dentro do jogo, o agente escolhe diretamente:
- rotação da peça
- coluna de destino

Isso torna o problema mais apropriado para RL e reduz o ruído da ação.

## 🚀 Fluxo recomendado

1. Treinar o modelo de posicionamento
2. Avaliar o modelo com vários episódios
3. Testar visualmente a política treinada
4. Afinar hiperparâmetros se necessário

## 📦 Arquivos principais

- `train_placement.py` — treino do modelo de posicionamento
- `evaluate_placement.py` — avaliação estatística do modelo treinado
- `test_placement.py` — visualização do agente jogando
- `src/tetris_env_placement.py` — ambiente de posicionamento final
- `config_train.py` — hiperparâmetros e diretórios de log
- `GUIA_OTIMIZACAO.md` — guia de otimização detalhado
- `README.md` — documentação do repositório

## ⚙️ Setup básico

```powershell
# Ative o virtualenv se ainda não estiver ativo
& .\.venv\Scripts\Activate.ps1

# Instale dependências
django? # no, sorry
pip install -r requirements.txt
```

> Se você não tiver `.venv`, crie com:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ▶️ Como treinar

```powershell
python train_placement.py --train
```

O modelo final será salvo em:

- `saved_models/tetris_ppo_model_placement.zip`

## 📊 Como avaliar

```powershell
python evaluate_placement.py --modelo ./saved_models/tetris_ppo_model_placement.zip --episodios 20
```

Use `--render` para ver um episódio durante avaliação, mas mantenha `False` para validações mais rápidas.

## 👁️ Como visualizar

```powershell
python test_placement.py --modelo ./saved_models/tetris_ppo_model_placement.zip
```

## 📌 Notas importantes

- Os arquivos antigos `train.py` e `evaluate_model.py` foram removidos.
- O novo fluxo deve usar `train_placement.py`, `evaluate_placement.py`, e `test_placement.py`.
- `main.py` ainda existe para jogo manual, mas não é parte da nova pipeline de RL.

## 🔧 Ajuste de hiperparâmetros

Se o desempenho do modelo for baixo:

- aumente `ent_coef` em `config_train.py`
- aumente `TOTAL_TIMESTEPS` em `config_train.py`
- ajuste recompensas em `src/config.py`

## 📌 Observações sobre o ambiente

O ambiente de posicionamento expõe:
- `grid` de 20x10
- `current_piece`
- `next_piece`

A ação discreta tem tamanho `40` (
4 rotações × 10 colunas).

## 💡 Recomendações rápidas

- use treino em modo headless (`render_mode=None`)
- faça avaliação em 20+ episódios
- monitore logs do TensorBoard em `tb_logs`
- se precisar, aumente o tempo de treino para `1_000_000` passos
