# Tetris AI (Placement-Based Reinforcement Learning)

Este projeto treina um agente de Tetris usando Reinforcement Learning com uma abordagem de posicionamento final. Em vez de controlar cada movimento no jogo, o agente escolhe uma combinação de rotação e coluna destino para a próxima peça.

## Estrutura principal

- `train_placement.py` — script principal de treinamento
- `evaluate_placement.py` — avaliação estatística do modelo treinado
- `test_placement.py` — visualização do modelo jogando
- `src/tetris_env_placement.py` — ambiente Gymnasium de posicionamento final
- `config_train.py` — hiperparâmetros e diretórios de logs
- `GUIA_OTIMIZACAO.md` — guia de otimização do projeto
- `requirements.txt` — dependências do projeto
- `saved_models/` — modelos salvos e checkpoints

## Configuração

Use o ambiente virtual do projeto e instale as dependências:

```powershell
cd "c:\Users\renat\Documents\Python\Games\Ai Tetris"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Se o ambiente virtual já existir, apenas ative-o:

```powershell
& .\.venv\Scripts\Activate.ps1
```

## Treinamento

Execute o script de treino para iniciar o treinamento do agente de posicionamento:

```powershell
python train_placement.py --train
```

O modelo final será salvo em:

- `saved_models/tetris_ppo_model_placement.zip`

## Avaliação

Para avaliar o modelo com vários episódios:

```powershell
python evaluate_placement.py --modelo ./saved_models/tetris_ppo_model_placement.zip --episodios 20
```

## Visualização

Para assistir o agente jogando em modo humano:

```powershell
python test_placement.py --modelo ./saved_models/tetris_ppo_model_placement.zip
```

## Guia de otimização

O arquivo `GUIA_OTIMIZACAO.md` contém o fluxo completo de treinamento, avaliação, ajustes de hiperparâmetros e recomendações de melhoria.

## Observações

- Os scripts antigos `train.py` e `evaluate_model.py` foram removidos da nova pipeline.
- `main.py` permanece para o jogo manual, mas não é parte da nova abordagem de RL.

## Problemas comuns

- Se o treinamento estiver muito lento, confirme que `RENDERIZAR_TREINAMENTO` está `False` em `config_train.py`.
- Se o score permanecer baixo, tente aumentar `TOTAL_TIMESTEPS`, `ent_coef` e ajustar as recompensas em `src/config.py`.

## Dependências principais

- `stable-baselines3`
- `gymnasium`
- `pygame`
- `numpy`
- `tensorboard`

---

Para o fluxo completo de otimização, abra `GUIA_OTIMIZACAO.md`.
