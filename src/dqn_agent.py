"""
Agente DQN Customizado para Tetris (dqn_agent.py).

Implementa um Deep Q-Network otimizado para o paradigma de avaliação de estados
do Tetris. Em vez de mapear (estado → ação), a rede avalia cada possível estado
resultante e retorna um Q-value. O agente escolhe a ação que leva ao estado
com maior Q-value.

Arquitetura da Rede:
    Input (4 features) → 64 (ReLU) → 64 (ReLU) → 1 (Q-value)

Componentes:
    - TetrisNet: Rede neural PyTorch
    - ReplayBuffer: Buffer circular de experiências
    - DQNAgent: Orquestra treinamento, seleção de ações e persistência
"""

import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TetrisNet(nn.Module):
    """
    Rede neural compacta para avaliação de estados do Tetris.

    Recebe um vetor de features heurísticas do tabuleiro e retorna
    um único valor (Q-value) representando a qualidade daquele estado.

    Arquitetura:
        4 → 64 (ReLU) → 64 (ReLU) → 1

    A rede é propositalmente pequena pois as features já são altamente
    informativas. Redes maiores tendem a overfitar sem ganho de performance.
    """

    def __init__(self, num_features: int = 4) -> None:
        """
        Inicializa a rede neural.

        Parâmetros:
            num_features (int): Número de features de entrada (default: 4).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propaga o input pela rede.

        Parâmetros:
            x (torch.Tensor): Tensor de features com shape (batch, num_features).

        Retorna:
            torch.Tensor: Q-values com shape (batch, 1).
        """
        return self.network(x)


class ReplayBuffer:
    """
    Buffer circular de experiências para Experience Replay.

    Armazena transições (state, reward, next_states, done) e permite
    amostragem aleatória uniforme para treino do DQN. A amostragem
    aleatória quebra a correlação temporal entre experiências consecutivas,
    estabilizando significativamente o treinamento.

    Atributos:
        buffer (deque): Fila circular com capacidade máxima fixa.
    """

    def __init__(self, capacity: int = 20_000) -> None:
        """
        Inicializa o buffer com capacidade máxima.

        Parâmetros:
            capacity (int): Número máximo de experiências armazenadas.
        """
        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Armazena uma transição no buffer.

        Parâmetros:
            state (np.ndarray): Vetor de features do estado atual (escolhido).
            reward (float): Recompensa obtida pela transição.
            next_state (np.ndarray): Vetor de features do próximo estado.
            done (bool): Se o episódio terminou após esta transição.
        """
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Amostra um minibatch aleatório do buffer.

        Parâmetros:
            batch_size (int): Tamanho do minibatch.

        Retorna:
            Tuple[np.ndarray, ...]: (states, rewards, next_states, dones)
                cada um como array NumPy empilhado.
        """
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Retorna o número de experiências armazenadas."""
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN completo para Tetris com features heurísticas.

    O agente usa o paradigma de "avaliação de estados": para cada peça,
    o ambiente gera todos os estados resultantes possíveis. O agente avalia
    cada um com a rede neural e escolhe o de maior Q-value.

    Componentes:
        - policy_net: Rede usada para seleção de ações (atualizada a cada step)
        - target_net: Rede estável para cálculo do target Q-value (sync periódico)
        - replay_buffer: Buffer de experiências para amostragem aleatória
        - optimizer: Adam optimizer com learning rate configurável

    Hiperparâmetros:
        - epsilon: Taxa de exploração (decai de epsilon_start até epsilon_end)
        - gamma: Fator de desconto para recompensas futuras
        - batch_size: Tamanho do minibatch para treino
    """

    def __init__(
        self,
        num_features: int = 4,
        buffer_capacity: int = 20_000,
        batch_size: int = 512,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 1e-3,
        epsilon_decay_episodes: int = 2000,
    ) -> None:
        """
        Inicializa o agente DQN com todos os hiperparâmetros.

        Parâmetros:
            num_features (int): Dimensão do vetor de features.
            buffer_capacity (int): Capacidade do replay buffer.
            batch_size (int): Tamanho do minibatch para treino.
            gamma (float): Fator de desconto (0-1).
            learning_rate (float): Taxa de aprendizado do Adam.
            epsilon_start (float): Epsilon inicial (exploração máxima).
            epsilon_end (float): Epsilon final (exploração mínima).
            epsilon_decay_episodes (int): Episódios para decaimento do epsilon.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features
        self.batch_size = batch_size
        self.gamma = gamma

        # Redes neural (policy e target)
        self.policy_net = TetrisNet(num_features).to(self.device)
        self.target_net = TetrisNet(num_features).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Otimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

    def choose_action(self, next_states: List[np.ndarray]) -> int:
        """
        Escolhe a melhor ação usando epsilon-greedy sobre os estados futuros.

        Com probabilidade epsilon, escolhe aleatoriamente (exploração).
        Caso contrário, avalia todos os estados futuros com a policy_net
        e escolhe aquele com maior Q-value (exploitation).

        Parâmetros:
            next_states (List[np.ndarray]): Lista de vetores de features,
                um para cada posição válida possível.

        Retorna:
            int: Índice da ação escolhida (posição na lista next_states).
        """
        if not next_states:
            return 0

        if random.random() < self.epsilon:
            return random.randint(0, len(next_states) - 1)

        # Avalia todos os estados futuros com a rede
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
            q_values = self.policy_net(states_tensor)
            return int(torch.argmax(q_values).item())

    def store(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Armazena uma transição no replay buffer.

        Parâmetros:
            state (np.ndarray): Features do estado atual.
            reward (float): Recompensa recebida.
            next_state (np.ndarray): Features do próximo estado.
            done (bool): Se o episódio terminou.
        """
        self.replay_buffer.store(state, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Executa um passo de treinamento usando experiências do replay buffer.

        Amostra um minibatch aleatório, calcula o target Q-value usando
        a target_net e atualiza a policy_net via gradiente descendente.

        Q_target = reward + gamma * max(target_net(next_state)) * (1 - done)
        Loss = MSE(policy_net(state), Q_target)

        Retorna:
            Optional[float]: Valor da loss, ou None se o buffer não tem
                experiências suficientes para um batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Converte para tensores PyTorch
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-values atuais da policy net
        q_values = self.policy_net(states_t)

        # Q-values alvo da target net (sem gradiente)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            q_targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # Loss e backpropagation
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self) -> None:
        """
        Sincroniza os pesos da target_net com a policy_net.

        Chamado periodicamente (tipicamente a cada episódio) para
        atualizar a rede alvo, estabilizando o treinamento.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, episode: int) -> None:
        """
        Decai o epsilon linearmente com base no episódio atual.

        O epsilon decai de epsilon_start até epsilon_end ao longo de
        epsilon_decay_episodes episódios, após isso permanece constante.

        Parâmetros:
            episode (int): Número do episódio atual.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end)
            * (episode / self.epsilon_decay_episodes),
        )

    def save(self, path: str) -> None:
        """
        Salva os pesos da policy_net em disco.

        Parâmetros:
            path (str): Caminho do arquivo .pt para salvar.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Carrega pesos salvos na policy_net e sincroniza a target_net.

        Parâmetros:
            path (str): Caminho do arquivo .pt para carregar.
        """
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()
