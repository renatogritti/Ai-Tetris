"""
Ponto de Entrada Principal - AI Tetris Clássico.

Este arquivo inicia o jogo Tetris clássico em modo manual por padrão, permitindo
que o usuário jogue usando o teclado. Também contém exemplos documentados
de como carregar e rodar o Wrapper de Ambiente Gymnasium para treinamento de IA.
"""

import sys
from src.tetris_engine import TetrisEngine
from src.tetris_gui import TetrisGUI

def rodar_modo_manual() -> None:
    """Inicializa e executa o jogo de Tetris controlado pelo teclado."""
    print("Iniciando o Tetris Clássico no modo manual...")
    print("Controles:")
    print("  - Setas Esquerda/Direita: Mover peça")
    print("  - Seta Cima: Rotacionar peça")
    print("  - Seta Baixo: Acelerar descida (Soft Drop)")
    print("  - Espaço: Queda instantânea (Hard Drop)")
    print("  - P: Pausar o jogo")
    print("  - Enter: Iniciar/Reiniciar após Game Over")
    
    engine = TetrisEngine()
    gui = TetrisGUI(engine)
    gui.rodar_jogo_manual()

def exemplo_ambiente_rl() -> None:
    """
    Exemplo documentado de como utilizar o Wrapper Gymnasium (TetrisEnv)
    para treinar ou testar agentes de Reinforcement Learning.
    
    Para rodar este exemplo, você pode descomentar a chamada no main
    ou criar um arquivo separado para o treinamento da IA.
    """
    # Importação tardia do ambiente Gymnasium
    from src.tetris_env import TetrisEnv
    import time
    
    print("Executando simulação de teste com ações aleatórias no ambiente Gymnasium...")
    
    # Inicializa o ambiente com renderização humana para podermos assistir a IA
    env = TetrisEnv(render_mode="human")
    
    # Reseta o ambiente para o estado inicial
    obs, info = env.reset()
    
    terminado = False
    truncado = False
    passos = 0
    recompensa_total = 0.0
    
    while not (terminado or truncado):
        # Sorteia uma ação aleatória do espaço de ações discretas (0 a 5)
        acao = env.action_space.sample()
        
        # Executa a ação
        obs, recompensa, terminado, truncado, info = env.step(acao)
        
        recompensa_total += recompensa
        passos += 1
        
        # Pequena pausa para conseguirmos ver as ações na tela
        time.sleep(0.05)
        
        if passos % 100 == 0:
            print(f"Passo: {passos} | Recompensa acumulada: {recompensa_total:.2f} | Pontos: {info['score']}")
            
    print(f"\nFim da simulação!")
    print(f"Total de passos executados: {passos}")
    print(f"Pontuação máxima atingida: {info['score']}")
    print(f"Recompensa acumulada final: {recompensa_total:.2f}")
    
    # Fecha a tela do ambiente
    env.close()

if __name__ == "__main__":
    # Por padrão, inicia o jogo em modo manual controlado por teclado.
    # Se desejar testar a integração do Gymnasium, basta alterar para 'exemplo_ambiente_rl()'
    rodar_modo_manual()
    # exemplo_ambiente_rl()
