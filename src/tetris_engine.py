# ==============================================================================
#  AI Tetris - Motor lógico do jogo
#
#  Author: Renato Gritti
#  Descrição: Implementa a lógica pura do Tetris, incluindo peças, colisões e pontuação.
# ==============================================================================
"""
Motor lógico do Tetris.

Este módulo implementa a lógica pura do jogo, incluindo a grade, o
gerenciamento de peças, colisões, pontuação clássica e detecção de fim de
jogo. Ele foi projetado para ser independente de qualquer biblioteca de
interface gráfica, permitindo execuções rápidas em simulações e treinamento de IA.
"""

import random
from typing import Dict, List, Optional, Tuple, TypeAlias
from src.config import (
    GRID_LARGURA,
    GRID_ALTURA,
    FORMATOS_PEÇAS,
    PONTOS_POR_LINHAS,
    LINHAS_POR_NIVEL
)

Grid: TypeAlias = List[List[str]]
PieceShape: TypeAlias = List[List[int]]


class TetrisEngine:
    """
    Classe que gerencia o estado e as regras de física/lógica do jogo Tetris.
    
    Atributos:
        grid (List[List[str]]): Matriz representando o tabuleiro (vazio = "", preenchido = tipo da peça).
        score (int): Pontuação atual do jogador.
        lines_cleared_total (int): Total de linhas limpas desde o início.
        level (int): Nível de dificuldade atual.
        game_over (bool): Flag indicando se o jogo terminou.
        current_piece_type (str): Tipo da peça atual ("I", "O", "T", etc.).
        current_piece_shape (List[List[int]]): Matriz da forma da peça atual.
        current_x (int): Posição X da peça atual no grid (canto superior esquerdo da matriz).
        current_y (int): Posição Y da peça atual no grid (canto superior esquerdo da matriz).
        next_piece_type (str): Tipo da próxima peça que irá entrar no jogo.
    """

    def __init__(self) -> None:
        """Inicializa um novo motor do Tetris resetado."""
        self.grid: Grid = [["" for _ in range(GRID_LARGURA)] for _ in range(GRID_ALTURA)]
        self.score: int = 0
        self.lines_cleared_total: int = 0
        self.level: int = 0
        self.game_over: bool = False
        
        # Estado das peças
        self.current_piece_type: str = ""
        self.current_piece_shape: PieceShape = []
        self.current_x: int = 0
        self.current_y: int = 0
        
        self.next_piece_type: str = ""
        
        # Inicializa o jogo
        self.reset()

    def reset(self) -> None:
        """Reinicia o estado do jogo para o padrão inicial."""
        self.grid: Grid = [["" for _ in range(GRID_LARGURA)] for _ in range(GRID_ALTURA)]
        self.score = 0
        self.lines_cleared_total = 0
        self.level = 0
        self.game_over = False
        
        # Sorteia as duas primeiras peças
        self.next_piece_type = self._get_random_piece_type()
        self.spawn_piece()

    def _get_random_piece_type(self) -> str:
        """
        Retorna um tipo de peça aleatório baseado na lista de formatos disponíveis.
        
        Retorna:
            str: Letra correspondente à peça (ex: 'I', 'O', 'T').
        """
        return random.choice(list(FORMATOS_PEÇAS.keys()))

    def spawn_piece(self) -> None:
        """
        Gera a nova peça atual a partir da próxima peça e sorteia uma nova
        próxima peça. Posiciona a peça no topo do tabuleiro.
        Se a nova peça colidir imediatamente ao nascer, o jogo é finalizado (Game Over).
        """
        self.current_piece_type = self.next_piece_type
        self.current_piece_shape = [list(row) for row in FORMATOS_PEÇAS[self.current_piece_type]]
        
        # Posição de spawn: centralizado no topo horizontalmente.
        # Ajusta Y para que a peça comece a descer gradativamente
        tamanho = len(self.current_piece_shape)
        self.current_x = (GRID_LARGURA - tamanho) // 2
        self.current_y = -tamanho  # Inicia oculto/no topo
        
        # Ajusta Y para peças que têm linhas vazias em sua matriz
        # (por exemplo, mover para baixo até que o primeiro bloco apareça)
        # O objetivo é não spawnar longe demais da grade visível
        while self.current_y < 0:
            # Verifica se a primeira linha (no índice relativo) contém blocos
            index_relativo = -self.current_y - 1
            if index_relativo < tamanho:
                if any(self.current_piece_shape[index_relativo]):
                    break
            self.current_y += 1

        # Sorteia o tipo da próxima peça
        self.next_piece_type = self._get_random_piece_type()

        # Verifica se colide logo no nascimento
        if self.check_collision(self.current_piece_shape, self.current_x, self.current_y):
            self.game_over = True

    def check_collision(self, shape: PieceShape, offset_x: int, offset_y: int) -> bool:
        """
        Verifica se a peça com a forma especificada colide com as bordas
        ou blocos já fixados no grid.
        
        Parâmetros:
            shape (List[List[int]]): Matriz representando a peça.
            offset_x (int): Coordenada X desejada para a peça.
            offset_y (int): Coordenada Y desejada para a peça.
            
        Retorna:
            bool: True se houver colisão, False caso contrário.
        """
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    grid_x = offset_x + c
                    grid_y = offset_y + r
                    
                    # Colisão lateral e inferior
                    if grid_x < 0 or grid_x >= GRID_LARGURA or grid_y >= GRID_ALTURA:
                        return True
                    
                    # Colisão com blocos já travados (apenas se estiver dentro da grade)
                    if grid_y >= 0:
                        if self.grid[grid_y][grid_x] != "":
                            return True
                    elif grid_y < -4:  # Proteção para não sair muito acima do grid
                        return True
        return False

    def move(self, dx: int, dy: int) -> bool:
        """
        Move a peça atual por um deslocamento horizontal (dx) e/ou vertical (dy).
        
        Parâmetros:
            dx (int): Deslocamento em X.
            dy (int): Deslocamento em Y.
            
        Retorna:
            bool: True se o movimento foi bem sucedido (sem colisão), False caso contrário.
        """
        if self.game_over:
            return False
            
        if not self.check_collision(self.current_piece_shape, self.current_x + dx, self.current_y + dy):
            self.current_x += dx
            self.current_y += dy
            return True
        return False

    def rotate(self) -> bool:
        """
        Rotaciona a peça atual no sentido horário.
        Inclui um sistema simples de "wall-kick" (empurrão de parede): se a rotação
        causar colisão, tenta deslocar a peça ligeiramente para os lados ou para
        cima para tentar acomodá-la.
        
        Retorna:
            bool: True se a rotação foi concluída com sucesso, False caso contrário.
        """
        if self.game_over:
            return False
            
        # O bloco 2x2 do "O" não precisa rotacionar
        if self.current_piece_type == "O":
            return True
            
        old_shape = self.current_piece_shape
        # Transpõe e reverte linhas para girar 90 graus no sentido horário
        new_shape = [list(row) for row in zip(*old_shape[::-1])]
        
        # Testes de Wall-Kick (Deslocamento X, Deslocamento Y)
        # Tenta: sem mover, esquerda 1, direita 1, cima 1, esquerda 2, direita 2
        kicks: List[Tuple[int, int]] = [(0, 0), (-1, 0), (1, 0), (0, -1), (-2, 0), (2, 0)]
        
        for dx, dy in kicks:
            if not self.check_collision(new_shape, self.current_x + dx, self.current_y + dy):
                self.current_piece_shape = new_shape
                self.current_x += dx
                self.current_y += dy
                return True
                
        return False

    def drop_soft(self) -> bool:
        """
        Acelera a queda da peça em 1 unidade para baixo (queda suave).
        Se a peça colidir, ela é travada na grade (locked), linhas são apagadas
        e uma nova peça é gerada.
        
        Retorna:
            bool: True se a peça desceu com sucesso, False se colidiu e travou.
        """
        if self.game_over:
            return False
            
        moved = self.move(0, 1)
        if not moved:
            self.lock_piece()
            return False
        return True

    def drop_hard(self) -> int:
        """
        Realiza a queda instantânea (Hard Drop) da peça até colidir com a base
        ou com outras peças. Em seguida, trava a peça e retorna a quantidade
        de linhas limpas.
        
        Retorna:
            int: Quantidade de linhas limpas por essa jogada.
        """
        if self.game_over:
            return 0
            
        lines_cleared = 0
        # Desce a peça até que colida
        while self.move(0, 1):
            pass
            
        lines_cleared = self.lock_piece()
        return lines_cleared

    def lock_piece(self) -> int:
        """
        Trava a peça atual no tabuleiro, fundindo seus blocos à matriz da grade.
        Se qualquer bloco for travado acima do limite visível (grid_y < 0), o jogo termina.
        Chama o método de limpeza de linhas e spawna a próxima peça.
        
        Retorna:
            int: Quantidade de linhas limpas.
        """
        for r, row in enumerate(self.current_piece_shape):
            for c, val in enumerate(row):
                if val:
                    grid_x = self.current_x + c
                    grid_y = self.current_y + r
                    
                    # Se tentar travar blocos acima do topo da grade visível, fim de jogo
                    if grid_y < 0:
                        self.game_over = True
                    
                    # Só trava dentro dos limites da grade visível
                    if 0 <= grid_y < GRID_ALTURA and 0 <= grid_x < GRID_LARGURA:
                        self.grid[grid_y][grid_x] = self.current_piece_type
                        
        if self.game_over:
            return 0
                        
        lines_cleared = self.clear_lines()
        self.spawn_piece()
        return lines_cleared

    def clear_lines(self) -> int:
        """
        Varre o tabuleiro de baixo para cima, removendo linhas totalmente completas,
        deslocando o grid superior e atualizando a pontuação e o nível.
        
        Retorna:
            int: Quantidade de linhas limpas.
        """
        lines_cleared = 0
        new_grid: Grid = []
        
        for row in self.grid:
            # Se houver qualquer bloco vazio, a linha não está completa
            if "" in row:
                new_grid.append(row)
            else:
                lines_cleared += 1
                
        # Adiciona linhas vazias no topo para as linhas removidas
        for _ in range(lines_cleared):
            new_grid.insert(0, ["" for _ in range(GRID_LARGURA)])
            
        self.grid = new_grid
        
        if lines_cleared > 0:
            self.lines_cleared_total += lines_cleared
            # Pontuação NES clássica multiplicada pelo nível correspondente
            base_pontos = PONTOS_POR_LINHAS.get(lines_cleared, 0)
            self.score += base_pontos * (self.level + 1)
            
            # Incremento de nível a cada 10 linhas limpas
            self.level = self.lines_cleared_total // LINHAS_POR_NIVEL
            
        return lines_cleared

    def get_column_heights(self) -> List[int]:
        """
        Calcula as alturas de cada coluna do tabuleiro.
        Útil para o cálculo de heurísticas da IA.
        
        Retorna:
            List[int]: Uma lista contendo as alturas de cada uma das colunas (0 a GRID_ALTURA).
        """
        heights = [0] * GRID_LARGURA
        for col in range(GRID_LARGURA):
            for row in range(GRID_ALTURA):
                if self.grid[row][col] != "":
                    heights[col] = GRID_ALTURA - row
                    break
        return heights

    def get_holes_count(self) -> int:
        """
        Calcula o número de buracos (células vazias que têm pelo menos um bloco preenchido acima delas).
        Essencial para heurísticas de RL.
        
        Retorna:
            int: Quantidade de buracos no tabuleiro.
        """
        holes = 0
        for col in range(GRID_LARGURA):
            block_found = False
            for row in range(GRID_ALTURA):
                if self.grid[row][col] != "":
                    block_found = True
                elif block_found and self.grid[row][col] == "":
                    holes += 1
        return holes

    def get_bumpiness(self) -> int:
        """
        Calcula a irregularidade (bumpiness) do tabuleiro, que é a soma absoluta
        das diferenças de altura entre colunas adjacentes.
        
        Retorna:
            int: Valor da irregularidade.
        """
        heights = self.get_column_heights()
        bumpiness = 0
        for col in range(GRID_LARGURA - 1):
            bumpiness += abs(heights[col] - heights[col + 1])
        return bumpiness

    def get_row_density_score(self) -> float:
        """
        Calcula uma pontuação de densidade das linhas do tabuleiro.

        Para cada linha, calcula a fração de células preenchidas (de 0.0 a 1.0)
        e aplica um expoente quadrático. Isso faz com que linhas quase completas
        valham muito mais do que linhas meio cheias, criando um gradiente de
        aprendizado suave para a IA progredir mesmo antes de limpar linhas.

        Exemplo:
            - Linha com 5/10 preenchidas → 0.5² = 0.25
            - Linha com 9/10 preenchidas → 0.9² = 0.81  (3x mais recompensa!)
            - Linha com 10/10 (completa) → 1.0² = 1.00

        Retorna:
            float: Pontuação total de densidade (soma de todas as linhas, de 0.0 a GRID_ALTURA).
        """
        score = 0.0
        for row in range(GRID_ALTURA):
            celulas_preenchidas = sum(1 for c in range(GRID_LARGURA) if self.grid[row][c] != "")
            densidade = celulas_preenchidas / GRID_LARGURA
            score += densidade ** 2  # Quadrático: recompensa linhas quase completas muito mais
        return score
