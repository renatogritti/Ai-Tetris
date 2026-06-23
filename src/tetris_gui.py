"""
Módulo de Interface Gráfica do Tetris (TetrisGUI).

Este módulo gerencia a renderização visual do jogo usando a biblioteca Pygame.
Possui recursos visuais avançados como paletas premium de cores (neon/dark mode),
peça fantasma (que indica onde a peça atual irá pousar), efeitos de brilho/sombra
e painéis para informações adicionais (próxima peça, pontuação, nível e linhas).
Também controla o loop principal de jogo para usuários humanos (controle manual).
"""

import sys
import pygame
from typing import List, Tuple, Dict
from src.config import (
    GRID_LARGURA,
    GRID_ALTURA,
    TAMANHO_BLOCO,
    LARGURA_JANELA,
    ALTURA_JANELA,
    FPS,
    COR_FUNDO,
    COR_GRID,
    COR_PAREDE,
    COR_PAINEL,
    COR_TEXTO,
    COR_TEXTO_MUTED,
    CORES_PEÇAS,
    FORMATOS_PEÇAS,
    VELOCIDADE_INICIAL,
    DIFICULDADE_DECRESCIMO_VELOCIDADE,
    VELOCIDADE_MINIMA
)
from src.tetris_engine import TetrisEngine

class TetrisGUI:
    """
    Classe responsável por renderizar o Tetris e gerenciar a entrada manual do usuário.
    
    Atributos:
        engine (TetrisEngine): Instância da lógica do jogo.
        screen (pygame.Surface): Superfície principal de exibição do Pygame.
        clock (pygame.time.Clock): Controlador de taxa de quadros (FPS) do jogo.
        board_x (int): Posição X de início da renderização da grade de jogo.
        board_y (int): Posição Y de início da renderização da grade de jogo.
    """

    def __init__(self, engine: TetrisEngine) -> None:
        """
        Inicializa a interface gráfica do Pygame.
        
        Parâmetros:
            engine (TetrisEngine): O motor de lógica que o GUI irá monitorar e renderizar.
        """
        pygame.init()
        pygame.display.set_caption("AI Tetris Clássico - Pygame")
        
        self.engine = engine
        self.screen = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
        self.clock = pygame.time.Clock()
        
        # Coordenadas de posicionamento da grade de jogo (centralizada verticalmente na esquerda)
        self.board_x = 40
        self.board_y = 40
        
        # Inicialização de Fontes
        self.fontes: Dict[str, pygame.font.Font] = {}
        self._inicializar_fontes()
        
        # Estado de Controle do Loop
        self.pausado = False
        self.jogo_iniciado = False

    def _inicializar_fontes(self) -> None:
        """Tenta carregar fontes modernas do sistema, com fallback para a fonte padrão."""
        fontes_preferidas = ["Segoe UI", "Outfit", "Arial", "Helvetica", "Trebuchet MS"]
        
        # Tentativa de inicializar fontes de diferentes tamanhos
        tamanhos = {"titulo": 36, "subtitulo": 24, "corpo": 18, "pequeno": 14}
        
        for nome, tamanho in tamanhos.items():
            fonte_carregada = None
            for fp in fontes_preferidas:
                try:
                    fonte_carregada = pygame.font.SysFont(fp, tamanho, bold=(nome in ["titulo", "subtitulo"]))
                    # Se carregou com sucesso, não usa fallback
                    if fonte_carregada:
                        break
                except Exception:
                    continue
            
            if fonte_carregada is None:
                fonte_carregada = pygame.font.Font(None, tamanho)
            
            self.fontes[nome] = fonte_carregada

    def _obter_posicao_fantasma_y(self) -> int:
        """
        Calcula a posição vertical onde a peça atual pousaria.
        Utilizado para desenhar a peça fantasma.
        
        Retorna:
            int: Posição Y (linha da grade) final de pouso da peça.
        """
        ghost_y = self.engine.current_y
        while not self.engine.check_collision(self.engine.current_piece_shape, self.engine.current_x, ghost_y + 1):
            ghost_y += 1
        return ghost_y

    def desenhar_bloco(self, surface: pygame.Surface, x: int, y: int, cor: Tuple[int, int, int], is_ghost: bool = False) -> None:
        """
        Desenha um único bloco do Tetris (quadradinho) com acabamento estético.
        Blocos normais possuem um efeito de chanfro 3D.
        Blocos fantasma são desenhados como contornos semi-transparentes.
        
        Parâmetros:
            surface (pygame.Surface): Superfície de desenho.
            x (int): Coordenada X na tela em pixels.
            y (int): Coordenada Y na tela em pixels.
            cor (Tuple[int, int, int]): Cor básica RGB do bloco.
            is_ghost (bool): Se verdadeiro, desenha em estilo fantasma/contorno.
        """
        rect = pygame.Rect(x, y, TAMANHO_BLOCO, TAMANHO_BLOCO)
        
        if is_ghost:
            # Desenha apenas o contorno com cor levemente transparente (simulada)
            cor_suave = tuple(int(c * 0.4) for c in cor)
            pygame.draw.rect(surface, cor_suave, rect, 2, border_radius=3)
        else:
            # Desenha o bloco preenchido principal
            pygame.draw.rect(surface, cor, rect, border_radius=4)
            
            # Efeito 3D: Destaque brilhante (superior e esquerdo)
            brilho = tuple(min(255, int(c * 1.4)) for c in cor)
            pygame.draw.line(surface, brilho, (x, y), (x + TAMANHO_BLOCO - 1, y), 2)
            pygame.draw.line(surface, brilho, (x, y), (x, y + TAMANHO_BLOCO - 1), 2)
            
            # Efeito 3D: Sombra escura (inferior e direito)
            sombra = tuple(int(c * 0.6) for c in cor)
            pygame.draw.line(surface, sombra, (x + 1, y + TAMANHO_BLOCO - 1), (x + TAMANHO_BLOCO - 1, y + TAMANHO_BLOCO - 1), 2)
            pygame.draw.line(surface, sombra, (x + TAMANHO_BLOCO - 1, y + 1), (x + TAMANHO_BLOCO - 1, y + TAMANHO_BLOCO - 1), 2)

    def desenhar_grade_jogo(self, surface: pygame.Surface) -> None:
        """
        Desenha o tabuleiro do jogo, incluindo o fundo escuro do grid,
        as linhas divisórias internas e a borda iluminada do tabuleiro.
        
        Parâmetros:
            surface (pygame.Surface): Superfície de desenho.
        """
        # Dimensões físicas do tabuleiro
        tabuleiro_largura = GRID_LARGURA * TAMANHO_BLOCO
        tabuleiro_altura = GRID_ALTURA * TAMANHO_BLOCO
        
        # Desenha o fundo escuro do grid
        rect_fundo = pygame.Rect(self.board_x, self.board_y, tabuleiro_largura, tabuleiro_altura)
        pygame.draw.rect(surface, COR_GRID, rect_fundo)
        
        # Desenha as linhas internas da grade para auxiliar o jogador
        for c in range(GRID_LARGURA + 1):
            x = self.board_x + c * TAMANHO_BLOCO
            pygame.draw.line(surface, COR_FUNDO, (x, self.board_y), (x, self.board_y + tabuleiro_altura), 1)
            
        for r in range(GRID_ALTURA + 1):
            y = self.board_y + r * TAMANHO_BLOCO
            pygame.draw.line(surface, COR_FUNDO, (self.board_x, y), (self.board_x + tabuleiro_largura, y), 1)
            
        # Desenha a moldura (borda externa do tabuleiro) com visual neon
        pygame.draw.rect(surface, COR_PAREDE, rect_fundo, 3, border_radius=4)

    def desenhar_painel_lateral(self, surface: pygame.Surface) -> None:
        """
        Desenha os painéis informativos na lateral direita:
        1. Painel "PRÓXIMA PEÇA"
        2. Painel "ESTATÍSTICAS" (Pontuação, Nível, Linhas)
        3. Painel "CONTROLES"
        
        Parâmetros:
            surface (pygame.Surface): Superfície de desenho.
        """
        # Posição inicial em X para os painéis
        painel_x = self.board_x + GRID_LARGURA * TAMANHO_BLOCO + 30
        largura_painel = LARGURA_JANELA - painel_x - 30
        
        # ----------------------------------------------------------------------
        # 1. Painel Próxima Peça
        # ----------------------------------------------------------------------
        rect_prox = pygame.Rect(painel_x, self.board_y, largura_painel, 180)
        pygame.draw.rect(surface, COR_PAINEL, rect_prox, border_radius=8)
        pygame.draw.rect(surface, COR_PAREDE, rect_prox, 2, border_radius=8)
        
        # Título do Painel Próxima Peça
        txt_prox_lbl = self.fontes["subtitulo"].render("SEGUINTE", True, COR_TEXTO)
        surface.blit(txt_prox_lbl, (painel_x + 15, self.board_y + 15))
        
        # Desenhar visualização da próxima peça centralizada no painel
        proxima = self.engine.next_piece_type
        if proxima:
            shape_prox = FORMATOS_PEÇAS[proxima]
            cor_prox = CORES_PEÇAS[proxima]
            
            # Calcula offsets para centralizar no painel de 180x180
            # A área disponível é de Y = board_y + 50 a board_y + 160
            tam_matriz = len(shape_prox)
            px_largura = tam_matriz * TAMANHO_BLOCO
            
            offset_desenho_x = painel_x + (largura_painel - px_largura) // 2
            offset_desenho_y = self.board_y + 60 + (100 - tam_matriz * TAMANHO_BLOCO) // 2
            
            for r, row in enumerate(shape_prox):
                for c, val in enumerate(row):
                    if val:
                        bx = offset_desenho_x + c * TAMANHO_BLOCO
                        by = offset_desenho_y + r * TAMANHO_BLOCO
                        self.desenhar_bloco(surface, bx, by, cor_prox)

        # ----------------------------------------------------------------------
        # 2. Painel Estatísticas
        # ----------------------------------------------------------------------
        rect_stats = pygame.Rect(painel_x, self.board_y + 200, largura_painel, 260)
        pygame.draw.rect(surface, COR_PAINEL, rect_stats, border_radius=8)
        pygame.draw.rect(surface, COR_PAREDE, rect_stats, 2, border_radius=8)
        
        # Informações textuais (Rótulo e Valor)
        estatisticas = [
            ("PONTUAÇÃO", str(self.engine.score)),
            ("NÍVEL DO JOGO", str(self.engine.level)),
            ("LINHAS LIMPAS", str(self.engine.lines_cleared_total))
        ]
        
        y_atual = self.board_y + 220
        for rotulo, valor in estatisticas:
            # Rótulo em cinza
            txt_rotulo = self.fontes["pequeno"].render(rotulo, True, COR_TEXTO_MUTED)
            surface.blit(txt_rotulo, (painel_x + 20, y_atual))
            
            # Valor em branco destacado
            txt_valor = self.fontes["titulo"].render(valor, True, COR_TEXTO)
            surface.blit(txt_valor, (painel_x + 20, y_atual + 18))
            
            y_atual += 75

        # ----------------------------------------------------------------------
        # 3. Painel Controles / Atalhos
        # ----------------------------------------------------------------------
        rect_ctrl = pygame.Rect(painel_x, self.board_y + 480, largura_painel, 220)
        pygame.draw.rect(surface, COR_PAINEL, rect_ctrl, border_radius=8)
        pygame.draw.rect(surface, COR_PAREDE, rect_ctrl, 2, border_radius=8)
        
        txt_ctrl_lbl = self.fontes["subtitulo"].render("CONTROLES", True, COR_TEXTO)
        surface.blit(txt_ctrl_lbl, (painel_x + 15, self.board_y + 495))
        
        controles = [
            ("← / →", "Mover Peça"),
            ("↑", "Rotacionar"),
            ("↓", "Descer Lento"),
            ("Espaço", "Queda Rápida"),
            ("P", "Pausar Jogo")
        ]
        
        y_atual = self.board_y + 535
        for tecla, acao in controles:
            # Tecla destacada
            txt_tecla = self.fontes["corpo"].render(tecla, True, COR_TEXTO)
            surface.blit(txt_tecla, (painel_x + 20, y_atual))
            
            # Ação associada
            txt_acao = self.fontes["pequeno"].render(acao, True, COR_TEXTO_MUTED)
            # Alinha a ação à direita do painel
            surface.blit(txt_acao, (painel_x + largura_painel - txt_acao.get_width() - 20, y_atual + 2))
            
            y_atual += 30

    def desenhar_elementos_jogo(self, surface: pygame.Surface) -> None:
        """
        Renderiza os blocos já fixados na grade e as peças atuais (jogável e fantasma).
        
        Parâmetros:
            surface (pygame.Surface): Superfície de desenho.
        """
        # 1. Desenhar blocos travados no grid
        for r in range(GRID_ALTURA):
            for c in range(GRID_LARGURA):
                tipo = self.engine.grid[r][c]
                if tipo != "":
                    bx = self.board_x + c * TAMANHO_BLOCO
                    by = self.board_y + r * TAMANHO_BLOCO
                    self.desenhar_bloco(surface, bx, by, CORES_PEÇAS[tipo])
                    
        # Se o jogo acabou ou não iniciou, não desenha peças ativas
        if self.engine.game_over:
            return
            
        # 2. Desenhar a Peça Fantasma (Ghost Piece)
        ghost_y = self._obter_posicao_fantasma_y()
        cor_ativa = CORES_PEÇAS[self.engine.current_piece_type]
        
        for r, row in enumerate(self.engine.current_piece_shape):
            for c, val in enumerate(row):
                if val:
                    gx = self.engine.current_x + c
                    gy = ghost_y + r
                    # Apenas desenha se estiver dentro dos limites do tabuleiro visível
                    if 0 <= gy < GRID_ALTURA:
                        bx = self.board_x + gx * TAMANHO_BLOCO
                        by = self.board_y + gy * TAMANHO_BLOCO
                        self.desenhar_bloco(surface, bx, by, cor_ativa, is_ghost=True)
                        
        # 3. Desenhar a Peça Atual (Normal)
        for r, row in enumerate(self.engine.current_piece_shape):
            for c, val in enumerate(row):
                if val:
                    gx = self.engine.current_x + c
                    gy = self.engine.current_y + r
                    # Apenas desenha se estiver dentro dos limites do tabuleiro visível
                    if 0 <= gy < GRID_ALTURA:
                        bx = self.board_x + gx * TAMANHO_BLOCO
                        by = self.board_y + gy * TAMANHO_BLOCO
                        self.desenhar_bloco(surface, bx, by, cor_ativa)

    def renderizar(self) -> None:
        """
        Limpa a tela e gerencia toda a árvore de renderização do jogo,
        incluindo a grade, os painéis laterais, as peças e as sobreposições de tela
        (Game Over, Pausa, Menu Inicial).
        """
        self.screen.fill(COR_FUNDO)
        
        # Desenha a área do jogo e os painéis
        self.desenhar_grade_jogo(self.screen)
        self.desenhar_elementos_jogo(self.screen)
        self.desenhar_painel_lateral(self.screen)
        
        # Sobreposições de Estado
        if not self.jogo_iniciado:
            self._renderizar_overlay_inicio()
        elif self.engine.game_over:
            self._renderizar_overlay_gameover()
        elif self.pausado:
            self._renderizar_overlay_pausa()
            
        pygame.display.flip()

    def _renderizar_overlay_inicio(self) -> None:
        """Desenha uma tela inicial estilosa com sobreposição escura e instruções."""
        self._desenhar_overlay_base()
        
        txt_titulo = self.fontes["titulo"].render("TETRIS CLASSIC", True, (0, 240, 240))
        txt_sub = self.fontes["subtitulo"].render("Pressione ENTER para Jogar", True, COR_TEXTO)
        txt_info = self.fontes["corpo"].render("Use as SETAS para controlar as peças", True, COR_TEXTO_MUTED)
        
        # Blit centralizado
        self.screen.blit(txt_titulo, (LARGURA_JANELA // 2 - txt_titulo.get_width() // 2, ALTURA_JANELA // 2 - 60))
        self.screen.blit(txt_sub, (LARGURA_JANELA // 2 - txt_sub.get_width() // 2, ALTURA_JANELA // 2))
        self.screen.blit(txt_info, (LARGURA_JANELA // 2 - txt_info.get_width() // 2, ALTURA_JANELA // 2 + 50))

    def _renderizar_overlay_pausa(self) -> None:
        """Desenha a tela de pausa."""
        self._desenhar_overlay_base()
        
        txt_pausa = self.fontes["titulo"].render("JOGO PAUSADO", True, (240, 240, 0))
        txt_sub = self.fontes["subtitulo"].render("Pressione P para Continuar", True, COR_TEXTO)
        
        self.screen.blit(txt_pausa, (LARGURA_JANELA // 2 - txt_pausa.get_width() // 2, ALTURA_JANELA // 2 - 30))
        self.screen.blit(txt_sub, (LARGURA_JANELA // 2 - txt_sub.get_width() // 2, ALTURA_JANELA // 2 + 20))

    def _renderizar_overlay_gameover(self) -> None:
        """Desenha a tela de fim de jogo (Game Over)."""
        self._desenhar_overlay_base()
        
        txt_go = self.fontes["titulo"].render("FIM DE JOGO", True, (240, 0, 0))
        txt_score = self.fontes["subtitulo"].render(f"Pontuação: {self.engine.score}", True, COR_TEXTO)
        txt_sub = self.fontes["corpo"].render("Pressione ENTER para Reiniciar", True, COR_TEXTO_MUTED)
        
        self.screen.blit(txt_go, (LARGURA_JANELA // 2 - txt_go.get_width() // 2, ALTURA_JANELA // 2 - 50))
        self.screen.blit(txt_score, (LARGURA_JANELA // 2 - txt_score.get_width() // 2, ALTURA_JANELA // 2 + 5))
        self.screen.blit(txt_sub, (LARGURA_JANELA // 2 - txt_sub.get_width() // 2, ALTURA_JANELA // 2 + 45))

    def _desenhar_overlay_base(self) -> None:
        """Desenha um retângulo preto semi-transparente sobre toda a tela para escurecer."""
        overlay = pygame.Surface((LARGURA_JANELA, ALTURA_JANELA), pygame.SRCALPHA)
        overlay.fill((10, 10, 15, 200))  # RGB + Alpha (Opacidade de ~80%)
        self.screen.blit(overlay, (0, 0))

    def rodar_jogo_manual(self) -> None:
        """
        Executa o loop de jogo manual clássico.
        Gerencia relógio do Pygame, entrada de teclado, velocidade de queda
        progressiva e renderizações frequentes.
        """
        # Variáveis de controle de tempo de queda
        tempo_acumulado = 0.0
        
        while True:
            # Limita a taxa de quadros ao FPS configurado
            dt = self.clock.tick(FPS) / 1000.0  # Converte delta time para segundos
            
            # 1. Processamento de Entradas (Eventos do Teclado)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    # Alterna estado de tela inicial ou game over
                    if not self.jogo_iniciado or self.engine.game_over:
                        if event.key == pygame.K_RETURN:
                            self.engine.reset()
                            self.jogo_iniciado = True
                            tempo_acumulado = 0.0
                        continue
                        
                    # Atalho para pausa
                    if event.key == pygame.K_p:
                        self.pausado = not self.pausado
                        continue
                        
                    # Se o jogo estiver pausado, bloqueia movimentação de peças
                    if self.pausado:
                        continue
                        
                    # Movimentação do Jogador
                    if event.key == pygame.K_LEFT:
                        self.engine.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.engine.move(1, 0)
                    elif event.key == pygame.K_UP:
                        self.engine.rotate()
                    elif event.key == pygame.K_SPACE:
                        self.engine.drop_hard()
                        tempo_acumulado = 0.0  # Reseta o timer de queda livre ao travar

            # 2. Lógica de Tempo e Gravidade de Queda
            if self.jogo_iniciado and not self.pausado and not self.engine.game_over:
                # Calcula a velocidade de acordo com o nível atual
                velocidade_queda = max(
                    VELOCIDADE_MINIMA,
                    VELOCIDADE_INICIAL - (self.engine.level * DIFICULDADE_DECRESCIMO_VELOCIDADE)
                )
                
                # Trata movimentação de descida contínua (Seta para baixo pressionada)
                teclas_pressionadas = pygame.key.get_pressed()
                multiplicador_queda = 10.0 if teclas_pressionadas[pygame.K_DOWN] else 1.0
                
                tempo_acumulado += dt * multiplicador_queda
                
                if tempo_acumulado >= velocidade_queda:
                    # Desce a peça suavemente. Se travar, reseta o tempo acumulado
                    if not self.engine.drop_soft():
                        tempo_acumulado = 0.0
                    else:
                        # Se moveu com sucesso sob gravidade acelerada pelo jogador,
                        # desconta apenas o custo de um passo para manter suavidade
                        tempo_acumulado -= velocidade_queda

            # 3. Renderização visual do estado atualizado
            self.renderizar()
