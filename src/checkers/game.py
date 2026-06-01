import pygame
from .constants import RED, WHITE, BLUE, SQUARE_SIZE
from checkers.board import Board
from piece import Piece

class Game:
    def __init__(self, win):
        self._init()
        self.win = win
    
    def update(self):
        self.board.draw(self.win)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def _init(self):
        self.selected = None
        self.board = Board()
        self.turn = RED
        self.valid_moves = {}

    def winner(self):
        return self.board.winner()

    def reset(self):
        self._init()

    def select(self, row, col):
        """
        picks a piece that was passed in, if another piece was already picked then it switches (_move then checks and switches, it uses valid moves to know)
        the bottom part picks the first piece and then grabs valid moves from that piece
        """
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)
        
        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True
            
        return False

    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
        else:
            return False

        return True

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.win, BLUE, (col * SQUARE_SIZE + SQUARE_SIZE//2, row * SQUARE_SIZE + SQUARE_SIZE//2), 15)

    def change_turn(self):
        self.valid_moves = {}
        if self.turn == RED:
            self.turn = WHITE
        else:
            self.turn = RED

    def get_data(self):
        return self.board.board

    def get_valid_pieces(self, color):
        valid_pieces = []
        for row in self.board.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    valid_pieces.append(piece)
        return valid_pieces

    def make_move(self, from_row, from_col, to_row, to_col, skipped: list):
        self.selected = self.board.get_piece(from_row, from_col)
        piece = self.board.get_piece(to_row, to_col)
        if self.selected and piece == 0:
            self.board.move(self.selected, to_row, to_col)
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
            return True
        return False

    def get_valid_game_moves(self, turn):
        pieces = self.get_valid_pieces(turn)
        moves: list[tuple[  tuple[int], dict[tuple[int], list[Piece] ]]] = [] # a list of ( (from_row, from_col), {(to_row, to_col): [skipped pieces]} )
        for piece in pieces:
            piece_moves = self.board.get_valid_moves(piece)
            moves.append(((piece.row, piece.col), piece_moves))
        return moves