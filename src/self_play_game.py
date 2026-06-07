import pygame
from checkers.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED
from checkers.game import Game
from mtcs import MTCS
from data import format_data
import copy



def self_play_game(network):
    # need to add the training loop setup

    run = True
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    game = Game(win)

    training_data = []
    move_number = 0

    while run:           # per move in a game

        mtcs = MTCS(game, network)

        # Infer the next move with the MTCS algorithm -- these are back in 8x8 format
        (from_row, from_col), (to_row, to_col), skipped = mtcs.predict()

        # Add to the training data
        dist = mtcs.calculate_distribution(game, move_number)
        data = [format_data(game, game.turn), dist, game.turn, 0]  # encoded state, distribution, turn, value
        training_data.append(data)

        # Make the move
        game.make_move(from_row, from_col, to_row, to_col, skipped) 


        move_number += 1
        if move_number > 300:
            break

        if game.winner() != None:
            print(game.winner())
            run = False

    pygame.quit()
   
    for data in training_data: # TODO this does not account for timeout draws rn
        data[3] = 1 if game.winner() == data[2] else -1

    return training_data
