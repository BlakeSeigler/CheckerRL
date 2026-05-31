import pygame
from checkers.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED
from checkers.game import Game


def self_play_game(network):
    # need to add the training loop setup

    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)

    training_data = []
    move_number = 0

    while run:

        mtcs = MTCS(game.copy(), network)
        first_move = True
        skipped = None

        clock.tick(FPS)
        
        if game.winner() != None:
                print(game.winner())
                run = False

        if first_move:
            mtcs.initialize_tree(game)
            first_move = False

        # Infer the next move with the MTCS algorithm -- these are back in 8x8 format
        inferenced_from, inferenced_to = mtcs.predict(game)

        # Add to the training data
        dist = mtcs.calculate_distribution(game, move_number)
        data = (format_data(game), dist, game.turn, 0)  # encoded state, distribution, turn, value
        training_data.append(data)

        # Make the move
        # potentally if and break if the move is a possible double jump -- simply make sure not to iterate the move number
        game.make_move(inferenced_from, inferenced_to, skipped=skipped) 

        game.update()   # does this just update the GUI?

        move_number += 1
    
    pygame.quit()
   
    for data in training_data:
        data[3] = 1 if game.winner() == data[2] else -1

    return training_data

main()