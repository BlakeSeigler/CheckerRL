import pygame
from checkers.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED
from checkers.game import Game


def self_play_game(network):
    # need to add the training loop setup

    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)

    training_data = []

    while run:

        mtcs = MTCS(game.copy(), network)
        first_move = True

        clock.tick(FPS)
        
        if game.winner() != None:
                print(game.winner())
                run = False

        if first_move:
            mtcs.initialize_tree(game)
            first_move = False

        # Infer the next move with the MTCS algorithm
        inferenced_move = mtcs.predict(game)

        # Add to the training data
        dist = ...     # TODO: Add the distribution
        data = (format_data(game), dist, game.turn, 0)  # encoded state, distribution, turn, value
        training_data.append(data)

        # Make the move
        game.make_move(inferenced_move)

        game.update()   # does this just update the GUI?
    
    pygame.quit()
   
    for data in training_data:
        data[3] = 1 if game.winner() == data[2] else -1

    return training_data

main()