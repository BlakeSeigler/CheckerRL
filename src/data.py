def format_data(game, color): 
    """
    64 spot long vector
    """
    data_vector = []
    board_data = game.board.board
    for row in board_data:
        for spot in row:

            if spot == 0:
                data_vector.append(0)
            elif not spot.king:
                data_vector.append(1)
            elif spot.king:
                data_vector.append(2)

            if spot.color != color:  # If the opposite side, make it negative
                data_vector[-1] = data_vector[-1] * -1

    return data_vector