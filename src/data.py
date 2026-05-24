def format_data(game, color):
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

            if spot.color == color:
                data_vector.append(1)
            else:
                data_vector.append(-1)
    return data_vector