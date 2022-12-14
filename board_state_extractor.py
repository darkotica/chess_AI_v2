import chess
from chess import *


def get_board_to_list(board: Board, old_one=True):
    ret_list = []
    for i in range(0, 64):
        ret_list.extend(get_piece_at(i, board))

    board_status = [
        1 if board.turn else -1,
        1 if board.has_kingside_castling_rights(WHITE) else 0,
        1 if board.has_queenside_castling_rights(WHITE) else 0,
        -1 if board.has_kingside_castling_rights(BLACK) else 0,
        -1 if board.has_queenside_castling_rights(BLACK) else 0
    ]

    ret_list.extend(board_status)

    return ret_list


def get_piece_at(index, board: Board):
    listica = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    piece = board.piece_at(index)
    if not piece:
        return listica

    set_i = 1
    offset = 0
    if piece.color == BLACK:
        set_i = -1
        offset = 6

    listica[piece.piece_type - 1 + offset] = set_i
    return listica


def get_board_state(chess_board: Board, old_one=True):
    # positions
    state = get_board_to_list(chess_board, old_one)

    return state
