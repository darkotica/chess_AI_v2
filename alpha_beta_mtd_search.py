from tensorflow.keras.models import model_from_json

from board_state_extractor import get_board_state
import math
import time
from collections import deque
import numpy as np
import tensorflow as tf
from chess import *
import mlflow
from lite_model_wrapper import LiteModel

nodes_total = 0
nodes_skipped = 0
total_depth = 4
position_dict = {}  # lowerbound, upperbound, move, depth, exact; LOWERBOUND, UPPERBOUND (0, 1)


def get_function_for_board_eval(board: Board, model):
    def funkc(move):
        copy_b = board.copy()
        copy_b.push(move)

        return model.predict_single(get_board_state(copy_b))[0]

    return funkc


def alpha_beta_with_memory(board: Board, alpha, beta, d, maximizing_player,
                           nn_model, moves=None, move_passed=None):
    global position_dict, nodes_total

    nodes_total += 1

    move = None

    if board.fen() in position_dict:
        if position_dict[board.fen()][3] >= d:
            pos = position_dict[board.fen()]

            if pos[0] is not None:
                if pos[0] >= beta:
                    return pos[0], pos[2]  # lowerbound
                alpha = max(alpha, pos[0])

            if pos[1] is not None:
                if pos[1] <= alpha:
                    return pos[1], pos[2]
                beta = min(beta, pos[1])

    else:
        position_dict[board.fen()] = [None, None, None, None]

    if d == 0:
        features = get_board_state(board)
        g = nn_model.predict_single(features)[0]
        move = move_passed

    elif maximizing_player:
        g = -math.inf
        a = alpha

        if moves is None:
            moves_deque = deque()
            for m in board.legal_moves:
                if board.is_capture(m) or board.is_castling(m) or board.gives_check(m):
                    moves_deque.appendleft(m)
                else:
                    moves_deque.append(m)
        else:
            moves_deque = moves

        for m in moves_deque:
            copy_board = board.copy()
            copy_board.push(m)
            res, _ = alpha_beta_with_memory(copy_board, a, beta, d-1, False,
                                               nn_model, move_passed=m)
            if res > g:
                move = m
                g = res

            a = max(a, g)

            if g >= beta:
                break
    else:
        g = math.inf
        b = beta

        if moves is None:
            moves_deque = deque()
            for m in board.legal_moves:
                if board.is_capture(m) or board.is_castling(m) or board.gives_check(m):
                    moves_deque.appendleft(m)
                else:
                    moves_deque.append(m)
        else:
            moves_deque = moves

        for m in moves_deque:
            copy_board = board.copy()
            copy_board.push(m)
            res, _ = alpha_beta_with_memory(copy_board, alpha, b, d - 1,
                                               True, nn_model, move_passed=m)
            if res < g:
                move = m
                g = res

            b = min(b, g)

            if g <= alpha:
                break

    if g <= alpha:
        node = position_dict[board.fen()]
        position_dict[board.fen()] = [node[0], g, move, d]
    if alpha < g < beta:
        position_dict[board.fen()] = [g, g, move, d]
    if g >= beta:
        node = position_dict[board.fen()]
        position_dict[board.fen()] = [g, node[1], move, d]

    return g, move


def mtdf(f, d, board, nn_model):
    g = f
    upperbound = math.inf
    lowerbound = -math.inf

    move = None

    moves = list(board.legal_moves)
    moves = sorted(moves, key=get_function_for_board_eval(board, nn_model),
                   reverse=board.turn)
    moves = deque(moves)

    while lowerbound < upperbound:
        if g == lowerbound:
            beta = g + 1
        else:
            beta = g
        g, move = alpha_beta_with_memory(board, beta-1, beta, d, board.turn,
                                         nn_model, moves=moves)
        if g < beta:
            upperbound = g
        else:
            lowerbound = g
    return g, move


def get_next_move(board, model, depth):
    global nodes_total, nodes_skipped, total_depth, position_dict
    board_copy = board.copy()
    print("\nStarted calculating")
    start = time.time()

    firstguess = model.predict_single(get_board_state(board))[0]
    moves = []

    for d in range(1, depth):
        firstguess, m = mtdf(firstguess, d, board, model)
        moves.append(m)
        print(firstguess, m)

    move = m

    prev_move = None
    new_move = moves[-1]
    move_orders = [new_move]
    while True:
        if not board_copy.is_legal(new_move):  # to znaci da je vraceni potez onaj na samom kraju
            break

        board_copy.push(new_move)
        new_move = position_dict[board_copy.fen()][2]  # move
        move_orders.append(new_move)
        if prev_move is None:
            prev_move = new_move
        else:
            if prev_move == new_move:
                break

            prev_move = new_move

    print(move_orders)

    end = time.time()
    print(f"Positions: {len(position_dict)}")
    print(f"Move heuristics: {firstguess}")
    print("Move (UCI format): " + move.uci())
    print(f"Solve time: {end - start}")
    print(f"Nodes total: {nodes_total}")

