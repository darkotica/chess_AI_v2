import random
import tensorflow as tf
from chess import Board, pgn
import math
import time
from collections import deque
from board_state_extractor import get_board_state
import sys
import numpy as np
import mlflow
from lite_model_wrapper import LiteModel

nodes_total = 0
nodes_skipped = 0
total_depth = 4
position_dict = {}
tt_hits = 0
old_one=True


def get_function_for_board_eval(board: Board, model):
    global position_dict, old_one

    def funkc(move):
        copy_b = board.copy()
        copy_b.push(move)

        pred = model.predict_single(get_board_state(copy_b, old_one))[0]
        position_dict[board.fen()] = [pred, 0, None]  # pred and depth
        return pred

    return funkc


def alpha_beta_pruning(board: Board, depth, alpha, beta, maximizingPlayer, nn_model, moves=None):
    global nodes_total, nodes_skipped, position_dict, tt_hits, old_one
    nodes_total += 1

    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0, None  # stalemate, draw

        if winner is True:
            return math.inf, None  # white won
        else:
            return -math.inf, None  # black won

    recomm_move = None
    fen = board.fen()
    if fen in position_dict:
        if position_dict[fen][1] >= depth:
            tt_hits += 1
            return position_dict[fen][0], None

        recomm_move = position_dict[fen][2]

    if depth == 0:
        features = get_board_state(board, old_one)
        pred = nn_model.predict_single(features)[0]
        position_dict[fen] = [pred, 0, None]  # pred and depth
        return pred, None

    if moves is None:
        moves_deque = deque()
        for m in board.legal_moves:
            if board.is_capture(m) or board.is_castling(m) or board.gives_check(m):
                moves_deque.appendleft(m)
            else:
                moves_deque.append(m)
    else:
        moves_deque = moves

    if recomm_move:
        moves_deque.appendleft(recomm_move)

    if maximizingPlayer:
        value = -math.inf
        move = moves_deque[0]
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _ = alpha_beta_pruning(copy_board, depth - 1, alpha, beta, False, nn_model)

            if value < pruning_res:
                value = pruning_res
                move = m

            alpha = max(alpha, value)
            if beta <= alpha:
                nodes_skipped += board.legal_moves.count() - i
                break

        position_dict[fen] = [value, depth, move]
        return value, move

    else:
        value = math.inf
        move = moves_deque[0]
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _ = alpha_beta_pruning(copy_board, depth - 1, alpha, beta, True, nn_model)

            if value > pruning_res:
                value = pruning_res
                move = m

            beta = min(beta, value)
            if beta <= alpha:
                nodes_skipped += board.legal_moves.count() - i
                break

        position_dict[fen] = [value, depth, move]
        return value, move


def initiate_alpha_beta_pruning(board, model, depth=None):
    # before starting with alpha beta pruning, we sort list of next moves, from best to worst for current player
    # this way, we improve the first depth analysis of game tree which alpha-beta pruning searches
    # and reduce calculation time by doing so

    global nodes_total, nodes_skipped, total_depth
    moves = list(board.legal_moves)
    moves = sorted(moves, key=get_function_for_board_eval(board, model), reverse=board.turn)
    moves = deque(moves)

    search_depth = depth if depth is not None else total_depth

    pruning_res = alpha_beta_pruning(board, search_depth - 3, -math.inf, math.inf, board.turn, model, moves)
    # print(pruning_res)
    pruning_res = alpha_beta_pruning(board, search_depth - 1, -math.inf, math.inf, board.turn, model, moves)

    return pruning_res[0], pruning_res[1]


def get_next_move(board, model, depth):
    global nodes_total, nodes_skipped, total_depth, position_dict, tt_hits
    position_dict.clear()
    print("\nStarted calculating")
    start = time.time()
    alpha_beta_res = initiate_alpha_beta_pruning(board, model, depth)
    end = time.time()
    print(f"Positions: {len(position_dict)}")
    print(f"Hits: {tt_hits}")
    print(f"Move heuristics: {alpha_beta_res[0]}")
    print("Move (UCI format): " + alpha_beta_res[1].uci())
    print(f"Solve time: {end - start}")
    print(f"Nodes total: {nodes_total}, nodes skipped (at least): {nodes_skipped}")
    position_dict.clear()
    return end - start, alpha_beta_res[1]


def get_next_move_user_info(board, model, depth):
    global nodes_total, nodes_skipped, total_depth, position_dict, tt_hits
    position_dict.clear()
    print("\nStarted calculating")
    start = time.time()
    alpha_beta_res = initiate_alpha_beta_pruning(board, model, depth)
    end = time.time()
    print(f"Positions calculated: {len(position_dict)}")
    print(f"Repeating positions (transpositions): {tt_hits}")
    print("Best move found (UCI format): " + alpha_beta_res[1].uci())
    print(f"Move heuristics: {alpha_beta_res[0]}")
    print(f"Solve time: {end - start}")
    print(f"Positions in search tree (total): {nodes_total}, positions skipped (at least): {nodes_skipped}")
    position_dict.clear()
    return end - start, alpha_beta_res[1]


def duel(model1, model2):
    global old_one
    print("DUEL")
    orig_model_1 = mlflow.keras.load_model(model1)
    orig_model_2 = mlflow.keras.load_model(model2)

    lmodel1 = LiteModel.from_keras_model(orig_model_1)
    lmodel2 = LiteModel.from_keras_model(orig_model_2)

    board = Board()

    game = pgn.Game()
    game.setup(board)
    node = game

    move_num = 0

    while not board.is_game_over():
        move_num += 1
        copy_b = board.copy()
        if board.turn:  # white
            old_one = True
            print("Model 1")
            _, res = get_next_move(copy_b, lmodel1, 5)
            print()
        else:
            old_one = False
            print("Model 2")
            _, res = get_next_move(copy_b, lmodel2, 5)
            print()

        board.push(res)

        node = node.add_variation(res)
        if move_num % 50 == 0:
            print(game)

        print(board.fen())

        position_dict.clear()

    game.headers["Result"] = board.result()
    print(game)


if __name__ == '__main__':
    loaded_m = tf.keras.models.load_model("nn_model")
    loaded_m.summary()

    lmodel = LiteModel.from_keras_model(loaded_m)
    calc_time = 0
    total_moves = 0

    old_one = True

    while True:
        print("\nInput fen: ")
        input_fen = input()
        if input_fen == "x":
            break
        starting_board = Board(input_fen)
        time_to_move = get_next_move(starting_board, lmodel, 5)
        position_dict.clear()
        calc_time += time_to_move[0]
        total_moves += 1

    print("Average time: " + str(calc_time/total_moves))
    print("Total moves: " + str(total_moves))
