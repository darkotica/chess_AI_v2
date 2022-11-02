from chess import Board
import math
import time
from collections import deque
from board_state_extractor import get_board_state
import sys
import numpy as np
import mlflow
from lite_model_wrapper import LiteModel
import tensorflow as tf

position_dict = {}


def get_function_for_board_eval(board: Board, model):
    global position_dict

    def funkc(move):
        copy_b = board.copy()
        copy_b.push(move)

        pred = model.predict_single(get_board_state(copy_b))[0]
        position_dict[board.fen()] = [pred, 0, copy_b.fen(), None]
        return pred

    return funkc


def alpha_beta_pruning(board: Board, depth, alpha, beta, maximizingPlayer, nn_model, moves=None):
    global position_dict

    recomm_move = None
    fen = board.fen()
    if fen in position_dict:
        if position_dict[fen][1] >= depth:
            return position_dict[fen][0], None, position_dict[fen][2]
        recomm_move = position_dict[fen][3]

    if depth == 0 or board.is_game_over():
        features = get_board_state(board)
        pred = nn_model.predict_single(features)[0]
        position_dict[fen] = [pred, 0, fen, None]  # pred and depth
        return pred, None, fen

    if moves is None:
        moves_deque = deque()
        for m in board.legal_moves:
            if board.is_capture(m) or len(board.attacks(m.to_square)) != 0:
                moves_deque.appendleft(m)
            else:
                moves_deque.append(m)
    else:
        moves_deque = moves

    if recomm_move:
        moves_deque.appendleft(recomm_move)

    result_fen = None

    if maximizingPlayer:
        value = -math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _, final_fen = alpha_beta_pruning(copy_board, depth - 1, alpha, beta, False, nn_model)

            if value < pruning_res:
                value = pruning_res
                move = m
                result_fen = final_fen

            alpha = max(alpha, value)
            if beta <= alpha:
                break

        position_dict[fen] = [value, depth, result_fen, move]
        return value, move, result_fen

    else:
        value = math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _, final_fen = alpha_beta_pruning(copy_board, depth - 1, alpha, beta, True, nn_model)

            if value > pruning_res:
                value = pruning_res
                move = m
                result_fen = final_fen

            beta = min(beta, value)
            if beta <= alpha:
                break

        position_dict[fen] = [value, depth, result_fen, move]
        return value, move, result_fen


def get_next_move(board, model_orig, model, depth):
    moves = list(board.legal_moves)
    if len(moves) != 0:
        moves = sorted(moves, key=get_function_for_board_eval(board, model), reverse=board.turn)
        moves = deque(moves)
    else:
        moves = None

    alpha_beta_pruning(board, depth - 3, -math.inf, math.inf, board.turn, model, moves)
    search_res = alpha_beta_pruning(board, depth - 1, -math.inf, math.inf, board.turn, model, moves)
    position_dict.clear()
    features = get_board_state(Board(search_res[2]))
    return model_orig(tf.reshape(features, [1, len(features)])), search_res[1]


if __name__ == '__main__':
    orig_model = mlflow.keras.load_model("runs:/15ff8fdc93cd44d888ca4069d4dc73e9/model")
    orig_model.summary()

    # features = get_board_state(Board("rnbqkbnr/ppp2ppp/8/3Bp3/8/6P1/PPPPPP1P/RNBQK1NR b KQkq - 0 3"))
    #
    # lmodel = LiteModel.from_keras_model(model)
    # start = time.time()
    # pred = lmodel.predict_single(features)
    # end = time.time()
    # print("Pred: " + str(pred))
    # print(f"Convert time: {end - start}")
    #
    # quit()

    # fen = sys.argv[1]
    # depth = int(sys.argv[2]) if len(sys.argv) >= 3 else None # if depth is passed as argument, else its default (4)

    # features = get_board_state(Board("rn1qkbnr/ppp2ppp/4b3/3pp3/8/5NP1/PPPPPP1P/RNBQK2R w KQkq - 2 5"))
    # features_reshaped = tf.reshape(features, [1, 768])
    # lmodel = LiteModel.from_keras_model(model)
    # start_1 = time.time()
    # pred_1 = lmodel.predict_single(features)
    # end_1 = time.time()
    # print("Pred: " + str(pred_1))
    # print(f"Convert time: {end_1 - start_1}")
    #
    # start = time.time()
    # pred = model(features_reshaped)
    # end = time.time()
    # print("Pred: " + str(pred))
    # print(f"Convert time: {end - start}")
    # quit()

    lmodel = LiteModel.from_keras_model(orig_model)
    # while True:
    #     print("\nInput fen: ")
    #     input_fen = input()
    #     features = get_board_state(Board(input_fen))
    #     pred_1 = model.predict_single(features)
    #     print("Pred: " + str(pred_1))
    # quit()

    while True:
        print("\nInput fen: ")
        input_fen = input()
        starting_board = Board(input_fen)
        print(get_next_move(starting_board, orig_model, lmodel, 5))
    #
    # # fen = sys.argv[1]
    # # depth = int(sys.argv[2]) if len(sys.argv) >= 3 else None # if depth is passed as argument, else its default (4)
    #
    # fen = "rnbqkbnr/pp4pp/2p2p2/3pN3/4p3/2N5/PPPPPPPP/R1BQKB1R w KQkq - 0 6"
    # depth = 5
    #
    # starting_board = Board(fen)
    # get_next_move(starting_board, model, depth)