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

# def add finisher

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
    moves = sorted(moves, key=get_function_for_board_eval(board, model),
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

    #firstguess, move = mtdf(firstguess, 4, board, model)
    move = m

    prev_move = None
    new_move = moves[-1]
    move_orders = [new_move]
    while True:
        if not board_copy.is_legal(new_move):  # to znaci da je vraceni potez onaj na samom kraju
            # msm da se da ispraviti ali ne u 3 ujutro kad je ovo kucano :)
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


if __name__ == '__main__':
    #model = load_model("working_model/model_chess_ai.json",
    #                    "working_model/model_chess_ai.h5")

    # json_file = open('working_model/model_chess_ai.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # loaded_model.load_weights("working_model/model_chess_ai.h5")
    # model = loaded_model
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = mlflow.keras.load_model("runs:/15ff8fdc93cd44d888ca4069d4dc73e9/model")
    model.summary()

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

    #features = get_board_state(Board("rn1qkbnr/ppp2ppp/4b3/3pp3/8/5NP1/PPPPPP1P/RNBQK2R w KQkq - 2 5"))
    # features_reshaped = tf.reshape(features, [1, 768])
    #lmodel = LiteModel.from_keras_model(model)
    # start_1 = time.time()
    #pred_1 = lmodel.predict_single(features)
    # end_1 = time.time()
    #print("Pred: " + str(pred_1))
    # print(f"Convert time: {end_1 - start_1}")
    #
    # start = time.time()
    # pred = model(features_reshaped)
    # end = time.time()
    # print("Pred: " + str(pred))
    # print(f"Convert time: {end - start}")
    #quit()

    model = LiteModel.from_keras_model(model)
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
        get_next_move(starting_board, model, 5)
        position_dict.clear()

    # todo videti ono da ne bude dodela liste kod uslova sa alfa beta, vec vrednosti na indeksu liste, mozda bude brze
    # todo potencijalno dodati tamo gore uslov elif d == 0 i exact vrednost da se cuva u recniku
