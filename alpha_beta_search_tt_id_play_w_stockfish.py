import chess
from chess import Board, pgn
import math
import time
from collections import deque
from board_state_extractor import get_board_state
import sys
import numpy as np
import mlflow
from lite_model_wrapper import LiteModel
from stockfish import Stockfish

nodes_total = 0
nodes_skipped = 0
total_depth = 4
position_dict = {}
tt_hits = 0
old_one = True


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

    # if board.is_game_over():
    #     winner = board.outcome().winner
    #     if winner is None:
    #         return 0, None  # stalemate, draw
    #
    #     if winner is True:
    #         return 1, None  # white won
    #     else:
    #         return -1, None  # black won
    outcome = board.outcome(claim_draw=True)
    if outcome:
        winner = outcome.winner
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
    print(pruning_res)
    pruning_res = alpha_beta_pruning(board, search_depth - 1, -math.inf, math.inf, board.turn, model, moves)

    return pruning_res[0], pruning_res[1]


def get_next_move(board, model, depth):
    global nodes_total, nodes_skipped, total_depth, position_dict, tt_hits
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
    return end - start, alpha_beta_res[1]


def duel_stockfish(model1, model_white=True, stockfish_rating=1700, start_pos=None, game_id=0):
    global old_one
    print(f"\nDUEL {game_id}")
    print(f"Model is white: {model_white}")
    orig_model_1 = mlflow.keras.load_model(model1)

    lmodel1 = LiteModel.from_keras_model(orig_model_1)

    board = Board() if not start_pos else Board(start_pos)

    game = pgn.Game()
    game.setup(board)
    node = game

    move_num = 0

    model_plays = model_white

    stockfish = Stockfish(
        path="/home/igor/Documents/Chess_bot/stockfish/stockfish_15_linux_x64/stockfish_15_src/src/stockfish")
    if start_pos:
        stockfish.set_fen_position(start_pos)
    stockfish.set_elo_rating(stockfish_rating)

    while not board.outcome(claim_draw=True):
        move_num += 1
        copy_b = board.copy()
        if model_plays:  # white
            model_plays = not model_plays
            print("Model")
            _, res = get_next_move(copy_b, lmodel1, 5)
            print()
        else:
            model_plays = not model_plays
            print("Stockfish")
            move = stockfish.get_best_move_time(5000)
            res = chess.Move.from_uci(move)
            print(move)
            print()


        board.push(res)
        stockfish.make_moves_from_current_position([res.uci()])

        node = node.add_variation(res)
        if move_num % 25 == 0:
            print(game)

        print(board.fen())

        position_dict.clear()

    game.headers["Result"] = board.result()
    print(game)
    return board.outcome(claim_draw=True).winner


def initiate_dueling(model_path):
    mod_is_white = True
    draws = 0
    model_won = 0
    stockfish_won = 0
    for i in range(10):
        try:
            res = duel_stockfish(model_path, stockfish_rating=1600,
                                 model_white=mod_is_white, game_id=i)
        except Exception as e:
            print(e)
            print("Error with match, starting new game")
            continue

        if res is None:
            draws += 1
        elif res and mod_is_white:
            model_won += 1
        elif not res and not mod_is_white:
            model_won += 1
        else:
            stockfish_won += 1

        mod_is_white = not mod_is_white
        print(f"Score: mod wins {model_won}, stockfish wins {stockfish_won}, draws {draws}")


if __name__ == '__main__':
    # import time
    # import chess.polyglot
    # board = Board()
    # start = time.time()
    # #board.fen()
    # prr = chess.polyglot.zobrist_hash(board)
    # end = time.time()
    # print(f"{(end-start)}")
    # old_one = False
    # print("rl, 4000, 1300, 1000")
    # initiate_dueling("runs:/8526f722c5624b979db3d8b15bbc5811/logged_model")
    # print("no rl, 4000, 1300, 1000")
    # initiate_dueling("runs:/ebd176887d9d4a8f9d461651069602fd/logged_model")

    old_one = True
    print("rl, 4k, 2k, 2k")
    initiate_dueling("runs:/207731ac18d645cb9bc4a4564b717028/logged_model")
    print("no 4k, 2k, 2k")
    initiate_dueling("runs:/15ff8fdc93cd44d888ca4069d4dc73e9/logged_model")

    # mod_is_white = True
    # draws = 0
    # model_won = 0
    # stockfish_won = 0
    # for i in range(10):
    #     try:
    #         res = duel_stockfish("runs:/15ff8fdc93cd44d888ca4069d4dc73e9/model", stockfish_rating=1700,
    #                              model_white=mod_is_white, game_id=i)
    #     except Exception as e:
    #         print(e)
    #         print("Error with match, starting new game")
    #         continue
    #
    #     if res is None:
    #         draws += 1
    #     elif res and mod_is_white:
    #         model_won += 1
    #     elif not res and not mod_is_white:
    #         model_won += 1
    #     else:
    #         stockfish_won += 1
    #
    #     mod_is_white = not mod_is_white
    #     print(f"Score: mod wins {model_won}, stockfish wins {stockfish_won}, draws {draws}")
