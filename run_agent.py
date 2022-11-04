from alpha_beta_search_tt_id import get_next_move_user_info
from lite_model_wrapper import LiteModel
import tensorflow as tf
from chess import Board
import argparse


def run(model_path):
    loaded_m = tf.keras.models.load_model(model_path)
    loaded_m.summary()

    lmodel = LiteModel.from_keras_model(loaded_m)
    calc_time = 0
    total_moves = 0

    while True:
        print("\nInput fen: ")
        input_fen = input()
        if input_fen == "x":
            break
        starting_board = Board(input_fen)
        time_to_move = get_next_move_user_info(starting_board, lmodel, 5)
        calc_time += time_to_move[0]
        total_moves += 1

    print("Average time: " + str(calc_time / total_moves))
    print("Total moves: " + str(total_moves))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    args = parser.parse_args()

    run(args.model_path)
