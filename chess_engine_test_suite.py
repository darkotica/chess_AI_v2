from alpha_beta_search_tt_id import get_next_move
import mlflow
from lite_model_wrapper import LiteModel
import csv
from chess import Board


def test_model_arasan(model_path, test_suite_epd):
    print("\n\n")
    print("Model: " + model_path)
    orig_model = mlflow.keras.load_model(model_path)
    orig_model.summary()
    lmodel = LiteModel.from_keras_model(orig_model)

    counter = 0

    with open(test_suite_epd, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        i = 1
        for row in spamreader:
            if "- -" in row[0]:
                fen_move = row[0].split("- -")
            else:
                fen_move = row[0].split("-")

            print(f"{i} : {fen_move}")
            i += 1

            fen = fen_move[0].strip()
            move = fen_move[1].strip().split(" ")[1:]

            board = Board(fen)
            res = get_next_move(board, lmodel, 5)

            res_move_san = board.san(res[1])
            print(f"Bm: {move}, found: {res_move_san}, correct: {res_move_san in move}")
            if res_move_san in move:
                counter += 1

    print("Counter: " + str(counter))


