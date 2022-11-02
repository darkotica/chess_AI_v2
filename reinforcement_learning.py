import tensorflow as tf
from loguru import logger
from dataset import get_dataset_for_rl_csv
from alpha_beta_search_tt_rl import get_next_move, get_function_for_board_eval
import random
import chess
import math
import mlflow
from mlflow import log_text, log_param
from lite_model_wrapper import LiteModel
from board_state_extractor import get_board_state


def td_alpha_full(model_orig, model_lite, alpha=1.0, lambda_in_sum=0.7,
                  iterations=1, num_start_pos=5, num_of_moves_search=2, depth=5):
    dataset_train, dataset_val = get_dataset_for_rl_csv()
    for s in range(iterations):
        print("STEP: " + str(s))
        batch_of_pos = dataset_train.take(num_start_pos)
        el_index = 0
        for el in batch_of_pos:
            logger.info("\nELEMENT: " + str(el_index))
            el_index += 1

            fen = str(el[0]["FEN"].numpy()[0], encoding="ascii")
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                print("No legal moves, skip")
                continue

            legal_moves = sorted(legal_moves, key=get_function_for_board_eval(board, model_lite), reverse=board.turn)
            move = random.choice(legal_moves[:5])  # we randomly choose 1 out of 5 top moves
            board.push(move)

            try:
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    predictions = []
                    for i in range(0, num_of_moves_search):
                        tape.watch(model_orig.trainable_weights)
                        print("-PREDICTION: " + str(i))
                        print(board.fen())
                        pos_eval, move = get_next_move(board, model_orig, model_lite, depth)
                        board.push(move)
                        print(pos_eval, move)

                        predictions.append(pos_eval)

                        if board.is_game_over():
                            break

                # there should be m predictions, and we need the gradient for each one
                gradients = [tape.gradient(mod_pred, model_orig.trainable_weights)
                             for mod_pred in predictions]

                for i in range(len(model_orig.trainable_weights)):
                    delta_weight = 0
                    for t in range(len(predictions)-1):
                        inner_sum = alpha*(predictions[t+1] - predictions[t])
                        grad_sum = 0
                        for k in range(t):
                            grad_sum += math.pow(lambda_in_sum, t-k) * gradients[k][i]

                        delta_weight += inner_sum*grad_sum

                    delta_weight = delta_weight/num_of_moves_search

                    # Select the layer
                    if delta_weight.shape[0] == 1:
                        model_orig.trainable_weights[i].assign_sub(delta_weight[0])
                    else:
                        model_orig.trainable_weights[i].assign_sub(delta_weight)

                del tape

                features = get_board_state(board)
                pred = model_orig(tf.reshape(features, [1, len(features)]))
                print("+Pred check: " + str(pred))

                if el_index != 0 and el_index % 32 == 0:  # model is saved every 32 positions
                    logger.info("Model saved")
                    res = model_orig.evaluate(dataset_val)
                    print("test loss, test acc:", res)
                    mlflow.keras.log_model(model_orig, "logged_model_" + str(s) + "_" + str(el_index))
            except Exception as e:
                print(e)
                print("Error with fen: " + fen)


if __name__ == "__main__":
    loaded_m = tf.keras.models.load_model("nn_model")
    loaded_m.summary()

    lmodel = LiteModel.from_keras_model(loaded_m)

    steps = 2
    num_of_moves = 6
    num_of_start_pos = 128
    search_depth = 5
    lam = 0.3

    a = 0.00001

    with mlflow.start_run():
        td_alpha_full(loaded_m, lmodel, iterations=steps, num_start_pos=num_of_start_pos, depth=search_depth,
                      num_of_moves_search=num_of_moves, alpha=a, lambda_in_sum=lam)

        mlflow.keras.log_model(loaded_m, "logged_model")
        log_text("Reinforcement learning. For " + str(steps) + " steps.",
                 "note.txt")
        log_param("steps", steps)
        log_param("alpha", a)
        log_param("num_of_moves", num_of_moves)
        log_param("num_of_start_pos", num_of_start_pos)
        log_param("search_depth", search_depth)

    logger.info("Finished training")
