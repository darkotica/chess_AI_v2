import tensorflow as tf
import sqlite3
import chess
import numpy as np
import math
from board_state_extractor import get_board_state


def get_stats_from_dataset(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"):
    """
        Print tables in dataset
    """
    # con = sqlite3.connect(dataset_path)
    # cursor = con.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())

    """
        Get std and mean
    """
    # con = sqlite3.connect(dataset_path)
    # cursor = con.cursor()
    # cursor.execute("SELECT AVG((evaluations.eval - sub.a) * (evaluations.eval - sub.a)) as var from evaluations, "
    #                "(SELECT AVG(eval) AS a FROM evaluations) AS sub")
    # std = math.sqrt(cursor.fetchall()[0][0])
    #
    # cursor = con.cursor()
    # cursor.execute("SELECT AVG(eval) AS a FROM evaluations")
    # mean = math.sqrt(cursor.fetchall()[0][0])



    """
        Get min max values, as well as num of values in ranges
    """
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("SELECT FEN, EVAL from evaluations")
    max_v = 0
    min_v = 0
    counter = [0, 0, 0, 0]
    for res in cursor:
        if isinstance(res[1], float):
            if 0 <= res[1] <= 25:
                counter[0] += 1
            elif 25 < res[1] < 50:
                counter[1] += 1
            elif 50 <= res[1] < 75:
                counter[2] += 1
            elif 75 <= res[1]:
                counter[3] += 1

            if -25 <= res[1] < 0:
                counter[0] += 1
            elif -50 <= res[1] < -25:
                counter[1] += 1
            elif -75 <= res[1] < -50:
                counter[2] += 1
            elif res[1] < -75:
                counter[3] += 1

            if res[1] > max_v:
                max_v = res[1]
            elif res[1] < min_v:
                min_v = res[1]
        else:
            print(res[1])

    print(min_v)  # -152.65
    print(max_v)  # 152.65
    print(counter)


def get_dataset_partitions_tf(ds, batch_size, ds_size=38000000, train_split=0.8):
    train_size = int(int(train_split * ds_size) / batch_size)

    print(train_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    return train_ds, val_ds


def convert_fen_to_array(dataset_el):
    stringic_valjda = bytes.decode(dataset_el)
    board_chess = chess.Board(stringic_valjda)
    features = get_board_state(board_chess)
    return np.asarray(features, dtype=np.float32)


def limit_eval_value(eval_value):
    yic = eval_value

    if yic > 25:
        yic = math.log2(yic) * 10 - 21
    elif yic < -25:
        yic = 21 - math.log2(-yic) * 10

    # yic = 2 * (yic + 20) / (20 + 20) - 1  # normalize between -1 and 1
    return tf.constant(float(yic))


def get_dataset(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db", bs=256):
    """
    Dataset creation
    """
    dataset = tf.data.experimental.SqlDataset(
        "sqlite",
        dataset_path,
        "SELECT FEN, EVAL FROM evaluations",
        (tf.string, tf.double)
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x],
                Tout=[tf.float32]),
            tf.numpy_function(
                func=limit_eval_value,
                inp=[y],
                Tout=[tf.float32])
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(bs)

    dataset_train, dataset_val = get_dataset_partitions_tf(dataset, bs, train_split=0.9)

    return dataset_train, dataset_val


if __name__ == "__main__":
    get_dataset()
