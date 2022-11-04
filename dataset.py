import tensorflow as tf
import sqlite3
import chess
import numpy as np
import math
from board_state_extractor import get_board_state
import csv


"""DATASET PROCESS FUNCTIONS"""


def split_dataset_into_train_val_test_csv_files(dataset_path,
                                                train_data_path,
                                                test_data_path,
                                                val_data_path
                                                ):
    """

    :param dataset_path: path to sqlite dataset file
    :param train_data_path: path to .csv file in which train data is to be saved
    :param test_data_path: path to .csv file in which validation data is to be saved
    :param val_data_path: path to .csv file in which test data is to be saved

    Function which reads positions from sqlite dataset file and stores them in .csv files,
    by ratio 80-10-10 (train, validation and test).
    """
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("SELECT FEN, EVAL from evaluations")
    total = 0
    num_in_train, num_in_val, num_in_test = 0, 0, 0

    header = ["FEN", "EVAL"]

    with open(train_data_path, 'w',
              encoding='UTF8', newline='') as train:
        with open(val_data_path, 'w',
                  encoding='UTF8', newline='') as validation:
            with open(test_data_path, 'w',
                      encoding='UTF8', newline='') as test:
                writer_train = csv.writer(train)
                writer_val = csv.writer(validation)
                writer_test = csv.writer(test)

                writer_train.writerow(header)
                writer_val.writerow(header)
                writer_test.writerow(header)

                counter_val = 0
                counter_test = 0

                for res in cursor:
                    if not isinstance(res[1], float):
                        print("Not float: " + str(res))
                        continue

                    if counter_val >= 10:
                        writer_val.writerow([res[0], res[1]])
                        counter_val = 0

                        num_in_val += 1
                    elif counter_test >= 10:
                        writer_test.writerow([res[0], res[1]])
                        counter_test = 0

                        num_in_test += 1
                    else:
                        writer_train.writerow([res[0], res[1]])
                        counter_test += 1
                        counter_val += 1

                        num_in_train += 1

                    total += 1

    print(total)
    print(num_in_train)
    print(num_in_val)
    print(num_in_test)


def cut_value_and_normalize(batch):
    batch_evals = []
    for evaluation_val in batch:
        if evaluation_val > 5:
            evaluation_val = 4.3 + math.log10(evaluation_val)
        elif evaluation_val < -5:
            evaluation_val = -4.3 - math.log10(-evaluation_val)

        evaluation_val = 2 * (evaluation_val + 6.48) / (6.48 + 6.48) - 1
        batch_evals.append(evaluation_val)
    return np.asarray(batch_evals, dtype=np.float32)


def convert_fen_to_array(dataset_el):
    batch_of_features = []
    for fen in dataset_el:
        stringic_valjda = bytes.decode(fen)
        board_chess = chess.Board(stringic_valjda)
        features = get_board_state(board_chess)
        batch_of_features.append(features)
    return np.asarray(batch_of_features, dtype=np.float32)


def get_dataset_csv(train_dataset_csv_path, val_dataset_csv_path, bs=256):
    """
    Dataset creation
    """
    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_csv_path,
        batch_size=bs,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=1000000,
        sloppy=True,
        select_columns=['FEN', 'EVAL'],
        label_name='EVAL'
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x['FEN']],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.experimental.make_csv_dataset(
        val_dataset_csv_path,
        batch_size=bs,
        num_epochs=1,
        shuffle=False,
        sloppy=True,
        select_columns=['FEN', 'EVAL'],
        label_name='EVAL'
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x['FEN']],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def get_dataset_for_rl_csv(train_dataset_path, val_dataset_path, bs=1):
    """
        Dataset creation
        """
    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_path,
        batch_size=bs,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=30000,
        sloppy=True,
        select_columns=['FEN', 'EVAL'],
        label_name='EVAL'
    ).map(
        lambda x, y: (
            x,
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.experimental.make_csv_dataset(
        val_dataset_path,
        batch_size=256,
        num_epochs=1,
        shuffle=False,
        sloppy=True,
        select_columns=['FEN', 'EVAL'],
        label_name='EVAL'
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x['FEN']],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def get_test_dataset_csv(dataset_path, bs=256):
    test_dataset = tf.data.experimental.make_csv_dataset(
        dataset_path,
        batch_size=bs,
        num_epochs=1,
        shuffle=False,
        sloppy=True,
        select_columns=['FEN', 'EVAL'],
        label_name='EVAL'
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x['FEN']],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return test_dataset
