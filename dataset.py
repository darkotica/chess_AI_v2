import tensorflow as tf
import sqlite3
import chess
import numpy as np
import math
from board_state_extractor import get_board_state
import csv


"""DATASET PROCESS FUNCTIONS"""


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


def standardize_eval(eval_value):
    yic = eval_value

    mean = 0.5439183057819399
    std = 9.861819123496254
    yic = (yic - mean)/std
    return tf.constant(float(yic))


"""~~~~~~~~~~~~~~~~~~~~~~~~~"""


def write_to_dataset_part(val_writer, test_writer, train_writer, rest_writer,
                          val_bound, test_bound, rest_bound, counter, res, has_rest=True):
    if counter[0] >= val_bound:
        val_writer.writerow([res[0], res[1]])
        counter[0] = 0

    elif counter[1] >= test_bound:
        test_writer.writerow([res[0], res[1]])
        counter[1] = 0

    elif has_rest and counter[2] >= rest_bound:
        rest_writer.writerow([res[0], res[1]])
        counter[2] = 0

    else:
        counter[0] = counter[0] + 1
        counter[1] = counter[1] + 1
        counter[2] = counter[2] + 1
        train_writer.writerow([res[0], res[1]])


def split_dataset_into_train_val_test_csv_files_balanced(
        dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"
):
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("SELECT FEN, EVAL from evaluations")
    total = 0

    header = ["FEN", "EVAL"]
    with open('/home/igor/Documents/Chess_bot/Datasets/rest_data_balanced.csv', 'w',
              encoding='UTF8', newline='') as rest:
        with open('/home/igor/Documents/Chess_bot/Datasets/train_data_balanced.csv', 'w',
                  encoding='UTF8', newline='') as train:
            with open('/home/igor/Documents/Chess_bot/Datasets/val_data_balanced.csv', 'w',
                      encoding='UTF8', newline='') as validation:
                with open('/home/igor/Documents/Chess_bot/Datasets/test_data_balanced.csv', 'w',
                          encoding='UTF8', newline='') as test:
                    writer_train = csv.writer(train)
                    writer_val = csv.writer(validation)
                    writer_test = csv.writer(test)
                    writer_rest = csv.writer(rest)

                    writer_train.writerow(header)
                    writer_val.writerow(header)
                    writer_test.writerow(header)
                    writer_rest.writerow(header)

                    counter_0_1_plus = [0, 0, 0]  # val test rest : 12, 12, 6
                    counter_1_5_plus = [0, 0, 0]  # 12, 12, 6
                    counter_5_10_plus = [0, 0, 0]  # 11, 11, 10
                    counter_10_25_plus = [0, 0, 0]  # 11, 11, 13

                    counter_0_1_neg = [0, 0, 0]
                    counter_1_5_neg = [0, 0, 0]
                    counter_5_10_neg = [0, 0, 0]
                    counter_10_25_neg = [0, 0, 0]

                    counter_others = [0, 0, 0]

                    for res in cursor:
                        if not isinstance(res[1], float):
                            print("Not float: " + str(res))
                            continue

                        if total % 100000 == 0:
                            print(total)

                        if 0 < res[1] <= 1:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=12, test_bound=12, rest_bound=6, counter=counter_0_1_plus)
                        elif 1 < res[1] <= 5:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=12, test_bound=12, rest_bound=6, counter=counter_1_5_plus)
                        elif 5 < res[1] <= 10:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=11, test_bound=11, rest_bound=10, counter=counter_5_10_plus)
                        elif 10 < res[1] <= 25:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=11, test_bound=11, rest_bound=13,
                                                  counter=counter_10_25_plus)

                        elif -1 <= res[1] < 0:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=10, test_bound=10, rest_bound=-1,
                                                  counter=counter_0_1_neg, has_rest=False)
                        elif -5 <= res[1] < -1:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=10, test_bound=10, rest_bound=-1,
                                                  counter=counter_1_5_neg, has_rest=False)
                        elif -10 <= res[1] < -5:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=10, test_bound=10, rest_bound=-1,
                                                  counter=counter_5_10_neg, has_rest=False)
                        elif -25 <= res[1] < -10:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=10, test_bound=10, rest_bound=-1,
                                                  counter=counter_10_25_neg, has_rest=False)
                        else:
                            write_to_dataset_part(val_writer=writer_val, test_writer=writer_test,
                                                  train_writer=writer_train, rest_writer=writer_rest, res=res,
                                                  val_bound=10, test_bound=10, rest_bound=-1,
                                                  counter=counter_others, has_rest=False)

                        total += 1

                    print(total)


def split_dataset_into_train_val_test_csv_files(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"):
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("SELECT FEN, EVAL from evaluations")
    total = 0
    num_in_train, num_in_val, num_in_test = 0, 0, 0

    header = ["FEN", "EVAL"]

    with open('/home/igor/Documents/Chess_bot/Datasets/train_data.csv', 'w',
              encoding='UTF8', newline='') as train:
        with open('/home/igor/Documents/Chess_bot/Datasets/val_data.csv', 'w',
                  encoding='UTF8', newline='') as validation:
            with open('/home/igor/Documents/Chess_bot/Datasets/test_data.csv', 'w',
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


def create_tables(cursor):
    # Creating table
    table_train = """ CREATE TABLE IF NOT EXISTS train_table (
                    id integer PRIMARY KEY,
                    FEN text NOT NULL,
                    EVAL real NOT NULL
                ); """
    table_val = """ CREATE TABLE IF NOT EXISTS val_table (
                        id integer PRIMARY KEY,
                        FEN text NOT NULL,
                        EVAL real NOT NULL
                    ); """
    table_test = """ CREATE TABLE IF NOT EXISTS test_table (
                        id integer PRIMARY KEY,
                        FEN text NOT NULL,
                        EVAL real NOT NULL
                    ); """

    cursor.execute(table_train)
    cursor.execute(table_val)
    cursor.execute(table_test)

    return cursor


def insert_into_table(table, cursor, fen_id, fen, eval_val):
    sqlite_insert_with_param = "INSERT INTO " + table + \
                               "(id, FEN, EVAL) VALUES (?, ?, ?);"""
    data_tuple = (fen_id, fen, eval_val)
    cursor.execute(sqlite_insert_with_param, data_tuple)
    return cursor


def split_dataset_into_train_val_test_sql(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"):
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("SELECT FEN, EVAL from evaluations")
    total = 1
    num_in_train, num_in_val, num_in_test = 0, 0, 0
    counter_val, counter_test = 0, 0

    cursor_tables = con.cursor()
    cursor_tables = create_tables(cursor_tables)

    for res in cursor:
        if not isinstance(res[1], float):
            print("Not float: " + str(res))
            continue

        if counter_val >= 10:
            counter_val = 0
            cursor_tables = insert_into_table("val_table", cursor_tables, total, res[0], res[1])

            num_in_val += 1
        elif counter_test >= 10:
            counter_test = 0
            cursor_tables = insert_into_table("test_table", cursor_tables, total, res[0], res[1])

            num_in_test += 1
        else:
            counter_test += 1
            counter_val += 1
            cursor_tables = insert_into_table("train_table", cursor_tables, total, res[0], res[1])

            num_in_train += 1

        total += 1

    con.commit()
    cursor.close()
    cursor_tables.close()

    print(total)
    print(num_in_train)
    print(num_in_val)
    print(num_in_test)


def drop_tables(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"):
    con = sqlite3.connect(dataset_path)
    cursor = con.cursor()
    cursor.execute("DROP TABLE train_table;")
    cursor.execute("DROP TABLE val_table;")
    cursor.execute("DROP TABLE test_table;")
    con.commit()
    cursor.close()


def get_stats_from_dataset(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db"):
    """
        Print tables in dataset
    """
    # con = sqlite3.connect(dataset_path)
    # cursor = con.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())
    # return

    """
        Get std and mean
    """
    # con = sqlite3.connect(dataset_path)
    # cursor = con.cursor()
    # cursor.execute("SELECT AVG((train_table.EVAL - sub.a) * (train_table.EVAL - sub.a)) as var from train_table, "
    #                "(SELECT AVG(EVAL) AS a FROM train_table) AS sub")
    # std = math.sqrt(cursor.fetchall()[0][0])
    #
    # cursor = con.cursor()
    # cursor.execute("SELECT AVG(EVAL) AS a FROM train_table")
    # mean = math.sqrt(cursor.fetchall()[0][0])
    # print("std: " + str(std))
    # print("mean: " + str(mean))



    """
        Get min max values, as well as num of values in ranges
    """
    # con = sqlite3.connect(dataset_path)
    # cursor = con.cursor()
    # cursor.execute("SELECT FEN, EVAL from evaluations")

    file = open('/home/igor/Documents/Chess_bot/Datasets/train_data_balanced.csv')
    csvreader = csv.reader(file)
    next(csvreader)

    max_v = 0
    min_v = 0
    counter_25_pos = [0, 0, 0, 0, 0, 0]
    counter_25_neg = [0, 0, 0, 0, 0, 0]
    counter_pos = [0, 0, 0, 0]
    counter_neg = [0, 0, 0, 0]
    counter_zero = 0
    total = 0
    whites_turn = 0
    black_turn = 0
    for res in csvreader:

        if total % 100000 == 0:
            print(total)

        res = [res[0], float(res[1])]

        if isinstance(res[1], float):

            # board = chess.Board(res[0])
            #
            # if board.turn:
            #     whites_turn += 1
            # else:
            #     black_turn += 1

            if res[1] == 0:
                counter_zero += 1

            if 0 < res[1] <= 25:
                #counter_pos[0] += 1
                if 0 < res[1] <= 1:
                    counter_25_pos[0] += 1
                elif 1 < res[1] <= 5:
                    counter_25_pos[1] += 1
                elif 5 < res[1] <= 10:
                    counter_25_pos[2] += 1
                elif 10 < res[1] <= 15:
                    counter_25_pos[3] += 1
                elif 15 < res[1] <= 20:
                    counter_25_pos[4] += 1
                else:
                    counter_25_pos[5] += 1
            elif 25 < res[1] <= 50:
                counter_pos[1] += 1
            elif 50 < res[1] <= 75:
                counter_pos[2] += 1
            elif 75 < res[1]:
                counter_pos[3] += 1

            if -25 <= res[1] < 0:
                #counter_neg[0] += 1
                if -1 <= res[1] < 0:
                    counter_25_neg[0] += 1
                elif -5 <= res[1] < -1:
                    counter_25_neg[1] += 1
                elif -10 <= res[1] < -5:
                    counter_25_neg[2] += 1
                elif -15 <= res[1] < -10:
                    counter_25_neg[3] += 1
                elif -20 <= res[1] < -15:
                    counter_25_neg[4] += 1
                else:
                    counter_25_neg[5] += 1
            elif -50 <= res[1] < -25:
                counter_neg[1] += 1
            elif -75 <= res[1] < -50:
                counter_neg[2] += 1
            elif res[1] < -75:
                counter_neg[3] += 1

            total += 1

            if res[1] > max_v:
                max_v = res[1]
            elif res[1] < min_v:
                min_v = res[1]
        else:
            print(res[1])

    print(min_v)  # -152.65
    print(max_v)  # 152.65
    print("Total el: " + str(total))
    print("Counter positives to 25: " + str(counter_25_pos))
    print("Counter positives: " + str(counter_pos))
    print("Counter negatives to 25: " + str(counter_25_neg))
    print("Counter negatives: " + str(counter_neg))
    print("Counter zero: " + str(counter_zero))
    print("Total white turn: " + str(whites_turn))
    print("Total black turn: " + str(black_turn))


def get_dataset_partitions_tf(ds, batch_size, ds_size=38000000, train_split=0.8):
    train_size = int(int(train_split * ds_size) / batch_size)

    print(train_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    return train_ds, val_ds


def convert_fen_to_array(dataset_el):
    # stringic_valjda = bytes.decode(dataset_el)
    # board_chess = chess.Board(stringic_valjda)
    # features = get_board_state(board_chess)
    # return np.asarray(features, dtype=np.float32)
    batch_of_features = []
    for fen in dataset_el:
        stringic_valjda = bytes.decode(fen)
        board_chess = chess.Board(stringic_valjda)
        features = get_board_state(board_chess)
        batch_of_features.append(features)
    return np.asarray(batch_of_features, dtype=np.float32)


def get_dataset_sql(dataset_path="/home/igor/Documents/Chess_bot/Datasets/test.db", bs=256):
    """
    Dataset creation
    """
    dataset_train = tf.data.experimental.SqlDataset(
        "sqlite",
        dataset_path,
        "SELECT FEN, EVAL FROM train_table",
        (tf.string, tf.double)
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).shuffle(100000, reshuffle_each_iteration=True).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    dataset_val = tf.data.experimental.SqlDataset(
        "sqlite",
        dataset_path,
        "SELECT FEN, EVAL FROM val_table",
        (tf.string, tf.double)
    ).map(
        lambda x, y: (
            tf.numpy_function(
                func=convert_fen_to_array,
                inp=[x],
                Tout=tf.float32),
            tf.numpy_function(
                func=cut_value_and_normalize,
                inp=[y],
                Tout=tf.float32)
        ),
        tf.data.experimental.AUTOTUNE, deterministic=False
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_train, dataset_val


def get_dataset_csv(bs=256):
    """
    Dataset creation
    """
    train_dataset = tf.data.experimental.make_csv_dataset(
        "/home/igor/Documents/Chess_bot/Datasets/train_data.csv",
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
        "/home/igor/Documents/Chess_bot/Datasets/val_data.csv",
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


def get_dataset_for_rl_csv(bs=1):
    """
        Dataset creation
        """
    train_dataset = tf.data.experimental.make_csv_dataset(
        "/home/igor/Documents/Chess_bot/Datasets/train_data.csv",
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
        "/home/igor/Documents/Chess_bot/Datasets/val_data.csv",
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


def get_test_dataset_csv(bs=256):
    test_dataset = tf.data.experimental.make_csv_dataset(
        "/home/igor/Documents/Chess_bot/Datasets/test_data.csv",
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


if __name__ == "__main__":
    # split_dataset_into_train_val_test_sql()
    # split_dataset_into_train_val_test_csv_files()
    get_stats_from_dataset()
    # split_dataset_into_train_val_test_csv_files_balanced()
