from tensorflow.keras.layers import Dense, Dropout, Add, ReLU, Input, BatchNormalization
import tensorflow as tf
from dataset import get_dataset_csv, get_test_dataset_csv
import mlflow
from loguru import logger
from mlflow import log_param, log_text
import argparse


def get_nn_model():
    input_layer = Input(shape=(773,))
    y = Dense(4096, activation="relu")(input_layer)
    y = Dropout(0.4)(y)
    y = Dense(2048, activation="relu")(y)
    y = Dropout(0.4)(y)
    y = Dense(2048, activation="relu")(y)
    y = Dropout(0.4)(y)
    y = Dense(1, activation="linear")(y)
    model = tf.keras.Model(inputs=input_layer, outputs=y)
    return model


def train(dataset_train_path, dataset_val_path, batch_size=256, epoch_num=8):
    logger.info("Started training")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    with mlflow.start_run():
        model = get_nn_model()
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
        model.summary()

        dataset_train, dataset_val = get_dataset_csv(dataset_train_path, dataset_val_path, bs=batch_size)

        mlflow.tensorflow.autolog(log_models=True)
        model.fit(x=dataset_train,
                  validation_data=dataset_val,
                  epochs=epoch_num,
                  callbacks=[early_stopping])

        mlflow.keras.log_model(model, "logged_model")
        log_param("batch_size_of_dataset", batch_size)
        log_text("Dataset is split 80-10-10."
                 "Created from csv database. \nOriginal evaluations above 5 are "
                 "calculated by: \n"
                 """
                    if evaluation_val > 5:
                        evaluation_val = 4.3 + math.log10(evaluation_val)
                    elif evaluation_val < -5:
                        evaluation_val = -4.3 - math.log10(-evaluation_val)

                    evaluation_val = 2 * (evaluation_val + 6.48) / (6.48 + 6.48) - 1\n""" +
                 "Network architecture: 3 dense layers: 4096, 2048, 1024. Betwen each is 0.4 dropout." +
                 "Also added early stopping.\n ",
                 "note.txt")

    logger.info("Finished training")


def test_network(model_path, dataset_test_path):
    model = tf.keras.models.load_model(model_path)

    test_dataset = get_test_dataset_csv(dataset_test_path)

    results = model.evaluate(test_dataset)
    print("test loss, test acc:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_path', required=True, type=str)
    parser.add_argument('--val_dataset_path', required=True, type=str)

    args = parser.parse_args()

    train(args.train_dataset_path, args.val_dataset_path)


