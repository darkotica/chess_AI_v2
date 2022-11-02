from tensorflow.keras.layers import Dense, Dropout, Add, ReLU, Input, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
from dataset import get_dataset_sql, get_dataset_csv, get_test_dataset_csv
import os
import mlflow
from loguru import logger
from mlflow import log_param, log_text


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


def train(mlrun_id=None, epoch_num=8, epoch_from_which_to_start=0):
    logger.info("Started training")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    with mlflow.start_run():
        if mlrun_id:
            model = mlflow.keras.load_model("runs:/" + mlrun_id + "/model")
            print("Continuing run, run id: " + mlrun_id)
        else:
            model = get_nn_model()
            model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
        model.summary()

        batch_size = 256

        dataset_train, dataset_val = get_dataset_csv(bs=batch_size)

        mlflow.tensorflow.autolog(log_models=True)
        model.fit(x=dataset_train,
                  validation_data=dataset_val,
                  epochs=epoch_num,
                  initial_epoch=epoch_from_which_to_start,
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


def test_network(mlrun_id):
    print("model id: " + mlrun_id)
    model = mlflow.keras.load_model("runs:/" + mlrun_id + "/logged_model")

    test_dataset = get_test_dataset_csv()

    results = model.evaluate(test_dataset)
    print("test loss, test acc:", results)


if __name__ == "__main__":
    # sudo ldconfig /usr/lib/cuda/lib64
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    train()


