from tensorflow.keras.layers import Dense, Dropout, Add, ReLU, Input, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
from dataset import get_dataset_sql, get_dataset_csv, get_test_dataset_csv
import os
import mlflow
from loguru import logger
from mlflow import log_param, log_text


def residual_block(input_layer, size):  # , drop_first=False):
    fx = BatchNormalization()(input_layer)
    fx = ReLU()(fx)
    fx = Dense(size)(fx)
    fx = Dropout(0.4)(fx)
    fx = BatchNormalization()(fx)
    fx = ReLU()(fx)
    fx = Dense(size)(fx)
    fx = Dropout(0.4)(fx)
    out = Add()([input_layer, fx])
    out = ReLU()(out)
    return out

    # if not drop_first:
    #     fx = Dense(size, activation="relu", kernel_regularizer=regularizers.l2(0.005))(input_layer)
    #     fx = Dropout(0.2)(fx)
    #     #fx = BatchNormalization()(fx)
    # else:
    #     fx = input_layer
    # fx = Dense(size, activation="relu", kernel_regularizer=regularizers.l2(0.005))(fx)
    # fx = Dropout(0.2)(fx)
    # #fx = BatchNormalization()(fx)
    # fx = Dense(size, activation="relu", kernel_regularizer=regularizers.l2(0.005))(fx)
    # fx = Dropout(0.2)(fx)
    # #fx = BatchNormalization()(fx)
    # out = Add()([input_layer, fx])
    # out = ReLU()(out)
    # return out


def get_nn_model():
    input_layer = Input(shape=(769,))
    y = Dense(4000, activation="relu")(input_layer)
    y = Dropout(0.4)(y)
    y = Dense(1300, activation="relu")(y)
    y = Dropout(0.4)(y)
    y = Dense(1000, activation="relu")(y)
    y = Dropout(0.4)(y)
    # y = residual_block(y, 2048)
    # y = residual_block(y, 2048)

    # y = Dense(1024)(y)  # prelaz
    # y = Dropout(0.4)(y)
    #
    # y = residual_block(y, 1024)
    # y = residual_block(y, 1024)
    y = Dense(1, activation="linear")(y)
    model = tf.keras.Model(inputs=input_layer, outputs=y)
    return model


def train(mlrun_id=None, epoch_from_which_to_start=0):
    logger.info("Started training")

    # steps = 5 * 120979  # steps * batches
    # lr_decayed_fn = tf.keras.experimental.CosineDecay(
    #     initial_learning_rate=0.001, decay_steps=steps
    # )
    #
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    with mlflow.start_run():
        if mlrun_id:
            model = mlflow.keras.load_model("runs:/" + mlrun_id + "/model")
            print("Continuing run, run id: " + mlrun_id)
        else:
            model = get_nn_model()
            # model.compile(
            #     loss="mean_squared_error",
            #     optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
            model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
        model.summary()
        dataset_train, dataset_val = get_dataset_csv(bs=64)

        mlflow.tensorflow.autolog(log_models=True)
        model.fit(x=dataset_train, validation_data=dataset_val, epochs=8, initial_epoch=epoch_from_which_to_start,
                  callbacks=[early_stopping])

        mlflow.keras.log_model(model, "logged_model")
        log_param("batch_size_of_dataset", 256)
        log_text("Dataset is split 80-10-10. Not balanced. "
                 "Created from sqlite database. \nOriginal evaluations above 5 are "
                 "calculated by: \n"
                 """
                    if evaluation_val > 5:
                        evaluation_val = 4.3 + math.log10(evaluation_val)
                    elif evaluation_val < -5:
                        evaluation_val = -4.3 - math.log10(-evaluation_val)

                    evaluation_val = 2 * (evaluation_val + 6.48) / (6.48 + 6.48) - 1\n""" +
                 "Params " +
                 "no res blocks, 3 dense layers: 4096, 2048, 1024. Uga buga. No norm. Only 0.4 dropout." +
                 "Also added early stopping. IMPORTANT: added 1 more status flags: turn\n" +
                 "Added prefetch after batch instead of before. Batch size 64",
                 "note.txt")

    logger.info("Finished training")


def test_network(mlrun_id):
    print("model id: " + mlrun_id)
    model = mlflow.keras.load_model("runs:/" + mlrun_id + "/logged_model")

    test_dataset = get_test_dataset_csv()

    results = model.evaluate(test_dataset)
    print("test loss, test acc:", results)


# training loss
# 0.098
# 0.090
# 0.087
# 0.085
# 0.084
# 0.083
# 0.082
# 0.081
# 0.080
# 0.079

# val loss
# 0.085
# 0.081
# 0.075
# 0.073
# 0.072
# 0.070
# 0.071
# 0.068
# 0.066
# 0.068


if __name__ == "__main__":
    # sudo ldconfig /usr/lib/cuda/lib64
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # a233306bf71440388a1e81194e3d3141

    # todo kao probaj batch size menjati xd
    train()
    # import os
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #
    # test_network("95630a707a764c89ab677679eb41ced2")
    # 0.0693 sa alpha 0.00001 na kraju 2 steps

    # test_network("15ff8fdc93cd44d888ca4069d4dc73e9")  # 4096, 2048, 2048 after 8 epochs
    # test loss: 0.068

    # test_network("f9b2db8ff217469fbbf43acd23d3ae60")  # 4096, 2048, 2048
    # test loss: 0.0720, val loss: 0.719

    # test_network("e687e9fa7050451fb91a0ed1b8510fb6")  # 2048, 1024, 512
    # test loss: 0.0742, val loss: 0.741

    # test_network("ca12886f20e54c6498c0143ad3d2bc83")  # 2048, 2048
    # test loss: 0.0754, val loss: 0.753

    # test_network("887f4506ebb94246b4f4aad26b3d5435")  # 2048, 2048, 2048
    # test loss: 0.772, val loss: 0.771



    # TODO - sredi disbalans crni i beli obavezno



    # TODO experiments
    # CONV2D
    # DATASET - vrednosti iznad 18-18.5 su uglavnom mat forsirane linije, mozda nesto da se uradi sa time
    #   - da se secnu sve te vrednosti
    #   - da se zakucaju na 1 il nesto slicno
    # Da se doda u niz feature-a i to ko igra sledeci potez
    # todo - mozda da li je check pozicija
    # todo - pogledaj ono za remaining depth u alfa beta
    # todo - mozda neka arhi sa drugacijim dropoutom
    # deepchess: ludacki autoencoder-siamese network mozda baciti oko

    # plan za ponedeljak:
    # - TESTIRATI NAD TESTNIM SKUPOM SVE MREZE KOJE SU OKEJ PA VIDI KOJA JE NAJBOLJA

    # - probati conv2d mozda cak bude radilo
    # - probaj sa 10 umesto 5 ona granica
    # - promena dropout-a


