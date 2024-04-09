# ====================================================================================
#                               FUNCTIONS
# ====================================================================================
import matplotlib.pyplot as plt 
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
import datetime
import numpy as np

def plot_loss(history):
  '''
  Plot epochs and errors
  :param history: fitting process of the NN model
  :return plot with epochs and error
  '''
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 100])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

def build_and_compile_model(norm):
  '''
  Build and compile NN model 
  '''
  model = tf.keras.Sequential([
      norm,
      layers.Dense(256, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['accuracy'])
  return model


def NN_model(dataset: pd.DataFrame, y_output_name:str, save_model: bool, model_name: str, n_predictions:int):
    '''
    Generate NN model
    :param dataset: overall dataset with features and labels
    :param y_output_name: name of the labels (output variable)
    :param save_model: True or False if wnbat to save the model 
    :param model_name: name of the NN model to be saved as pickle
    :return
      **NN_model**: NN model 
      **test_feature**: test features to be used in the validation (predictions)
      **test_label**: values of the output in validation (predictions)
    '''
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # sns.pairplot(train_dataset[['ETH', 'TRASMITTANZA_MEDIA_INVOLUCRO', 'SUPERFICIE_NETTA','DEGREE_DAYS']], diag_kind='kde')

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(y_output_name)
    test_labels = test_features.pop(y_output_name)

    # train_dataset.describe().transpose()[['mean', 'std']]
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())

    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()

    # Store logs
    log_folder = "Traininglogs/"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define callbacks
    callbacks = [TensorBoard(log_dir=log_folder,
                            histogram_freq=1,
                            write_graph=True,
                            write_images=True,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=1)]

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=300,
        callbacks=callbacks
    )  

    plot_loss(history)
    # Save model
    if save_model == 'yes':
        dnn_model.save(model_name)

    # MAKE PREDICITIONS
    test_predictions = dnn_model.predict(test_features[0:n_predictions]).flatten()

    # Validation plot: prediction vs testing data
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels[0:n_predictions], test_predictions[:n_predictions])
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [0, 300]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    # Error 
    pd.DataFrame({
        'real_value':test_labels.values.tolist()[0:n_predictions],
        'predictions':list(test_predictions)
    })

    error = test_predictions - test_labels[0:n_predictions]
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')
    
    return dnn_model