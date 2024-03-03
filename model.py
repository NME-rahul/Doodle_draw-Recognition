from tensorflow.keras import layers, models
import tensorflow as tf

def model_x():
  input_layer = layers.Input(shape=(9, ))
  x = tf.expand_dims(input_layer, axis=1)
  x = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(x)
  skip_connection = x
  x = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(x)
  x = layers.Concatenate()([x, skip_connection])
  x = layers.Bidirectional(layers.LSTM(128))(x)
  x = layers.LayerNormalization()(x)
  x = layers.Flatten()(x)
  output_layer = layers.Dense(256, activation='relu')(x)
  return models.Model(input_layer, output_layer)


def model_y():
  input_layer_y = layers.Input(shape=(9, ))
  y = tf.expand_dims(input_layer_y, axis=1)
  y = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(y)
  skip_connection = y
  y = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(y)
  y = layers.Concatenate()([y, skip_connection])
  y = layers.Bidirectional(layers.LSTM(128))(y)
  y = layers.LayerNormalization()(y)
  y = layers.Flatten()(y)
  output_layer_y = layers.Dense(256, activation='relu')(y)
  return models.Model(input_layer_y, output_layer_y)


def combine_model():
  in_x = layers.Input(shape=(9, ))
  in_y = layers.Input(shape=(9, ))
  x = model_x([in_x])
  y = model_y([in_y])
  xy = layers.Concatenate()([x, y])
  prediction_layer = layers.Dense(stop_pt, activation='softmax')(xy)
  return models.Model([in_x, in_y], prediction_layer)


def DoodleDraw():
  model = combine_model()
  model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
  )
  return model
