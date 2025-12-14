import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras

def build_mlp_model(input_dim, hidden_layers=[128, 64], output_dim=10):
    """
    Constructs the Multi-Layer Perceptron from Lab 3.3.
    """
    layers = [keras.Input(shape=(input_dim,))]
    
    for units in hidden_layers:
        layers.append(keras.layers.Dense(units, activation="relu"))
        layers.append(keras.layers.Dropout(0.2)) # Best practice
        
    layers.append(keras.layers.Dense(output_dim, activation="softmax"))
    
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model