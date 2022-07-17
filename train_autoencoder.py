import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib 
import glob
from matplotlib import pyplot as plt
import argparse

# Create custom callback to plot model loss after each epoch
class PlotModelLossCallback(keras.callbacks.Callback):
    
    def __init__(self, output_loc=''):
        self.output_loc = output_loc
        self.train_losses = []
        self.val_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        
        fig, axs = plt.subplots(figsize=(10, 7))

        axs.plot(self.train_losses, label='Train')
        axs.plot(self.val_losses, label='Val')
        axs.set_title('Model loss')
        axs.set_ylabel('Loss')
        axs.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(self.output_loc + 'history.png')
        plt.close()

def build_model(tile_size):
    '''Creates and returns the model.'''
    
    input_img = keras.Input(shape=(tile_size, tile_size, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return(autoencoder)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train an autoencoder to reconstruct tiles derived from Barchan Dune images.')
    
    parser.add_argument('-ts', '--tile_size', default=64, type=int, required=True, help="Size of tiles to use.")
    parser.add_argument('-tiles', '--tile_dir', required=True, help="The directory containing input tiles.")
    parser.add_argument('-p', '--file_pattern', required=True, help="Expression used with glob to match input files.")
    parser.add_argument('-o', '--output_dir', required=True, help="Directory where the trained model and other outputs will be stored.")
    parser.add_argument('-e', '--epochs', default=50, type=int, help="Number of epochs to train the model for.")
    parser.add_argument('-b', '--batchsize', default=64, type=int, help="Batch size when training.")
    
    args = parser.parse_args()

    # Get list of all tile files
    tile_list = glob.glob(args.tile_dir + args.file_pattern + '/{}/tile_data.npy'.format(args.tile_size))

    print("Found {} tile stacks in {}".format(len(tile_list), args.tile_dir))

    # Load all tiles into a single list
    all_tiles = []

    for stack in tile_list:

        tile_arr = np.load(stack)
        tile_shape = tile_arr.shape
        tile_arr = tile_arr.reshape((tile_shape[0], args.tile_size, args.tile_size, 1))
        all_tiles.append(tile_arr)

    # Make one big numpy array
    all_tiles_arr = np.concatenate(all_tiles)

    # Normalise to 0-1 range
    # Images are uint16 but most values seem to be on the low end
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(all_tiles_arr.reshape(-1, 1))

    x_data = scaled_data.reshape(all_tiles_arr.shape)
    
    # Save scaler for re-use
    joblib.dump(scaler, args.output_dir + 'scaler.pkl')

    model = build_model(args.tile_size)

    # Set up a callback to save the model with the lowest loss so far
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=args.output_dir + 'best_model.h5',
                    save_weights_only=True,
                    monitor='loss',
                    mode='min',
                    save_best_only=True)

    plot_callback = PlotModelLossCallback(output_loc=args.output_dir)

    history = model.fit(x_data, x_data,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback, plot_callback])

    pred_tiles = model.predict(x_data)

    plt.figure(figsize=(20, 4))

    for i in range(10):

        # Pick a random tile
        tile_num = int(np.random.choice(pred_tiles.shape[0]))

        # Display original tile
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_data[tile_num])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed tile
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(pred_tiles[tile_num])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.text(-350, -10, 'Original', rotation='vertical')
    plt.text(-350, 30, 'Reconstructed', rotation='vertical')
    plt.savefig(args.output_dir + 'example_tiles.png')

