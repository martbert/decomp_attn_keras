import os
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout
from keras.layers import Permute, concatenate, Masking, dot
from keras.models import Model
from keras import optimizers
from keras.models import load_model

from model.layers import Masking2D, Softmax2D, MaskedConv1D, MaskedGlobalAveragePooling1D
from utils.data import SNLIData

class DecomposableAttention:
    def __init__(self):
        # Model Parameters
        self.EmbeddingSize = 300
        self.HiddenSize = 200
        self.SentMaxLen = 42
        self.DropProb = 0.4
        self.Optimizer = optimizers.Adagrad(lr=0.01)
        
        self.FKernelSize = 3
        self.GKernelSize = 1

        self.prefix = 'decomp_attn_snli'

    # Feed forward network implemented as a 1D convolution
    def feed_forward(self, shape, n_hidden=200, n_layers=1, 
        dropout=0.2, masking=True, use_bias=True, activation='relu',
        kernel_size=1):
        
        inputs = Input(shape=shape)
        t = inputs
        for i in range(n_layers):
            t = Masking()(t)
            t = MaskedConv1D(n_hidden, kernel_size, 
                activation=activation, use_bias=use_bias)(t)

        t = Dropout(dropout)(t)

        return Model(inputs=inputs, outputs=t)

    # Create the model based on [A Decomposable Attention Model, Ankur et al.] 
    def create_model(self, test_mode = False):
        # Sentence A and B are the inputs
        A = Input(shape=(None,self.EmbeddingSize))
        B = Input(shape=(None,self.EmbeddingSize))

        # In pratice, first projecting A and B has been found to improve results
        EmbdProject = self.feed_forward((None,self.EmbeddingSize),
            dropout=0, n_hidden=self.HiddenSize, activation=None)
        Ap = EmbdProject(A)
        Bp = EmbdProject(B)

        # Score each sentence
        scoreF = self.feed_forward((None, self.HiddenSize), n_layers=2,
            dropout=self.DropProb, n_hidden=self.HiddenSize,
            kernel_size=self.FKernelSize)
        S_A = scoreF(Ap)
        S_B = scoreF(Bp)

        # Calculate the align matrix
        # S_AB = S_AS_B^T
        S_AB = dot([S_A, Permute((2,1))(S_B)], axes=(2,1))

        # Row-wise attention 
        Sp_AB = Softmax2D()(Masking2D()(S_AB))
        Sp_BA = Softmax2D()(Masking2D()(Permute((2,1))(S_AB)))

        # A soft aligned with B and vice-versa
        A_B = dot([Sp_BA, Ap], axes=(2,1))
        B_A = dot([Sp_AB, Bp], axes=(2,1))

        # Concatenate the sentences to their aligned counterparts
        v1 = concatenate([Ap, B_A]) 
        v2 = concatenate([Bp, A_B])

        # Score the concatenations and perform average pooling
        scoreG = self.feed_forward((None, 2*self.HiddenSize), n_layers=2,
            dropout=self.DropProb, n_hidden=self.HiddenSize, 
            kernel_size=self.GKernelSize)
        v1 = scoreG(v1)
        v2 = scoreG(v2)

        v1 = MaskedGlobalAveragePooling1D()(Masking()(v1))
        v2 = MaskedGlobalAveragePooling1D()(Masking()(v2))

        # Concatenate and score
        final = concatenate([v1, v2])
        
        ## scoreH
        for i in range(2):
            final = Dense(self.HiddenSize, activation='tanh')(final)

        # Predict the label
        y_hat = Dense(3, activation='softmax')(final)

        self.model = Model(inputs=[A,B], outputs=y_hat)

    def compile_model(self, gaussian_size=0):
        """ Load Possible Existing Weights and Compile the Model """
        if gaussian_size:
            weights = self.model.get_weights()
            weights = [np.random.normal(scale=gaussian_size,size=w.shape) for w in weights]
            self.model.set_weights(weights)

        self.model.compile(optimizer=self.Optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())

    def load(self, path=None):
        print('Loading a previously trained model...')
        if os.path.exists(path):
            self.model = load_model(path, custom_objects={
                'MaskedConv1D':MaskedConv1D,
                'Masking2D':Masking2D,
                'Softmax2D':Softmax2D,
                'MaskedGlobalAveragePooling1D':MaskedGlobalAveragePooling1D}
                )

    def train_on_generator(self, train_generator, valid_generator, n_train, 
        n_valid, batch_size=128, epochs=10, checkpointing=True):
        # Create folder and path for checkpoints
        pfx = "{}.h{}.d{}.l{}.fk{}.gk{}".format(self.prefix, self.HiddenSize, 
            self.DropProb, self.SentMaxLen, self.FKernelSize, self.GKernelSize)
        if not os.path.exists(pfx):
            os.mkdir(pfx)
        self.save_path = os.path.join(pfx, pfx)

        # Save structure to yaml file
        yaml_str = self.model.to_yaml()
        with open(self.save_path+'.yml', 'w') as f:
            f.write(yaml_str)

        if checkpointing:
            callbacks = [
              ModelCheckpoint(
                  self.save_path+'_best.hdf5', 
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=True,
                  mode='min')
            ]
        else:
            callbacks = []

        self.model.fit_generator(
          train_generator,
          int(n_train / batch_size),
          epochs=epochs,
          validation_data=valid_generator,
          validation_steps=int(n_valid / batch_size),
          callbacks=callbacks
        )

        # Aggregate history
        history = self.model.history.history

        # Save history
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(self.save_path+'.hist.csv', index=None)

################################################################################
################################################################################
################################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', 
        default='data/glove.840B.300d.snli_pruned.txt', type=str)
    parser.add_argument('--prefix', default='decomp_attn_snli', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--max_length', default=42, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--fk_size', default=1, type=int)
    parser.add_argument('--gk_size', default=1, type=int)
    parser.add_argument('--load', default=None, type=str)
    args = parser.parse_args()

    noise = 0

    print('Loading the SNLI dataset...')
    sd = SNLIData(
        data_path='data/snli.csv',
        max_length=args.max_length,
        nlp_vectors=args.vectors,
        pos_to_remove=['SPACE'],
        categorize=True,
        normed=True
      )

    model = DecomposableAttention()
    model.Optimizer = optimizers.Adagrad(lr=0.01)
    model.DropProb = args.dropout
    model.SentMaxLen = args.max_length
    model.HiddenSize = args.hidden_size
    model.FKernelSize = args.fk_size
    model.GKernelSize = args.gk_size
    model.prefix = args.prefix
    model.create_model()
    model.compile_model(gaussian_size=0.05)

    if args.load is not None:
        model.load(args.load)

    model.train_on_generator(
        sd.batch_generator(indices_set='train', batch_size=args.batch_size, noise=noise),
        sd.batch_generator(indices_set='valid', batch_size=args.batch_size, noise=0),
        sd.train_idx.shape[0],
        sd.valid_idx.shape[0],
        batch_size=args.batch_size,
        epochs=args.epochs
      )

    sd.save_oov_to_path()