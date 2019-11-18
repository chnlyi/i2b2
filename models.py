from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNGRU, CuDNNLSTM, GRU, LSTM
       
# rnn models 
def get_rnn_model(nb_words, 
                  num_labels, 
                  embed_size, 
                  latent_dim, 
                  model_type, 
                  embedding_matrix, 
                  dropout, 
                  train_embed=True):
    
    inp = Input(shape=(None, ))
    if train_embed:
        x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    else:    
        x = Embedding(nb_words, embed_size)(inp)
    if model_type=='CuDNNGRU':
        x = Bidirectional(CuDNNGRU(latent_dim, return_sequences=True))(x)
    elif model_type=='GRU':
        x = Bidirectional(GRU(latent_dim, return_sequences=True))(x)
    elif model_type=='CuDNNLSTM':
        x = Bidirectional(CuDNNLSTM(latent_dim, return_sequences=True))(x)
    elif model_type=='LSTM':
        x = Bidirectional(LSTM(latent_dim, return_sequences=True))(x)
    else:
        raise ValueError('Please specify model_type as one of the following:n\CuDNNGRU, CuDNNLSTM, GRU, LSTM')
    #x = SeqSelfAttention(attention_width=15, attention_activation='sigmoid')(x)
    x = SpatialDropout1D(dropout)(x)
    x = Dense(300, activation="relu")(x)
    outp = Dense(num_labels, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    
    return model