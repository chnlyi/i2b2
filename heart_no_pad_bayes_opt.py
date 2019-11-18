import numpy as np
from gensim.models import Word2Vec
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNGRU, CuDNNLSTM, GRU, LSTM, Reshape, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
#from keras_self_attention import SeqSelfAttention
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, hamming_loss, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from tqdm import tnrange, tqdm_notebook
from utils import process_data, multilabel_confusion_matrix, get_embedding_matrix, get_cat_labels, data_generator, get_all

# Customized Evaluation
class CustomEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = list(validation_data)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = []
            for x in self.X_val:
                y = np.squeeze(self.model.predict_on_batch(x))
                y_pred.append(y)
            y_pred = np.concatenate(y_pred)
            y_pred_ham = y_pred > 0.5
            y_val = np.concatenate(self.y_val)
            roc = roc_auc_score(y_val, y_pred, average='micro')
            loss = log_loss(y_val, y_pred)
            ham = hamming_loss(y_val, y_pred_ham)
            sub = accuracy_score(y_val, y_pred_ham)
            f1 = f1_score(y_val, y_pred_ham, average='micro')
            print("Adiitional val metrics: - ROC-AUC: %.6f - Log-Loss: %.6f - Hamming-Loss: %.6f - Subset-Accuracy: %.6f - F1-Score: %.6f" % (roc, loss, ham, sub, f1))

def no_pad_time_tuning(param, notes_train, labels_train, up_notes_train, up_labels_train, gold_labels_train, notes_test, labels_test, gold_labels_test, verbose=1):
    
    up = param['up']
    window_size = param['window_size']
    embed_size = param['embed_size'] * 10
    latent_dim = param['latent_dim'] * 64
    dropout_rate = param['dropout_rate']
    epochs = 30 #param['epochs']
    max_features = 60000 #param['max_features']
    category = False #param['category']
    embedding = True #param['embedding']
    model_type = 'CuDNNLSTM' #param['model_type']
    
    # upsampling
    if up > 0:
        if verbose != 0: print('upsampling for %d times...' % (up))
        notes_train = [note + up * up_note for note, up_note in zip(notes_train, up_notes_train)]
        labels_train = [label + up * up_label for label, up_label in zip(labels_train, up_labels_train)]
        if verbose != 0: print('upsampling done\n')
    notes = notes_train + notes_test
    labels = labels_train + labels_test
    gold_labels = gold_labels_train + gold_labels_test
    
    # prepare features
    if verbose != 0: print('preparing features ...')
    X_txt = [' '.join(i) for i in notes]
    X_train_txt = [' '.join(i) for i in notes_train]
    X_test_txt = [' '.join(i) for i in notes_test]
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(X_txt)
    X_seq = tokenizer.texts_to_sequences(X_txt) 
    X_train_seq = tokenizer.texts_to_sequences(X_train_txt) 
    X_test_seq = tokenizer.texts_to_sequences(X_test_txt) 
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    if verbose != 0: print('preparing features done\n')

    # prepare embedding matrix
    if embedding:
        if verbose != 0: print('preparing embedding matrix ...')
        w2v = Word2Vec(notes, size=embed_size, window=window_size, min_count=1, workers=4)
        embedding_index = dict(zip(w2v.wv.index2word, w2v.wv.vectors))
        embedding_matrix = get_embedding_matrix(embedding_index=embedding_index, word_index=word_index, max_features=max_features, embed_size=embed_size)
        if verbose != 0: print('preparing embedding matrix done\n')
        
    # prepare targets
    if verbose != 0: print('preparing targets ...')
    if category:
        # prepare cagtegory label targets
        labels = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels]
        labels_train = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels_train]
        labels_test = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels_test]
    all_labels = [label for notes_label in labels for label in notes_label]
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    num_labels = len(mlb.classes_)
    Y_train = []
    Y_test = []
    for i in labels_train:
        l = mlb.transform(i)
        Y_train.append(l)
    for i in labels_test:
        l = mlb.transform(i)
        Y_test.append(l)
    if verbose != 0: print('preparing targets done\n')

    # model function with pretrained embedding matrix and Timedistributed
    def get_model(nb_words, num_labels, model_type, embedding):
        inp = Input(shape=(None, ))
        if embedding:
            x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
        else:    
            x = Embedding(nb_words, embed_size)(inp)
        x = SpatialDropout1D(dropout_rate)(x)
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
        outp = TimeDistributed(Dense((num_labels), activation="sigmoid"))(x)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    # model summary
    model = get_model(nb_words=nb_words, num_labels=num_labels, model_type=model_type, embedding=embedding)
    if verbose != 0: print('\nmodel summary:')
    if verbose != 0: print(model.summary())

    # model training
    if verbose != 0: print('\ntraining model ...')
    custevl = CustomEvaluation(validation_data=(X_test_seq, Y_test), interval=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=3e-4, patience=3, verbose=0, mode='auto')
    train_gen = data_generator(X_train_seq, Y_train)
    test_gen = data_generator(X_test_seq, Y_test)
    v = 1 if verbose != 0 else 0  
    hist = model.fit_generator(train_gen,
                                steps_per_epoch=len(Y_train),
                                epochs=epochs,
                                validation_data=test_gen,
                                validation_steps=len(Y_test),
                                callbacks=[custevl, earlystop],
                                verbose=v)
    if verbose != 0: print('training model done')

    # prediction of test data
    if verbose != 0: print('predicting test data ...')
    Y_pred = []
    for x in X_test_seq:
        x = np.array(x).reshape((1,-1))
        y_pred = np.squeeze(model.predict_on_batch(x))
        Y_pred.append(y_pred)
    Y_pred_concat = np.concatenate(Y_pred)
    Y_val = np.concatenate(Y_test)
    if verbose != 0: print('predicting test data done\n')

    # confusion matrix 
    if verbose == 2: 
        cm = multilabel_confusion_matrix(Y_val, np.where(Y_pred_concat > 0.5, 1, 0))
        for i, j in zip(cm, mlb.classes_):
            print(j+':\n', i,'\n')

    # prepare gold label targets
    if verbose != 0: print('predicting gold label targets ...')
    gold_labels_pred = [{i for s in mlb.inverse_transform(y_pred>0.5) for i in s if i != 'O'} for y_pred in Y_pred]
    gmlb = MultiLabelBinarizer()
    gmlb.fit(gold_labels)
    num_gold_labels = len(gmlb.classes_)
    Y_gold_test = gmlb.transform(gold_labels_test)
    Y_gold_pred = gmlb.transform(gold_labels_pred)
    if verbose != 0: print('predicting gold label targets done\n')

    # confusion matrix for gold label
    if verbose == 2: 
        gcm = multilabel_confusion_matrix(np.concatenate(Y_gold_test), np.concatenate(Y_gold_pred))
        for i, j in zip(gcm, gmlb.classes_):
            print(j+':\n', i,'\n')

    # f1 scores for gold label
    f1 = f1_score(Y_gold_test, Y_gold_pred, average='micro')
    print('Parameters: up = %d, window_size = %d, embed_size = %d, latent_dim = %d, dropout_rate = %.3f' % (up, window_size, embed_size, latent_dim, dropout_rate))
    print('F1 Scores for global labels:\nALL (average="micro"):', f1)
    
    with open("results.txt","a") as f:
        f.write('Parameters: up = %d, window_size = %d, embed_size = %d, latent_dim = %d, dropout_rate = %.3f\n' % (up, window_size, embed_size, latent_dim, dropout_rate))
        f.write('F1 Scores for global labels(average="micro"): %.3f\n' % f1)
        
    
    if verbose == 2: 
        f1_all = f1_score(Y_gold_test, Y_gold_pred, average=None)
        for i, j in zip(f1_all, gmlb.classes_):
            print(j+': '+str(i))
    
    print('\n\n')
          
    return f1

def my_bayes_opt(space):
    param = {
        'up': space[0],               # Times of upsampling for training data
        'window_size': space[1],                # Window size for word2vec
        'embed_size': space[2],                # Length of the vector that we willl get from the embedding layer
        'latent_dim': space[3],               # Hidden layers dimension 
        'dropout_rate': space[4]}#,             # Rate of the dropout layers
        #'epochs': space[0],                    # Number of epochs
        #'max_features': space[0],           # Max num of vocabulary
        #'category': space[0],               # Is categoty labels
        #'embedding': space[0],               # Using pre-made embedidng matrix as weight
        #'model_type': space[0]
        #}
    f1 = no_pad_time_tuning(param, notes_train, labels_train, up_notes_train, up_labels_train, gold_labels_train, notes_test, labels_test, gold_labels_test)
    return (-f1)     

if __name__ == "__main__":
    
    # loading data 
    notes_train_1, labels_train_1, up_notes_train_1, up_labels_train_1, gold_labels_train_1 = get_all('/host_home/data/i2b2/2014/training/training-RiskFactors-Complete-Set1') 
    notes_train_2, labels_train_2, up_notes_train_2, up_labels_train_2, gold_labels_train_2 = get_all('/host_home/data/i2b2/2014/training/training-RiskFactors-Complete-Set2') 

    notes_train = notes_train_1 + notes_train_2
    labels_train = labels_train_1 + labels_train_2
    up_notes_train = up_notes_train_1 + up_notes_train_2
    up_labels_train = up_labels_train_1 + up_labels_train_2
    gold_labels_train = gold_labels_train_1 + gold_labels_train_2

    notes_test, labels_test, _1, _2, gold_labels_test = get_all('/host_home/data/i2b2/2014/testing/testing-RiskFactors-Complete') 

    space = [Integer(5, 10, name='up'),
            Integer(3, 7, name='window_size'),
            Integer(2, 4, name='embed_size'),
            Integer(1, 3, name='latent_dim'),
            Real(0, 0.3, name='dropout_rate')]
            #Integer(30, 30, name='epochs'),
            #Integer(1, 60000, name='max_features'),
            #Categorical([False], name='category'),
            #Categorical([True], name='embedding'),
            #Categorical(['CuDNNLSTM'], name='model_type')]
    x0 = [7, 4, 3, 2, 0.15253569878187465]
    res = gp_minimize(my_bayes_opt, space, x0=x0, n_calls=100, verbose=True)
    
    # python command: python heart_no_pad_bayes_opt.py > result_heart_no_pad_bayes_opt.txt
    
    print(res)

