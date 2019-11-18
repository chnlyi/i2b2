import os
import re 
import string
import numpy as np
import xml.etree.ElementTree as et
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm_notebook

######### for data #########
def get_text_tags(xml_file, mae=False):
    """
    This function loads individual xml file in ibi2 2014 Heart Disease Risk Factors Challenge. 
    Returns the clinical note as a string and tags as a list of xml object
    """
    text = et.parse(xml_file).getroot()[0].text
    tags = []
    if mae:
        for i in et.parse(xml_file).getroot()[1]:
            tag = i.attrib
            tag.update({'tag': i.tag})
            tags.append(tag)  
    else:
        for i in et.parse(xml_file).getroot()[1]:
            for j in i:
                tag = j.attrib
                tag.update({'tag': j.tag})
                tags.append(tag)       
    return text, tags

def get_gold_tags_direct(xml_file):
    """
    This function loads individual gold xml file in ibi2 2014 Heart Disease Risk Factors Challenge. 
    Returns the gold tags as a list
    """
    tags = []
    for i in et.parse(xml_file).getroot()[1]:
        tag = i.attrib
        
        if i.tag == 'MEDICATION':
            token_tag = 'I'+'.'+i.tag+'.'+tag.get('type1',).replace(' ', '_')+'.'+tag.get('time').replace(' ', '_')
        elif i.tag == 'FAMILY_HIST':
            token_tag = 'I'+'.'+i.tag+'.'+tag.get('indicator').replace(' ', '_')+'.NA'
        elif i.tag == 'SMOKER':
            token_tag = 'I'+'.'+i.tag+'.'+tag.get('status')+'.NA'
        else:
            token_tag = 'I'+'.'+i.tag+'.'+tag.get('indicator').replace(' ', '_').replace('.', '')+'.'+tag.get('time').replace(' ', '_')
            
        tags.append(token_tag)
    return set(tags)

def get_gold_labels_direct(xml_folder):
    gold_labels = []
    for i in os.listdir(xml_folder):
        file = os.path.join(xml_folder, i)
        if file.endswith('.xml'):
            gold_tags = get_gold_tags_direct(file)
            gold_labels.append(gold_tags)
    return gold_labels

##########################################################################################################################
# def get_words_locations(text, punc='!",.:;?'):
#     """
#     This function tokenize the clinical note as a list of words, with the consideration of removing parenthesis pairs. 
#     It also returns the start and end index of each words in original clinical note string. 
#     """
#     words = []
#     locations = []
#     for i in WhitespaceTokenizer().span_tokenize(text):
#         start = i[0]
#         end = i[1]
#         original_word = text[start:end]
#         if original_word.startswith('(') and original_word.endswith(')'):
#             word = original_word.lstrip('(').rstrip(')')
#             start = start + 1
#             end = end - 1
#         elif original_word.startswith('(') and not(original_word.endswith(')')):
#             word = original_word.lstrip('(')
#             start = start + 1
#         elif not(original_word.startswith('(')) and original_word.endswith(')') and '(' not in original_word:
#             word = original_word.rstrip(')')
#             end = end - 1
#         else:
#             word = original_word.rstrip(punc)
#             end = end - len(original_word) + len(word)
#         if word != '':
#             words.append(word)
#             locations.append((start, end))
#     return words, locations
##########################################################################################################################
def get_words_locations(txt):
    sents = PunktSentenceTokenizer().tokenize(txt)
    words = [TreebankWordTokenizer().tokenize(i) for i in sents]
    words = [i for word in words for i in word]
    span_sents = [i for i in PunktSentenceTokenizer().span_tokenize(txt)]
    span_words = [[j for j in TreebankWordTokenizer().span_tokenize(i)] for i in PunktSentenceTokenizer().tokenize(txt)]
    new_span_words = []
    for i, j in zip(span_sents, span_words):
        new_span_words_in_sent = []
        for k in j:
            new_span_words_in_sent.append((k[0] + i[0], k[1] + i[0]))
        new_span_words.append(new_span_words_in_sent)
    new_span_words = [i for span_word in new_span_words for i in span_word]
    _words = []
    _span_words = []
    for i, j in zip(words, new_span_words):
        if i not in string.punctuation:
            punkt_list = re.split('\W', i.lower())
            if len(punkt_list) >= 2 and not all([ii.isnumeric() for ii in punkt_list]):
                mm = j[0]
                for k in punkt_list:
                    if k == '':
                        mm += 1
                    elif re.match('^\d+[a-zA-Z]+$', k):
                        k0 = re.search('(^\d+)',k).group()
                        k1 = re.search('([a-zA-Z]+$)',k).group()
                        _words.extend([k0, k1])
                        _span_words.append((mm, mm+len(k0)))
                        mm += len(k0)
                        _span_words.append((mm, mm+len(k1)))
                        mm += len(k1)
                        mm += 1                                           
                    else:
                        _words.append(k)
                        _span_words.append((mm, mm+len(k)))
                        mm += len(k)
                        mm += 1
            else:    
                _words.append(i.lower())
                _span_words.append(j)    
    return _words, _span_words
##########################################################################################################################
# def get_words_locations(text, punc='!",.:;?'):
#     """
#     This function tokenize the clinical note as a list of words, with the consideration of removing parenthesis pairs. 
#     It also returns the start and end index of each words in original clinical note string. 
#     """
#     sep = r'\s+|(?<!\d)/(?!\d)|\(|\)|(?<!\d)\.(?!\d)|(?<=\d)\.(?=\w)|\-|\:|,'
#     words = []
#     locations = []
#     for i in RegexpTokenizer(sep, gaps=True).span_tokenize(text):
#         start = i[0]
#         end = i[1]
#         original_word = text[start:end]
#         word = original_word.rstrip(punc)
#         end = end - len(original_word) + len(word)
#         if word != '':
#             words.append(word)
#             locations.append((start, end))
#     return words, locations
##########################################################################################################################


def get_labels(tags, locations):
    """
    This function takes the tags and the start and end index of each words in original clinical note string, 
    converts to a list of customized tags correspnding to the words list.  
    """
    labels = []
    for start, end in locations:
        token_tags = ['O']
        for j in tags:
            try:
                _start = int(j.get('start'))
                _end = int(j.get('end'))
#                 if set(range(start, end)).intersection(set(range(_start, _end))) != set():
                if start >= int(j.get('start')) and end <= int(j.get('end')):
#                 if start >= int(j.get('start')) and end < int(j.get('end')):
                    token_tags = []
            except:
                continue
        for j in tags:
            try:
#                 if set(range(start, end)).intersection(set(range(_start, _end))) != set():
                if start >= int(j.get('start')) and end <= int(j.get('end')):
#                 if start >= int(j.get('start')) and end < int(j.get('end')):
                    if j['tag'] == 'MEDICATION':
                        token_tag = 'I'+'.'+j['tag']+'.'+j.get('type1',).replace(' ', '_')+'.'+j.get('time').replace(' ', '_')
                    elif j['tag'] == 'FAMILY_HIST':
                        token_tag = 'I'+'.'+j['tag']+'.'+j.get('indicator').replace(' ', '_')+'.NA'
                    elif j['tag'] == 'SMOKER':
                        token_tag = 'I'+'.'+j['tag']+'.'+j.get('status')+'.NA'
                    else:
                        token_tag = 'I'+'.'+j['tag']+'.'+j.get('indicator').replace(' ', '_').replace('.', '')+'.'+j.get('time').replace(' ', '_')
                    token_tags.append(token_tag)
                    type_tag = j['tag']
                    tagged = True
            except:
                continue
        labels.append(set(token_tags))
    return labels

def get_words_labels(file):
    """
    This function loads individual xml file in ibi2 2014 Heart Disease Risk Factors Challenge. 
    Returns the words list of the clinical note, list of customized tags correspndingly and the set of all customized tags except 'O'.
    """
    text, tags = get_text_tags(file)    
    words, locations = get_words_locations(text)
    labels = get_labels(tags, locations)
    gold_labels = set([j for i in labels for j in i if j != 'O'])
    if any([i.startswith('I.FAMILY_HIST.') for i in gold_labels]) is False:
        gold_labels.add('I.FAMILY_HIST.not_present.NA')
    if any([i.startswith('I.SMOKER.') for i in gold_labels]) is False:
        gold_labels.add('I.SMOKER.unknown.NA')
    return words, labels, gold_labels

def get_tagged_sents(file, re_sep='\n'): # re_sep='\n|(?<!\d)[,.]|[,.](?!\d)' -- some bugs
    """
    This function loads individual xml file in ibi2 2014 Heart Disease Risk Factors Challenge. 
    Returns the list of lines (sentences---bugs) that are tagged with any except 'O', in the form of words lists.
    """
    p = re.compile(re_sep)
    tagged_sents = []
    text, tags = get_text_tags(file)  
    pl = [i.start() for i in p.finditer(text)]
    for i, _ in enumerate(pl):
        start = pl[i]
        try:
            end = pl[i+1]
        except:
            end = len(text)
        sent = text[start:end]
        tagged = False
        for j in tags:
            try:
                if start <= int(j.get('start')) and end >= int(j.get('end')):
#                 if start <= int(j.get('start')) and end > int(j.get('end')):
                    tagged = True
            except:
                continue
        if tagged:
            s = sent.strip()
            #if s.startswith('.') or s.startswith(','):
            #    s = s[1:].strip()
            words, _ = get_words_locations(s)
            tagged_sents.append(words)
    return tagged_sents

def slice_x_in_y(x, y, sep='|'):
    """
    Gets the start and end index of the string list x out of the string list y.
    Make sure sep is not in either x or y.
    """
    x1 = sep.join(x)
    y1 = sep.join(y)
    nx = x1.count(sep)
    n1 = None
    n2 = None
    if x1 in y1:
        if x[0] != y[0] and x[-1] != y[-1]:
            y12 =  y1.replace(x1,'').split(sep*2)
            n1 = y12[0].count(sep) + 1
            n2 = nx + n1 + 1
        if x[0] == y[0]:
            n1 = 0
            n2 = len(x)
        if x[-1] == y[-1]:
            n1 = len(y) - len(x)
            n2 = len(y)
    return slice(n1, n2)

def get_tagged_sents_labels(tagged_sents, words, labels):
    """
    This function gets the list of corresponding tags lists of the tagged sentences in the clinical notes.
    """
    tagged_sents_labels = []
    for sent in tagged_sents:
        l = labels[slice_x_in_y(sent, words)]
        tagged_sents_labels.append(l)
    return tagged_sents_labels

def get_all_words_labels(file):
    """
    This function gets words, tags, concatenated words of tagged sentences and their taggs.
    """
    words, labels, gold_labels = get_words_labels(file)
    tagged_sents = get_tagged_sents(file)
    tagged_sents_labels = get_tagged_sents_labels(tagged_sents, words, labels)
    up_words = [item for sublist in tagged_sents for item in sublist]
    up_labels = [item for sublist in tagged_sents_labels for item in sublist]
    return words, labels, gold_labels, up_words, up_labels

def up_sampling(file, n=0):
    """
    This function gets words and tags. It then add tagged sentences and their taggs for n times for up-sampling.
    """
    words, labels, gold_labels, up_words, up_labels = get_all_words_labels(file)
    words += n * up_words
    labels += n * up_labels
    return words, labels, gold_labels

def get_all_notes_labels(xml_folder):
    notes = []
    notes_labels = []
    up_notes = []
    up_notes_labels = []
    notes_gold_labels = []
    for i in tqdm_notebook(os.listdir(xml_folder)):
        file = os.path.join(xml_folder, i)
        if file.endswith('.xml'):
            words, labels, gold_labels, up_words, up_labels = get_all_words_labels(file)
            notes.append(words)
            notes_labels.append(labels)
            up_notes.append(up_words)
            up_notes_labels.append(up_labels)
            notes_gold_labels.append(gold_labels)
    return notes, notes_labels, up_notes, up_notes_labels, notes_gold_labels

def process_data(xml_folder, up=0):
    notes = []
    notes_labels = []
    notes_gold_labels = []
    if up > 0:
        print('Loading files with '+str(up)+' times upsampling for tagged lines in '+xml_folder)
    else:
        print('Loading files in '+xml_folder)
    for i in tqdm_notebook(os.listdir(xml_folder)):
        file = os.path.join(xml_folder, i)    
        words, labels, gold_labels = up_sampling(file, n=up)
        notes.append(words)
        notes_labels.append(labels)
        notes_gold_labels.append(gold_labels)
    return notes, notes_labels, notes_gold_labels

def check_data(xml_folder):
    notes = []
    notes_labels = []
    tagged_notes = []
    tagged_notes_labels = []
    for i in os.listdir(xml_folder):
        file = os.path.join(xml_folder, i)    
        words, labels, _ = get_words_labels(file)
        tagged_sents = get_tagged_sents(file)
        tagged_sents_labels = get_tagged_sents_labels(tagged_sents, words, labels)
        notes.append(words)
        notes_labels.append(labels)
        tagged_notes.append(tagged_sents)
        tagged_notes_labels.append(tagged_sents_labels)
    return notes, notes_labels, tagged_notes, tagged_notes_labels

######### other helper #########
# function help convert labels to category labels
def get_cat_labels(label):
    c = '.'
    positions = [pos for pos, char in enumerate(label) if char == c]
    if label != 'O':
        sl = slice(positions[0]+1,positions[1])
        cat_label = label[sl]
    else:
        cat_label = label
    return cat_label

# function help convert labels to category and indicator labels
def get_cat_ind_labels(label):
    c = '.'
    positions = [pos for pos, char in enumerate(label) if char == c]
    if label != 'O':
        sl = slice(positions[0]+1,positions[2])
        cat_ind_label = label[sl]
    else:
        cat_ind_label = label
    return cat_ind_label

# function help convert labels to time_flattened labels
def get_time_flattened_label(label):
    c = '.'
    positions = [pos for pos, char in enumerate(label) if char == c]
    if label != 'O':
        sl = slice(positions[0]+1,positions[2])
        time_flattened_labels = [label[sl], label[positions[2]+1:]]
    else:
        time_flattened_labels = [label]
    return time_flattened_labels

# function help convert flattened labels back to gold labels
def get_flattened_reverted(y_pred, mlb, all_gold_label):
    
    flattened_pred = mlb.inverse_transform(y_pred>0.5)
    categories = ['CAD', 'DIABETES', 'FAMILY_HIST', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'MEDICATION', 'O', 'OBESE', 'SMOKER']
    times = [ 'after_DCT', 'before_DCT', 'during_DCT', 'NA']
    indicators = [i for i in mlb.classes_ if i not in times + categories]
    
    x = set()
    for i in flattened_pred:
        cat = [j for j in i if j in categories]
        ind = [j for j in i if j in indicators]
        tim = [j for j in i if j in times]
        for m in cat:
            if m !='O':
                for n in ind:
                    for l in tim:
                        tag = 'I.' + m + '.' + n + '.' + l
                        if tag in all_gold_label:
                            x.add(tag)
                            
    return x

# function help convert time_flattened labels back to gold labels
def get_time_flattened_reverted(y_pred, mlb, all_gold_label):
    
    time_flattened_pred = mlb.inverse_transform(y_pred>0.5)
    cat_ind_lst = ['CAD.event', 'CAD.mention', 'CAD.symptom', 'CAD.test',
                   'DIABETES.A1C', 'DIABETES.glucose', 'DIABETES.mention',
                   'FAMILY_HIST.present', 'HYPERLIPIDEMIA.high_LDL',
                   'HYPERLIPIDEMIA.high_chol', 'HYPERLIPIDEMIA.mention',
                   'HYPERTENSION.high_bp', 'HYPERTENSION.mention',
                   'MEDICATION.ACE_inhibitor', 'MEDICATION.ARB',
                   'MEDICATION.DPP4_inhibitors', 'MEDICATION.anti_diabetes',
                   'MEDICATION.aspirin', 'MEDICATION.beta_blocker',
                   'MEDICATION.calcium_channel_blocker', 'MEDICATION.diuretic',
                   'MEDICATION.ezetimibe', 'MEDICATION.fibrate', 'MEDICATION.insulin',
                   'MEDICATION.metformin', 'MEDICATION.niacin', 'MEDICATION.nitrate',
                   'MEDICATION.statin', 'MEDICATION.sulfonylureas',
                   'MEDICATION.thiazolidinedione', 'MEDICATION.thienopyridine',
                   'OBESE.BMI', 'OBESE.mention', 'SMOKER.current', 'SMOKER.ever',
                   'SMOKER.never', 'SMOKER.past', 'SMOKER.unknown']
    times = ['after_DCT', 'before_DCT', 'during_DCT', 'NA']
    
    x = set()
    for i in time_flattened_pred:
        cat_ind = [j for j in i if j in cat_ind_lst]
        tim = [j for j in i if j in times]
        for m in cat_ind:
            for n in tim:
                tag = 'I.' + m + '.' + n
                if tag in all_gold_label:
                    x.add(tag)

    return x

# prepare features
def get_features(max_features, notes_train, notes_test, verbose=1):
    
    if verbose != 0: print('preparing features ...')
    notes = notes_train + notes_test
    X_txt = [' '.join(i) for i in notes]
    X_train_txt = [' '.join(i) for i in notes_train]
    X_test_txt = [' '.join(i) for i in notes_test]
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(X_txt)
    X_seq = tokenizer.texts_to_sequences(X_txt) 
    X_train_seq = tokenizer.texts_to_sequences(X_train_txt) 
    X_test_seq = tokenizer.texts_to_sequences(X_test_txt)
    word_index = tokenizer.word_index
    
    return X_train_seq, X_test_seq, word_index
    
# prepare targets
def get_targets(labels_train, labels_test, category=None, verbose=1):    
    if verbose != 0: print('preparing targets ...')
    labels = labels_train + labels_test
       
    if category == 'cat_only':
        # prepare cagtegory label targets
        labels = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels]
        labels_train = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels_train]
        labels_test = [[set([get_cat_labels(i) for i in list(j)]) for j in k] for k in labels_test]
    elif category == 'cat_ind':
        # prepare cagtegory indicator label targets
        labels = [[set([get_cat_ind_labels(i) for i in list(j)]) for j in k] for k in labels]
        labels_train = [[set([get_cat_ind_labels(i) for i in list(j)]) for j in k] for k in labels_train]
        labels_test = [[set([get_cat_ind_labels(i) for i in list(j)]) for j in k] for k in labels_test]
    elif category == 'flattened':    
        labels = [[set([m for n in [i.split('.') for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels]
        labels_train = [[set([m for n in [i.split('.') for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels_train]
        labels_test = [[set([m for n in [i.split('.') for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels_test]
    elif category == 'time_flattened':   
        labels = [[set([m for n in [get_time_flattened_label(i) for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels]
        labels_train = [[set([m for n in [get_time_flattened_label(i) for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels_train]
        labels_test = [[set([m for n in [get_time_flattened_label(i) for i in list(j)] for m in n if m != 'I']) for j in k] for k in labels_test]
    else:
        pass
    
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
    return Y_train, Y_test, mlb, num_labels

# prepare gold label targets
def get_gold_label_targets(Y_pred, gold_labels, gold_labels_test, mlb, category=None, verbose=1):
    if verbose != 0: print('preparing gold label targets ...')
    if category == 'cat_only':
        gold_labels = [{get_cat_labels(val) for val in gold_label} for gold_label in gold_labels]
        gold_labels_test = [{get_cat_labels(val) for val in gold_label} for gold_label in gold_labels_test]
        gold_labels_pred = [{i for s in mlb.inverse_transform(y_pred>0.5) for i in s if i != 'O'} for y_pred in Y_pred]
    elif category == 'cat_ind':
        gold_labels = [{get_cat_ind_labels(val) for val in gold_label} for gold_label in gold_labels]
        gold_labels_test =[{get_cat_ind_labels(val) for val in gold_label} for gold_label in gold_labels_test]
        gold_labels_pred = [{i for s in mlb.inverse_transform(y_pred>0.5) for i in s if i != 'O'} for y_pred in Y_pred]
    elif category == 'flattened':
        all_gold_label = {i for gold_label in gold_labels for i in gold_label}
        gold_labels_pred = [get_flattened_reverted(i, mlb, all_gold_label) for i in Y_pred]    
    elif category == 'time_flattened':
        all_gold_label = {i for gold_label in gold_labels for i in gold_label}
        gold_labels_pred = [get_time_flattened_reverted(i, mlb, all_gold_label) for i in Y_pred] 
    else:
        gold_labels_pred = [{i for s in mlb.inverse_transform(y_pred>0.5) for i in s if i != 'O'} for y_pred in Y_pred]
        
    gmlb = MultiLabelBinarizer()
    gmlb.fit(gold_labels)
    Y_gold_test = gmlb.transform(gold_labels_test)
    Y_gold_pred = gmlb.transform(gold_labels_pred)
    return Y_gold_test, Y_gold_pred, gmlb

# data generator function
def data_generator(X_seq, Y):
    while True:
        for x, y in zip(X_seq, Y):
            x = np.array(x).reshape((1,-1))
            y = np.array(y).reshape((1,-1, y.shape[1]))
            yield x, y
    
######### for embedding matrix #########
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_embedding_matrix(embedding_index, word_index, max_features, embed_size):
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: 
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i-1] = embedding_vector
    return embedding_matrix

######### for multiple comfusion matrix #########
def _check_targets(y_true, y_pred):
    
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred


def multilabel_confusion_matrix(y_true, y_pred, sample_weight=None, labels=None, samplewise=False):
    
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError("Samplewise metrics are not available outside of "
                             "multilabel classification.")

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
                                 minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight,
                                   minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight,
                                   minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 1 if samplewise else 0

        # All labels are index integers for multilabel.
        # Select labels:
        if not np.array_equal(labels, present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d > %d' %
                                 (np.max(labels), np.max(present_labels)))
            if np.min(labels) < 0:
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d < 0' % np.min(labels))

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = count_nonzero(true_and_pred, axis=sum_axis,
                               sample_weight=sample_weight)
        pred_sum = count_nonzero(y_pred, axis=sum_axis,
                                 sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis,
                                 sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)