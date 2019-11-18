import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import LabelEncoder

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


def plot_confusion_matrix(cm, labels=None, true_labels=None,
                          pred_labels=None, title=None, normalize=False,
                          hide_zeros=False, x_tick_rotation=0, ax=None,
                          figsize=None, cmap='Blues', title_fontsize="large",
                          text_fontsize="medium"):
    """
    shamelessly borrow from scikit-plot
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if labels is None:
        classes = ['-','+']
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.in1d(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.in1d(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    plt.colorbar(mappable=image)
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize=text_fontsize,
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize=text_fontsize)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if not (hide_zeros and cm[i, j] == 0):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=text_fontsize,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=text_fontsize)
    ax.set_xlabel('Predicted label', fontsize=text_fontsize)
    ax.grid('off')

    return ax




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