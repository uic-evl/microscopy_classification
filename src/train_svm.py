from argparse import ArgumentParser
from pathlib import Path
from sklearn.svm import SVC
# from thundersvm import SVC
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from os import listdir
import joblib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import yaml
# import tensorflow as tf

CLASSES = np.array(['DMEL', 'DMFL', 'DMLI', 'DMTR'])

# TODO: Try linear classifier for SVC


def get_consecutive(path):
    f = [x for x in listdir(path) if (path / x).is_dir() and 'SVM' in x]
    return len(f) + 1


def load_values(path, filename):
    d = np.load(path / filename)
    return d.f.arr_0, d.f.arr_1, d.f.arr_2


def save_model(model, path):
    filepath = path / 'model.pkl'
    joblib.dump(model, filepath, compress=4)


# plot function from sklearn example
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          output=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.savefig(str(output / 'confusion.png'))
    return ax


def save_opts(path, val_acc, bb, ds, f1, weigths, kernel, pca, n_components):
    opts = {
        'kernel': kernel,
        'val_acc': str(val_acc),
        'backbone': bb,
        'dataset': ds,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'f1': str(f1),
        'weights': str(weigths),
        'classes': str(CLASSES),
        'pca': pca,
        'n_components': str(n_components)
    }

    filepath = path / 'opts.yml'
    with open(str(filepath), 'w') as outfile:
        yaml.dump(opts, outfile)


def save_mistakes(path, mistakes):
    filename = path / 'val_mistakes.npz'
    np.savez(filename, mistakes)


def main():
    parser = ArgumentParser()
    parser.add_argument('-path', type=str, help='features location')
    parser.add_argument('-ds', type=str, help='dataset name')
    parser.add_argument('-aug', type=str, help='augmentation')
    parser.add_argument('-bb', type=str, default='resnet50', help='random s eed')
    parser.add_argument('-out', type=str, help='output path')
    parser.add_argument('-w', type=str, default='true', help='use weights')
    parser.add_argument('-k', type=str, default='rbf', help='SVM kernel: rbf, linear, poly or sigmoid')
    parser.add_argument('-pca', type=str, default='false', help='Apply PCA reduction')
    parser.add_argument('-n_components', type=float, default=1024.0, help='PCA components')
    args = parser.parse_args()

    args.out = Path(args.out)
    args.path = Path(args.path)

    if args.aug == 'true':
        aug = 'aug'
    else:
        aug = 'noaug'
    train_feat_filename = '%s_%s_%s_train.npz' % (args.ds, args.bb, aug)
    val_feat_filename = '%s_%s_noaug_validation.npz' % (args.ds, args.bb)

    # create the export path
    output_path = Path(args.out)
    output_path.mkdir(exist_ok=True)

    # create a directory for the experiment artifacts
    exp_code = 'SVM_%d' % get_consecutive(args.out)
    exp_path = output_path / exp_code
    exp_path.mkdir(exist_ok=True)

    # load extracted features
    train_features, train_labels, train_filenames = \
        load_values(args.path, train_feat_filename)
    val_features, val_labels, val_filenames = \
        load_values(args.path, val_feat_filename)

    # train the model
    train_x = [f.ravel() for f in train_features]
    train_y = [np.argmax(l) for l in train_labels]
    val_x = [f.ravel() for f in val_features]
    val_y = [np.argmax(l) for l in val_labels]
    
    # reduce features?        
    if args.pca == 'true':
        if args.n_components >= 1:
            n_components = int(args.n_components)
            pca = PCA(n_components=n_components)
        else:
            # produce PCA that explains n_components % of the variance
            pca = PCA(n_components=args.n_components, svd_solver='full')
                
        pca.fit(train_x)
        print(pca.explained_variance_ratio_.cumsum())
        
        train_x = pca.transform(train_x)
        val_x = pca.transform(val_x)
        # save pca means for testing
        np.save(exp_path / 'pca_means.npy', pca.mean_)
    
    # use weights?
    class_weights = 'balanced'
    if args.w == 'true':
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(train_y),
                                                          train_y)
        class_weights = {idx: class_weights[idx] for idx, d in enumerate(class_weights)}

    svm = SVC(kernel=args.k, class_weight=class_weights)
    svm.fit(train_x, train_y)
    val_y_pred = svm.predict(val_x)

    mistakes = []
    for i in range(len(val_y)):
        if val_y[i] != val_y_pred[i]:
            mistakes.append([val_filenames[i], val_y[i], val_y_pred[i]])

    # create artifacts
    save_model(svm, exp_path)
    val_acc = svm.score(val_x, val_y)
    f1 = f1_score(val_y, val_y_pred,  average='weighted')
    plot_confusion_matrix(val_y, val_y_pred, classes=CLASSES, output=exp_path)
    save_opts(exp_path, val_acc, args.bb, args.ds, f1, class_weights, args.k, args.pca, args.n_components)
    save_mistakes(exp_path, mistakes)
    print("Finished. Results saved at %s" % str(exp_path))

if __name__ == "__main__":
    main()
