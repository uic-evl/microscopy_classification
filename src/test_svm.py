from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, f1_score
from joblib import load
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

CLASSES = np.array(['DMEL', 'DMFL', 'DMLI', 'DMTR'])

def load_values(path, filename):
    d = np.load(path / filename)
    return d.f.arr_0, d.f.arr_1, d.f.arr_2

def main():
    parser = ArgumentParser()
    parser.add_argument('-feats_path', type=str, help='test features folder')
    parser.add_argument('-model_path', type=str, help='model folder')
    parser.add_argument('-ds', type=str, help='dataset name')
    parser.add_argument('-bb', type=str, default='resnet50', help='random seed')
    parser.add_argument('-pca', type=str, default='false', help='Apply PCA reduction')
    parser.add_argument('-n_components', type=float, default=1024, help='PCA components')
    parser.add_argument('-aug', default='false', type=str, help='augmentation')
    
    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    args.feats_path = Path(args.feats_path)
    
    if args.aug == 'true':
        aug = 'aug'
    else:
        aug = 'noaug'
                        
    svm = load(args.model_path / 'model.pkl')

    test_feat_filename = '%s_%s_noaug_test.npz' % (args.ds, args.bb)
    test_features, test_labels, test_filenames = \
        load_values(args.feats_path, test_feat_filename)        
    test_x = [f.ravel() for f in test_features]
    test_y = [np.argmax(l) for l in test_labels]
    
    if args.pca == 'true':
        if args.n_components >= 1:
            n_components = int(args.n_components)
            pca = PCA(n_components=n_components)
        else:
            # produce PCA that explains n_components % of the variance
            pca = PCA(n_components=args.n_components, svd_solver='full')
                
        train_feat_filename = '%s_%s_%s_train.npz' % (args.ds, args.bb, aug)
        train_features, train_labels, train_filenames = \
            load_values(args.feats_path, train_feat_filename)
        train_x = [f.ravel() for f in train_features]
        train_y = [np.argmax(l) for l in train_labels]
                
        pca.fit(train_x)        
        test_x = pca.transform(test_x)      
    
    test_y_pred = svm.predict(test_x)
    f1 = f1_score(test_y, test_y_pred,  average='weighted')
    print(f1)
    print(classification_report(test_y, test_y_pred, target_names=CLASSES, digits=4))

if __name__ == "__main__":
    main()