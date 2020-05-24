
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier

from preprocessing import *

from skimage.transform import resize
import cv2
import os

def load_image_files(container_path, dimension=(128, 128)):
  """
  Load image files with categories as subfolder names 
  which performs like scikit-learn sample dataset
  
  Parameters
  ----------
  container_path : string or unicode
      Path to the main folder holding one subfolder per category
  dimension : tuple
      size to which image are adjusted to
      
  Returns
  -------
  Bunch
  """
  image_dir = Path(container_path)
  folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
  categories = [fo.name for fo in folders]

  descr = "A image classification dataset"
  images = []
  flat_data = []
  target = []
  for i, direc in enumerate(folders):
    print("Folder: ", direc)
    n = len(os.listdir(direc))
    it = 0
    for file in direc.iterdir():
      if it%100==0:
        print("{} of {} done...".format(it, n))
      it+=1
      img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
      dst = canny(img, 0.4)
      # img_resized = img
      img_resized = resize(dst, dimension, anti_aliasing=False, mode='reflect')
      flat_data.append(img_resized.flatten()) 
      images.append(img_resized)
      target.append(i)
    print("{} of {} done...".format(n, n))
  flat_data = np.array(flat_data)
  target = np.array(target)
  images = np.array(images)

  return Bunch(data=flat_data,
                target=target,
                target_names=categories,
                images=images,
                DESCR=descr)

image_dataset = load_image_files("./chest-xray-pneumonia/chest_xray/train")
X_train, _, y_train, _ = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.0001, random_state=42)

image_test_dataset = load_image_files("./chest-xray-pneumonia/chest_xray/test")
X_test, _, y_test, _ = train_test_split(
    image_test_dataset.data, image_test_dataset.target, test_size=0.0001, random_state=42)

print(X_train.shape, y_train.shape)

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
clf = RandomForestClassifier()
# clf = GridSearchCV(svc, param_grid)

print("Training...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))



"""
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.97      0.41      0.58       234
           1       0.74      0.99      0.85       389

    accuracy                           0.77       623
   macro avg       0.85      0.70      0.71       623
weighted avg       0.82      0.77      0.74       623


Erosion: No sub
              precision    recall  f1-score   support

           0       0.94      0.32      0.47       234
           1       0.71      0.99      0.82       389

    accuracy                           0.74       623
   macro avg       0.82      0.65      0.65       623
weighted avg       0.79      0.74      0.69       623

Erosion: (Sub)
              precision    recall  f1-score   support

           0       0.81      0.18      0.29       234
           1       0.66      0.97      0.79       389

    accuracy                           0.68       623
   macro avg       0.74      0.58      0.54       623
weighted avg       0.72      0.68      0.60       623

Dilate: (Subtraction)
// Skipped

Closing - Open (20):
              precision    recall  f1-score   support

           0       0.86      0.14      0.24       234
           1       0.66      0.99      0.79       389

    accuracy                           0.67       623
   macro avg       0.76      0.56      0.51       623
weighted avg       0.73      0.67      0.58       623


Global Eq:
              precision    recall  f1-score   support

           0       0.92      0.28      0.43       234
           1       0.70      0.98      0.81       389

    accuracy                           0.72       623
   macro avg       0.81      0.63      0.62       623
weighted avg       0.78      0.72      0.67       623

Local Eq:
// Skipped

Adaptive:
              precision    recall  f1-score   support

           0       0.92      0.20      0.32       234
           1       0.67      0.99      0.80       389

    accuracy                           0.69       623
   macro avg       0.80      0.59      0.56       623
weighted avg       0.77      0.69      0.62       623

CS:
              precision    recall  f1-score   support

           0       0.92      0.35      0.51       234
           1       0.72      0.98      0.83       389

    accuracy                           0.75       623
   macro avg       0.82      0.67      0.67       623
weighted avg       0.79      0.75      0.71       623

Canny:
              precision    recall  f1-score   support

           0       0.68      0.28      0.39       234
           1       0.68      0.92      0.78       389

    accuracy                           0.68       623
   macro avg       0.68      0.60      0.59       623
weighted avg       0.68      0.68      0.64       623
"""