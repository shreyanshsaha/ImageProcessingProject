
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from skimage.transform import resize
import cv2
import os

from preprocessing import *

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

clf = LogisticRegression()

print("Training...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))



"""

No Preprocessing
Classification report for - 
GridSearchCV(cv=None, error_score=nan,
             estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='deprecated', n_jobs=None,
             param_grid=[{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                         {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                          'kernel': ['rbf']}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0):
              precision    recall  f1-score   support

           0       0.94      0.93      0.94       420
           1       0.97      0.98      0.98      1145

    accuracy                           0.97      1565
   macro avg       0.96      0.96      0.96      1565
weighted avg       0.97      0.97      0.97      1565

Classification report for - 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False):
              precision    recall  f1-score   support

           0       0.94      0.32      0.48       234
           1       0.71      0.99      0.82       389

    accuracy                           0.74       623
   macro avg       0.82      0.65      0.65       623
weighted avg       0.79      0.74      0.69       623

Dilate:(subtracting)
              precision    recall  f1-score   support

           0       0.91      0.44      0.59       234
           1       0.74      0.97      0.84       389

    accuracy                           0.77       623
   macro avg       0.83      0.71      0.72       623
weighted avg       0.81      0.77      0.75       623

Opening: (subtracting)
              precision    recall  f1-score   support

           0       0.93      0.34      0.50       234
           1       0.71      0.98      0.83       389

    accuracy                           0.74       623
   macro avg       0.82      0.66      0.66       623
weighted avg       0.79      0.74      0.70       623

Opening (10): (No subtraction) 
              precision    recall  f1-score   support

           0       0.95      0.30      0.45       234
           1       0.70      0.99      0.82       389

    accuracy                           0.73       623
   macro avg       0.82      0.64      0.64       623
weighted avg       0.79      0.73      0.68       623

Closing - Open (20):
              precision    recall  f1-score   support

           0       0.87      0.28      0.42       234
           1       0.69      0.97      0.81       389

    accuracy                           0.71       623
   macro avg       0.78      0.63      0.61       623
weighted avg       0.76      0.71      0.66       623

Gradient:
              precision    recall  f1-score   support

           0       0.89      0.34      0.49       234
           1       0.71      0.97      0.82       389

    accuracy                           0.74       623
   macro avg       0.80      0.66      0.66       623
weighted avg       0.78      0.74      0.70       623

Dilatoin: skimage subtraction
              precision    recall  f1-score   support

           0       0.94      0.31      0.46       234
           1       0.70      0.99      0.82       389

    accuracy                           0.73       623
   macro avg       0.82      0.65      0.64       623
weighted avg       0.79      0.73      0.69       623

Hist Eq - Global:
              precision    recall  f1-score   support

          0       0.94      0.32      0.48       234
          1       0.71      0.99      0.82       389

    accuracy                           0.74       623
   macro avg       0.82      0.65      0.65       623
weighted avg       0.79      0.74      0.69       623

Hist Eq - Local:
//skipped
Adaptive:
              precision    recall  f1-score   support

           0       0.97      0.32      0.49       234
           1       0.71      0.99      0.83       389

    accuracy                           0.74       623
   macro avg       0.84      0.66      0.66       623
weighted avg       0.81      0.74      0.70       623

CS:
              precision    recall  f1-score   support

           0       0.92      0.35      0.50       234
           1       0.71      0.98      0.83       389

    accuracy                           0.74       623
   macro avg       0.82      0.66      0.66       623
weighted avg       0.79      0.74      0.71       623

Canny:
              precision    recall  f1-score   support

           0       0.80      0.28      0.42       234
           1       0.69      0.96      0.80       389

    accuracy                           0.70       623
   macro avg       0.75      0.62      0.61       623
weighted avg       0.73      0.70      0.66       623



"""