# -*- coding: UTF-8 -*-
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score
from dnlp.utils.constant import TAG_BEGIN, TAG_INSIDE, TAG_END, TAG_SINGLE


def get_cws_statistics(correct_labels, predict_labels) -> (int, int, int):
  if len(correct_labels) != len(predict_labels):
    raise Exception('length of correct labels and predict labels is not equal')

  true_positive_count = 0
  corrects = {}
  predicts = {}
  correct_start = 0
  predict_start = 0

  for i, (correct_label, predict_label) in enumerate(zip(correct_labels, predict_labels)):
    if correct_label == TAG_BEGIN:
      correct_start = i
      corrects[correct_start] = correct_start
    elif correct_label == TAG_SINGLE:
      correct_start = i
      corrects[correct_start] = correct_start
    elif correct_label == TAG_INSIDE or correct_label == TAG_END:
      corrects[correct_start] = i

    if predict_label == TAG_BEGIN:
      predict_start = i
      predicts[predict_start] = predict_start
    elif predict_label == TAG_SINGLE:
      predict_start = i
      predicts[predict_start] = predict_start
    elif predict_label == TAG_INSIDE or predict_label == TAG_END:
      predicts[predict_start] = i

  for predict in predicts:
    if predict in corrects and corrects[predict] == predicts[predict]:
      true_positive_count += 1

  return true_positive_count, len(predicts), len(corrects)


def get_ner_statistics(correct_labels, predict_labels) -> (int, int, int):
  if len(correct_labels) != len(predict_labels):
    raise Exception('length of correct labels and predict labels is not equal')

  true_positive_count = 0
  corrects = {}
  predicts = {}
  correct_start = 0
  predict_start = 0

  for i, (correct_label, predict_label) in enumerate(zip(correct_labels, predict_labels)):
    if correct_label == TAG_BEGIN:
      correct_start = i
      corrects[correct_start] = correct_start
    elif correct_label == TAG_INSIDE:
      corrects[correct_start] = i

    if predict_label == TAG_BEGIN:
      predict_start = i
      predicts[predict_start] = predict_start
    elif predict_label == TAG_INSIDE:
      predicts[predict_start] = i

  for predict in predicts:
    if corrects.get(predict) is not None and corrects[predict] == predicts[predict]:
      true_positive_count += 1

  return true_positive_count, len(predicts), len(corrects)


def evaluate_cws(model, data_path: str):
  with open(data_path, 'rb') as f:
    data = pickle.load(f)
    characters = data['characters']
    labels_true = data['labels']
    c_count = 0
    p_count = 0
    r_count = 0

    all_labels_true = []
    all_labels_predict = []
    for sentence, label in zip(characters, labels_true):
      if len(sentence) <= 3:
        continue
      words, labels_predict = model.predict(sentence, return_labels=True)
      all_labels_predict.extend(labels_predict)
      all_labels_true.extend(label)
      c, p, r = get_cws_statistics(label, labels_predict)
      c_count += c
      p_count += p
      r_count += r
    print(c_count / p_count)
    print(c_count / r_count)
    average = 'macro'
    print(precision_score(all_labels_true,all_labels_predict,average=average))
    print(recall_score(all_labels_true,all_labels_predict,average=average))

def evaluate_ner(model, data_path:str):
  with open(data_path, 'rb') as f:
    data = pickle.load(f)
    characters = data['characters']
    labels_true = data['labels']
    c_count = 0
    p_count = 0
    r_count = 0

    all_labels_true = []
    all_labels_predict = []
    for sentence, label in zip(characters, labels_true):
      if len(sentence) <= 3:
        continue
      entities, labels_predict = model.predict_ll(sentence, return_labels=True)
      all_labels_predict.extend(labels_predict)
      all_labels_true.extend(label)
      c, p, r = get_ner_statistics(label, labels_predict)
      c_count += c
      p_count += p
      r_count += r
    p = c_count/p_count
    r = c_count/r_count
    f1 = 2*p*r/(p+r)
    print(p,r,f1)

    return fmt(p*100),fmt(r*100),fmt(f1*100)
    # average = 'macro'
    # print(precision_score(all_labels_true, all_labels_predict, average=average))
    # print(recall_score(all_labels_true, all_labels_predict, average=average))

def fmt(f):
  return '{0:.2f}'.format(f)