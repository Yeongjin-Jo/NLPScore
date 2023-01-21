import argparse
import numpy as np
import pandas as pd

from PyKomoran import *
from hanspell import spell_checker
from nltk.translate.bleu_score import sentence_bleu

def correct_sentence(text):
    spelled_text = [spell_checker.check(i) for i in text]
    handspell_text = [i.checked for i in spelled_text]
    return handspell_text

# 일반명사,고유명사,수사,대명사,동사,형용사,관형사,어간,외국어,한자,숫자
# 구두문자같은건 include에서 전부 제외됨
# include는 언제든지 수정가능 
def normalize_text(text):
    include = ['NNG','NNP','NR','NP','VV','VA','MM','XR','SL','SH','SN'] # IC:감탄사
    komoran = Komoran("STABLE")
    pos_list = [komoran.get_plain_text(i) for i in text]
    text = []
    for i in pos_list:
        temp = ""
        for j in i.split():
            if j.split('/')[1] in include:
                temp = temp + " " + j.split('/')[0]
                temp = temp.strip()
        text.append(temp)
    return text

# EM : Exact_match : 모든 형태소가 똑같아야함. 맞으면 1 틀리면 바로 0
def compute_exact_match(prediction, truth):
    assert len(prediction) == len(truth)    
    EMScore = sum([prediction[i] == truth[i] for i in range(len(prediction))])/len(prediction)
    return EMScore

def compute_f1(prediction, truth):
    F1Score = 0
    for i in range(len(prediction)):
        pred_tokens = prediction[i].split()
        truth_tokens = truth[i].split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        
        F1Score += 2 * (prec * rec) / (prec + rec)
    return F1Score / len(prediction)

# BLEU Score : 1,2,3,4-gram의 기하평균 * 길이 패널티(bp)
def compute_bleu(prediction, truth, weights:list):
    assert len(weights) == 4, "weights를 4개 입력해주세요."
    assert sum(weights) == 1, "weights의 합이 1이 되도록 설정해주세요."
    BLEUScore = 0
    for i in range(len(prediction)):
        BLEUScore += sentence_bleu([truth[i].split()],prediction[i].split(), weights=weights)
    return BLEUScore / len(prediction)

def main():
    args = parser.parse_args()
    data = pd.read_csv(args.data,encoding='cp949')
    Prediction = normalize_text(correct_sentence(data['Prediction']))
    Answer = normalize_text(correct_sentence(data['Answer']))
    
    with open('Score.txt', 'w') as f:
        f.write(f'Avg EM score : {compute_exact_match(Prediction, Answer)}\n')
        f.write(f'Avg F1 score : {compute_f1(Prediction, Answer)}\n')
        f.write(f'Avg BLEU score : {compute_bleu(Prediction, Answer, weights=args.weights)}\n')

parser = argparse.ArgumentParser(description='NLPScoreCompute')
parser.add_argument('--data', required=True, help='csv file')
parser.add_argument('--weights', required=True, help='n-gram weights, must be 1 when added, must be 4 weight', type=float, nargs='+')

if __name__ == "__main__":
    main()
    