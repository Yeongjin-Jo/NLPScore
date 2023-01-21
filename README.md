# Korean NLP Score 계산

## 특징

- 한글 전처리 도구로 Pykomoran 사용 (konlpy의 komoran보다 정확)
- 일반명사,고유명사,수사,대명사,동사,형용사,관형사,어간,외국어,한자,숫자만 추출
- 여러개의 쌍으로 된 문장들 CSV 파일을 처리하여 평균 EM, F1, BLEU를 계산


## 사용법 
### NLPScore.py 실행
- csv data 경로를 입력해준다.
- BLEU Score에서 n-gram(1~4 gram)의 weight를 정해준다.(논문에서는 0.25이나 문장의 길이가 짧은 편이라면 4,3,2 gram 순으로 가중치를 내리면 된다.)
```sh- 
python NLPScore.py  --data NLPScoreTest.csv
                    --weights 0.25 0.25 0.25 0.25
```
- Score.txt에서 결과 확인 가능


