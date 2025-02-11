# DB_Finance

**제 15회 DB보험금융공모전**  
> **기간:** 2025.02.01(토) ~ 2025.02.28(금) 오후 3시

---

## 팀명: DeepLife
**주제:** Survival Analysis via Deep Learning

### 팀 구성
- **팀장**: 서울대학교 통계학과 석사과정 이성은 
- **팀원**: 서울대학교 통계학과 석사과정 권헌정  

---

## Repository Structure

1. **Data**  
   - 원본 데이터(`fraud_oracle.csv`)와 전처리된 데이터를 포함
   
2. **fraud_preprocessing.ipynb**  
   - `Data/fraud_oracle.csv` 파일의 전처리 과정  
   - [📊 Kaggle](https://www.kaggle.com/code/hobeomlee/fraud-detection-modeling-guidebook-a-to-z)

3. **ignnet.py**  
   - Interpretable Graph Neural Networks for Tabular Data Architecture  
   - [📄 arXiv](https://arxiv.org/abs/2308.08945)

4. **runner.py**  
   - 프로젝트 실행을 위한 파이썬 스크립트

---

## 데이터 설명
**Vehicle Insurance Fraud Detection**  
차량 보험 사기(Fraud) 사례를 판단하기 위한 데이터로, 사고 정보와 보험 정책 정보가 포함되어 있습니다.  
- **주요 특징**: 차량 속성, 모델 정보, 사고 발생 시점 및 피해 규모, 보험 가입 정보 등  
- **타겟**: `FraudFound_P` (사기 의심 여부)  
- **데이터 크기**: (15420, 33)

---

## 모델 설명 (IGNNet)
IGNNet(Interpretable Graph Neural Network)은 표 형식(Tabular) 데이터를 그래프 구조로 해석하여, GNN을 통해 변수 간 복잡한 상호작용을 학습하는 모델입니다.  
- **해석 가능성**: 모델 내부 연산 과정을 명시적으로 확인 가능  

---

## 실행 방법
```bash
python runner.py