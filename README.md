# Tick Trader - 선물옵션 틱데이터 기반 AI 트레이딩 시스템

방대한 선물옵션 틱데이터를 분석하여 머신러닝 및 딥러닝 기법으로 가격 상승/하락 패턴을 예측하는 프로젝트입니다. 하루 수십만 건의 틱데이터에서 의미 있는 패턴을 발견하고 예측 모델을 구축하는 것을 목표로 합니다.

## 🎯 프로젝트 목표

- 호가창 데이터(가격, 수량, 미결제수량, 거래량)의 패턴 분석
- 머신러닝/딥러닝을 통한 상승/하락 예측 모델 개발
- 실시간 트레이딩 시스템 구현을 위한 기반 마련

## 📊 데이터 구조

### 호가창 데이터
- **매수/매도 호가**: 호가가격, 수량, 미결제 수량
- **체결 데이터**: 가격, 수량 등

### 데이터 특징
- 수개월간의 틱데이터 보유
- 일일 수십만 건의 방대한 데이터 양
- 시간 순서에 따른 고차원 시계열 데이터

## 🛠️ 기술 접근법

### 1. 데이터 전처리 및 특징 엔지니어링

#### 시계열 특징 추출
- 호가 불균형 지표
- 호가 압력 지표
- 미결제약정 변화율
- VWAP (Volume Weighted Average Price)
- 체결강도 및 스프레드 변화

#### 마이크로스트럭처 분석
- 호가창 깊이 분석
- 플로우 톡시시티(Flow Toxicity)
- VPIN (Volume-synchronized Probability of Informed Trading)

### 2. 머신러닝 모델 아키텍처

#### 기본 모델
- **LSTM 기반 시계열 분류**
- **앙상블 접근법** (LightGBM + XGBoost + RandomForest)
- **CNN-LSTM 하이브리드**

#### 고급 모델
- **Transformer 기반 어텐션 메커니즘**
- **Graph Neural Networks** (호가창 그래프 구조 모델링)
- **Neural ODE** (연속 시간 동역학 모델링)

### 3. 혁신적인 접근법

#### 이벤트 기반 모델링
- **Hawkes Process** (연쇄적 주문 패턴 포착)
- **Causal Discovery** (인과관계 그래프 학습)

#### 위상 데이터 분석 (TDA)
- Persistent Homology를 활용한 위상학적 특징 추출
- 호가창의 고차원 패턴 분석

#### 메타러닝
- **MAML** (Model-Agnostic Meta-Learning)
- 시장 체제 변화에 대한 빠른 적응

#### 시장 체제 탐지
- Hidden Markov Model을 활용한 시장 상태 추론
- Change Point Detection으로 체제 변화 감지

## 🔧 구현 전략

### 데이터 파이프라인
- Apache Parquet을 활용한 효율적 데이터 저장
- 윈도우 슬라이딩 기반 학습 데이터 생성
- Redis를 활용한 실시간 특징 캐싱

### 모델 최적화
- 클래스 불균형 처리 (SMOTE, Focal Loss)
- 과적합 방지 (Walk-Forward Analysis, Purged Cross-Validation)
- 액티브 러닝을 통한 효율적 레이블링

### 다중 스케일 분석
- Wavelet Transform을 활용한 다중 시간 스케일 분석
- Multifractal Analysis로 프랙탈 특성 추출
- Detrended Fluctuation Analysis (DFA)

## 📈 평가 지표

- **Sharpe Ratio**: 리스크 대비 수익률
- **Maximum Drawdown**: 최대 손실폭
- **분류 성능**: 정확도, 정밀도, 재현율
- **PnL Curve**: 실제 수익 곡선

## 🚀 개발 로드맵

### Phase 1: 기반 구축
- [ ] 데이터 전처리 파이프라인 구축
- [ ] 기본 특징 엔지니어링 구현
- [ ] 베이스라인 모델 (RandomForest, XGBoost) 개발

### Phase 2: 심화 모델링
- [ ] LSTM/Transformer 기반 딥러닝 모델 구현
- [ ] 어텐션 메커니즘 적용
- [ ] 앙상블 모델 개발

### Phase 3: 고급 기법
- [ ] Graph Neural Networks 구현
- [ ] Topological Data Analysis 적용
- [ ] Meta-Learning 파이프라인 구축

### Phase 4: 최적화
- [ ] 실시간 예측 시스템 구현
- [ ] 성능 최적화 및 튜닝
- [ ] 백테스팅 및 검증

## 📋 요구사항

### Python 라이브러리
```
pandas, numpy
scikit-learn
tensorflow/pytorch
xgboost, lightgbm
pyarrow (Parquet)
redis
```

### 고급 라이브러리 (선택적)
```
torch-geometric (GNN)
tick (Hawkes Process)
ripser, persim (TDA)
tensorly (Tensor Networks)
learn2learn (Meta-Learning)
```

## 💡 핵심 특징

1. **해석가능성**: Attention weights, 인과 그래프로 모델 결정 과정 이해
2. **적응성**: Meta-learning으로 시장 변화에 빠른 대응
3. **노이즈 강건성**: VAE, Kalman filter로 노이즈 제거
4. **다중 스케일**: Wavelet, fractal 분석으로 다양한 시간 패턴 포착
5. **효율성**: Active learning으로 레이블링 비용 절감

---

이 프로젝트는 금융 데이터 분석과 최신 머신러닝 기법을 결합하여 틱데이터 속에 숨겨진 패턴을 발견하고, 이를 바탕으로 예측 정확도를 높이는 것을 목표로 합니다.