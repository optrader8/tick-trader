# AI Trading Platform - 선물옵션 틱데이터 기반 AI 트레이딩 플랫폼

종합적인 AI 트레이딩 플랫폼으로, 방대한 선물옵션 틱데이터를 분석하여 머신러닝 및 딥러닝 기법으로 가격 상승/하락 패턴을 예측합니다. 모노레포 구조로 개발되어 AI 엔진, API 백엔드, 웹 프론트엔드를 통합적으로 제공합니다.

## 🏗️ 모노레포 구조

```
tick-trader/
├── aiend/                     # AI 엔진 (Python ML)
│   ├── src/                   # 머신러닝 소스 코드
│   ├── config/                # 설정 파일
│   ├── data/                  # 데이터 디렉토리
│   ├── tests/                 # 테스트 스위트
│   └── requirements.txt       # Python 의존성
├── backend/                   # API 백엔드 (Node.js)
├── frontend/                  # 웹 프론트엔드 (React TypeScript)
└── README.md                 # 메인 프로젝트 문서
```

## 🎯 핵심 기능

### 🤖 AI 엔진 (aiend)
- **고급 머신러닝 모델**: LSTM, Transformer, CNN-LSTM, 앙상블
- **실시간 예측**: 틱데이터 기반 실시간 트레이딩 시그널 생성
- **특징 엔지니어링**: 호가창 분석, 시계열 특징 추출
- **백테스팅**: 포괄적인 거래 시뮬레이션 및 성능 평가

### 🔧 API 백엔드 (backend)
- **RESTful API**: 표준화된 API 엔드포인트 제공
- **실시간 통신**: WebSocket을 통한 실시간 데이터 전송
- **데이터베이스**: 효율적인 데이터 저장 및 관리
- **인증/인가**: 보안된 사용자 관리 시스템

### 🎨 웹 프론트엔드 (frontend)
- **트레이딩 대시보드**: 실시간 시장 데이터 및 포트폴리오 관리
- **차트 및 시각화**: 고급 차트 라이브러리를 통한 데이터 시각화
- **모델 성능 모니터링**: AI 모델 성능 실시간 추적
- **사용자 인터페이스**: 직관적인 트레이딩 운영 환경

## 🚀 빠른 시작

### AI 엔진 개발
```bash
cd aiend
pip install -r requirements.txt
pip install -e .

# 모델 훈련
python -m src.training.trainer --model lstm --config config/config.yaml

# 실시간 예측
python -m src.prediction.realtime --config config/config.yaml
```

### 백엔드 개발
```bash
cd backend
npm install
npm run dev
```

### 프론트엔드 개발
```bash
cd frontend
npm install
npm start
```

## 📊 기술 스택

### AI 엔진
- **Python 3.8+**: 머신러닝 개발 언어
- **TensorFlow/PyTorch**: 딥러닝 프레임워크
- **Scikit-learn**: 머신러닝 라이브러리
- **Pandas/NumPy**: 데이터 처리
- **Apache Arrow**: 효율적 데이터 저장
- **Redis**: 실시간 캐싱

### 백엔드
- **Node.js**: 서버 사이드 자바스크립트
- **Express/Fastify**: 웹 프레임워크
- **TypeScript**: 타입 안전성
- **PostgreSQL**: 관계형 데이터베이스
- **WebSocket**: 실시간 통신

### 프론트엔드
- **React 18**: 사용자 인터페이스 라이브러리
- **TypeScript**: 타입 안전성
- **Tailwind CSS**: 스타일링 프레임워크
- **Chart.js/D3.js**: 데이터 시각화
- **Vite**: 빠른 빌드 도구

## 🔄 개발 워크플로우

1. **AI 엔진**: 머신러닝 모델 개발 및 훈련
2. **API 백엔드**: 모델 서빙 및 데이터 관리
3. **웹 프론트엔드**: 사용자 인터페이스 및 시각화
4. **통합 테스트**: 전체 시스템 통합 및 테스트
5. **배포**: 프로덕션 환경 배포

## 📈 개발 로드맵

### Phase 1: 기반 구축
- [x] AI 엔진 핵심 기능 구현
- [ ] API 백엔드 기본 구조 완성
- [ ] 프론트엔드 기본 UI 구현

### Phase 2: 통합 개발
- [ ] AI 엔진과 백엔드 연동
- [ ] 실시간 데이터 파이프라인 구축
- [ ] 웹 기반 트레이딩 대시보드 개발

### Phase 3: 고급 기능
- [ ] 다중 전략 트레이딩 시스템
- [ ] 포트폴리오 관리 기능
- [ ] 리스크 관리 시스템

### Phase 4: 최적화 및 배포
- [ ] 성능 최적화
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포

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