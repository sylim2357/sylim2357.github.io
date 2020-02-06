---
title:  "Stock Selection via Nonlinear Multi-Factor Models"
excerpt: "Levin, Asriel E. \"Stock selection via nonlinear multi-factor models.\" Advances in Neural Information Processing Systems. 1996."
toc: true
toc_sticky: true

categories:
  - Paper Review
tags:
  - Machine Learning
  - Finance
  - Market
  - Trading
last_modified_at: 2020-02-07T06:05:00+09:00
---

논문 링크([NIPS](http://papers.nips.cc/paper/1114-stock-selection-via-nonlinear-multi-factor-models.pdf))

NIPS에 발표된 96년도 논문. 샌프란 Barclays에 있던 Asriel Uzi Levin이 발표한 논문. 2006년에 바클레이즈 나와서 펀드 하나 차린듯.

96년도에 나온 논문인데, 지금 내가 하고싶은거 그대로 했어서 조금 좌절함. 결국 팩터들와 수익률과의 비선형적인 관계를 찾고 싶다는건데.. 정리합시다.

# Introduction
보통 multifactor model이라고 하면 아래의 식을 가리킨다.

$$R_{i} = a_{i} + u_{i1}F_{1} + u_{i2}F_{2} + \cdots + u_{iL}F_{L} + e_{i}$$

$$R_{i}$$는 i종목의 수익률, $$F_{l}$$은 각각의 팩터값, $$u_{il}$$은 i종목이 l팩터에 노출된 risk정도, $$a_{i}$$는 알파, $$e_{i}$$는 idiosyncrasy를 뜻한다.
팩터값 $$F_{l}$$은 다른 팩터들 다 제끼고 이 특정 팩터의 위험에만 노출되었을 때에 얻을 수 있는 단위 초과기대수익을 뜻한다.  
팩터모델은 선형회귀이다. 각 F값을 독립변수로 두고 수익률을 종속변수로 둔 다음 회귀를 하면 구할 수 있다. 즉, 애초에 팩터모델은 수익률을 설명하는 모델이지, 미래의 수익률을 예측하는 모델로서 세상에 나온 것은 아니라는 것이다. 즉, 미래 수익률을 예측하기 위해서 위의 팩터모델을 쓰려면,

$$R_{i} = a_{i} + u_{i1}\hat{F_{1}} + u_{i2}\hat{F_{2}} + \cdots + u_{iL}\hat{F_{L}} + e_{i}$$

와 같이 $$F_{l}$$대신 해당 팩터의 예측값인 $$\hat{F_{l}}$$을 사용해야 하는 것이다. 이 예측은 간단히 과거 몇 분기동안의 값을 평균낼 수도 있고 시계열예측 기법을 쓸 수도 있다. 아니면 조금 더 복잡한 예측 방법을 쓸 수도 있다.[Arxiv](https://arxiv.org/abs/1711.04837)

어찌 되었든, 위의 예측모델이 제대로 만들어졌다면 그 의의는 상당하다. 먼저 1) 각 베타에 대한 노출정도(베타)를 알고 있으니 리스크관리 측면에 활용할 수 있고, 2) 기대수익을 계산하는 법을 알면 높은 기대수익을 보이는 종목 중심으로 포트폴리오를 꾸려 투자를 할 수 있다. Risk와 return 개선에 모두 적용될 수 있으니 애초에 맞는 모델을 구축하기가 정말 힘든 것이다.

하지만 저자는 팩터와 수익률의 선형 관계에는 아무런 이론적 근거가 없기에 더 일반적인 함수로 그 둘의 관계를 나타내고 싶어한다. 선형 모델을 택하는 것 자체가 각 팩터의 독립성을 가정하고 factor loading을 이해하는 것이기 때문이다. 그래서 택한 것이 바로 MLP. 96년 당시에도 universal approximation theorem, backprop등이 다 발표가 되어있었던 상황이기 때문에 충분히 고려할 수 있었던 현실적인 방안이었다고 생각한다. 특히 거대 은행에서 일하고 있으면 컴퓨팅 파워도 짱짱할테니..

# Model & Data
결국 저자가 찾고자 하는 함수는 다음과 같다.

$$R_{i} = f(u_{i1}, u_{i2}, \cdots, u_{iL}) + e_{i}$$

여기서 각 $$u_{il}$$은 수익률을 예측하고자 하는 기간의 각 $$\hat{F_{l}}$$에 해당하는 factor loading이다. $$\hat{F_{l}}$$은 historical mean으로 근사했다.

그리하여 팩터-수익률간의 새로운 관계를 찾기 위해 1989년~1995년 사이에 발생된 약 1300여개 종목의 월별 수익률($$R_{i}$$)을 사용했다. BARRA HiCap 종목 universe를 사용했다고 한다. 팩터($$F_{l}$$)로는 ([Fama-French, 1992](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1992.tb04398.x)) 에서 사용한 P/E, P/B 등을 사용했다.  
모델 스펙은 다음과 같다.

2층 뉴럴넷
SGD
Early-stopping
앙상블을 썼는데, 1달치 데이터로 M개의 모델을 만들고, N달치 모델을 종합하여 그 후 월수익률을 예측
각 모델당 랜덤하게 추출한 1분기치 데이터를 validation set으로 사용

# Portfolio Construction
구축한 모델에 따라서 모의투자를 진행한다면 해당 팩터들을 사용한 선형 모델에 비해 Sharpe ratio가 상당히 호전된 모습을 볼 수 있다.

| Portfolio/Model | Linear | Nonlinear |
|:---------------:|:------:|:---------:|
|    All HiCap    |  6.43  |    6.92   |
|  100 long-short |  4.07  |    5.49   |
|  50 long-short  |  3.07  |    4.23   |

물론 저자도 경고하듯 이것을 그대로 실제 투자에 적용하는 것은 위험하다. Turnover가 엄청나고 아직 수수료 계산도 하지 않은 등 고려할 사항이 더 있기 때문이다. 하지만 일단 눈에 띄는 차이가 있다는 것은 분명하고 long-short 포트폴리오에서 그 차이는 더 두드러진다.

그렇다면 이를 바탕으로 실제 투자 전략은 어떤 식으로 세울 수 있을까? 저자는 QP를 제안한다. 마켓 중립성, 포트폴리오 크기, asset turnover, active risk 등의 제약 조건과 거래 수수료 등을 고려 대상으로 삼은 것이다. 그리고 나서 T-Bill, S&P등과 비교하는데 QP로 만든 포트폴리오와 S&P선물을 조합한 'equitised' 포트폴리오(마켓 중립성 상실)가 제일 좋은 out-of-sample성적을 보였다고 한다. 사실 난 결과엔 별로 관심이 없지만..

그리고 귀여운 comment를 다는데, 아무리 이 결과가 좋다고 한들 누가 블랙박스에 돈을 넣겠냐고 하면서 CART같은 트리모델을 사용한 투자전략이 더 잘팔릴 것이라는 뉘양스의 문단을 썼다. 하지만 본인은 이걸로 실제 투자를 해보겠단다.

# My Conclusion
베타 + MLP가 전혀 새로운 방식이 아니라는 것을 보여준 legacy 논문. 안일하게 생각하다가 이도저도 안될 것 같다. 정신 차려야 할듯.
