---
title:  "Deep Portfolio Theory"
excerpt: "Heaton, J. B., N. G. Polson, and Jan Hendrik Witte. "Deep learning for finance: deep portfolios." Applied Stochastic Models in Business and Industry 33.1 (2017): 3-12."
toc: true
toc_sticky: true

categories:
  - Paper Review
tags:
  - Finance
  - Machine Learning
  - Autoencoder
last_modified_at: 2020-01-25T20:17:00+09:00
date: 2020-01-25
---

\* 틀린 게 있을 수 잇음, any kind of feedback are welcome.


논문 링크 1([Arxiv](https://arxiv.org/abs/1605.07230)), 논문 링크2([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2838013))  
둘이 다른 논문임. 커버하는 내용이 조금씩 다름.

Chicago Booth, UCL, Conjecture의 연구진이 2016년 발표한 논문.  
Markowitz, Black-Litterman 등 전통적인 포트폴리오의 철학을 계승하는 동시에 딥러닝 오토인코더 기법을 적용하여 'Deep Portfolio Theory'를 제안함.
언젠가 나도 새로운 학문 분야를 개척할 수 있었으면 좋겠음.

# Introduction
Risk-return trade-off의 개념을 딥러닝 기반의 포트폴리오 구축에 적용하기 위하여 4가지 단계(Encode, Calibrate, Validate, Verify)를 제안했다.
시장의 정보를 비선형으로 압축하는 'deep factor'를 찾아내어 그것을 기반으로 포트폴리오를 구축할 수 있으며, Kolmogorov-Arnold Theorem에 따라 적절한 activation function을 선택하면 실제로 투자가 가능한 포트폴리오를 만들 수 있다는 것이 논문의 기본 주장.

# An Encode-Decode View of the Market
저자가 주장하는 방법 뿐 아니라 Markowitz, Black-Litterman 등 포트폴리오 구축 기법과 APT(에 기반한 선형 팩터모델)등 자산 가격모형 모두 시장의 정보를 압축하는 접근을 취한다고 말한다. 특히 Markowitz의 mean-variance법은 시장을 1차, 2차 moment로 압축하는 것이고(따라서 L2거리로 재기 힘든 fluctutaion 정보를 담기 힘듬), Black-Litterman은 Markowitz와 비슷하지만 Ridge처럼 투자자의 사전 지식을 Regularisation의 형태로 반영한다.  
하지만 위의 두 방법은 모두 ex-ante 효율적 프론티어에 의존해 투자 전략을 세우게 된다. 이 논문에서는 오토인코더를 사용해 시장의 정보를 압축하는 방법을 제안할 뿐 아니라, 머신러닝 모델 선택에 쓰이는 기법을 사용해 ex-post적인 하이퍼파라미터 선택을 선보인다.

# 4 Steps of Deep Portfolio Theory
## Encoding
Encoding step의 목표는 일종의 전처리로서, 시장의 정보를 압축하는 'Market Map', $$F^{m}_{W}(X)$$를 찾는 것이다. 주어진 universe(혹은 index, 같은말임)을 잘 추적하는 새로운 representation을 찾는 것이라고 이해할 수 있다. 아래의 최적화 문제를 풀어 찾는다.

$$\underset{W}{\text{min}}||X-F^{m}_{W}(X)||^{2}_{2} \quad \text{subject to} \quad ||W|| \leq L^{m}$$  

$$F^{m}_{W}(X)$$은 input인 수익률($$X$$)을 reconstruct하는 함수이며, 이 논문에선 오토인코더를 사용한다. 각 종목의 수익률을 얼마나 잘 reconstruct하는지(communal information)를 계산한 다음, well-reconstructed 10개와 least well-reconstructed n개를 뽑아 다음 단계를 진행한다.

## Calibrating
Calibrating step는 1번의 결과물인 시장의 정보를 사용하여 목표 수익률($$Y$$)을 만들어내는 과정이다. 이를 'Portfolio Map', $$F^{p}_{W}(X)$$를 찾는다고 한다.  

$$\underset{W}{\text{min}}||Y-F^{p}_{W}(X)||^{2}_{2} \quad \text{subject to} \quad ||W|| \leq L^{p}$$

## Validating
Validating step은 위 각 최적화 문제의 오차들 사이의 trade-off를 조절해 $$L^{m}$$과 $$L^{p}$$를 찾는 과정이다.

$$\epsilon_{m} = \Vert\hat{X}-F^{m}_{W^{\ast}_{m}}(\hat{X})\Vert^{2}_{2} \quad \text{and} \quad \epsilon_{p} = \Vert\hat{Y}-F^{p}_{W^{\ast}_{p}}(\hat{X})\Vert^{2}_{2}$$  

$$W^{\ast}_{m}$$ 와 $$W^{\ast}_{p}$$는 각각 Encoding과 Calibrating step의 해이다. $$\hat{X}$$와 $$\hat{Y}$$는 모두 test set의 데이터를 가리킨다.

## Verifying
Validating step에서 찾은 적절한 regulatisation을 사용해 실제 'Market Map'과 'Portfolio Map'을 찾는 과정이다. 논문에선 앞서 Encoding step에서 정한 n (non-communal stock의 갯수)를 regularisation의 척도로 보고 out-of-sample accuracy와의 그래프를 그렸다.

# Deep 'Portfolio'인 이유는?
Activation function으로 자주 쓰이는 함수를 보면, 대부분의 non-linearity는 수평선과 기울기가 있는 직선으로 이루어져 있다. 물론 모든 점에서 미분 가능한 부드러운 함수도 많지만 직선과 수평선의 조합으로 어느정도 나타낼 수 있다. 이는 옵션의 payoff의 조합으로 나타낼 수 있는 것이다. 풋과 콜을 조합하면 tanh의 모양이 나오고, ReLU는 그저 콜의 payoff 함수와 동일하게 생겼다. 게다가 Feller(1971)에 따르면 max-layer의 합성은 한개의 max-sum layer와 동일한 것이다. 즉 ReLU를 사용한 깊은 뉴럴넷의 결과값은 여러 옵션의 조합으로 나타낼 수 있는 것이다.  
논문에서 1개의 인덱스와 그 구성 종목 중 2개의 종목을 예시로 들고 있는데, 그 구성종목을 ReLU를 사용하여 조합하면 인덱스의 drawdown을 잡을 수 있다는 것을 보여준다. ReLU 등을 사용한 다층 함수 구조가 시장의 정보를 효율적으로 담아낼 수 있다고 주장하고 있는 것이다. (설명이 더 필요하다.)

# 실험
2012년 1월에서 2016년 4월 사이 Blackrock의 IBB(iShares NASDAQ Biotechnology Index)의 수익률과 해당 지표를 구성하는 종목들의 수익률을 데이터로 사용했다. Encoding과 Calibration step에는 2012년 1월에서 2013년 12월 사이의 데이터를, Validation과 Verification step에는 2014년 1월에서 2016년 4월 사이의 데이터를 사용했다.  
오토인코더의 구조는 5개의 뉴런으로 이루어진 1개의 은닉층을 포함하게 구성하였다.

IBB index를 추종하는 포트폴리오를 구축하고 싶다면 Calibration step에서 $$Y$$를 $$X$$와 동일하게 주고, IBB index를 이기는 포트폴리오를 구축하고 싶다면 해당 step에서 조금 변화를 주어야 한다. 본 논문에서는 $$X$$에서 -5% 미만의 수익률을 보이는 때를 +5%로 인위적으로 바꾸어 주었다. 5%라는 숫자에 대한 reasoning은 모르겠지만, 우리의 목표 수익률이라는 개념으로는 이해할 수 있다. 이는 위에서 설명한, 수익률에 ReLU를 적용한 것과 같다. 결과는 이쁘게 나왔으니까 논문을 냈겟지.
