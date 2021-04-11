---
title:  "[Paper Review] Transformer-xl: Attentive language models beyond a fixed-length context"
excerpt: "Dai, Zihang, et al. \"Transformer-xl: Attentive language models beyond a fixed-length context.\" arXiv preprint arXiv:1901.02860 (2019)."
toc: true
toc_sticky: true

categories:
  - Paper Review
tags:
  - Machine Learning
  - Natural Language Processing
last_modified_at: 2020-12-04T16:00:00+09:00
---

Arxiv [링크](https://arxiv.org/abs/1901.02860)

카네기랑 구글에서 낸 논문. 기존 트랜스포머의 인풋 시퀀스 길이가 고정되어야 한다는 단점을 보완하였다. XL for extra long. Shijak.

# Introduction

잘나가던 트랜스포머에 무슨 문제가 있던 걸까. 언어 모델링에서 전통적으로 쓰이던 RNN은 gradient-vanishing/explosion 등 시퀀스 길이가 길어지면 문제가 많은 모델이었다. 물론 LSTM이나 클리핑 등 여러 시도가 있긴 했으나 같은 구조의 모델이 계속해서 쓰인다는 특성상 어쩔 수 없는 단점이다. 트랜스포머는 이 부분을 획기적으로 해결하는 모습을 보여주었다. Self-attention을 사용하여 인풋 토큰의 의미를 알아내기 위해 시퀀스 내의 다른 토큰 중 어디에 'attend'해야하는지를 학습하게 하였고, 큰 성공을 거두었다.

하지만 이 트랜스포머도 결국에는 RNN처럼 '시퀀스 길이'에서 단점을 드러낸다. (물론 둘의 단점은 다른 맥락이다.) 트랜스포머는 구조의 특성상 인풋 시퀀스의 길이가 고정되어있다. 만약 트랜스포머가 기대하는 시퀀스의 길이보다 짧은 데이터가 들어온다면 끝부분을 0으로 패딩해주면 그만이지만, 데이터의 길이가 더 길 경우에는 문제가 생긴다. 긴 데이터를 트랜스포머 인풋 길이에 맞게 잘라 여러 조각을 만든 다음 순차적으로 모델에 넣을 수 밖에 없는 것인데, 조금만 생각해보면 최선이 아니라는 것을 알게 될 것이다. 인간은 한 문장을 한번에 읽고 이해한다. 하나의 문장을 여러 조각으로 나눈 다음 각 조각을 각기 다른 사람에게 주고 이해하라고 하면 잘 작동될 리가 없는 것과 마찬가지이다. 이 논문에서는 '조각'을 'segment'라고 칭한다.

특히, 언어 모델은 문장을 만들어내거나 예측해야 하는 목적을 가지고 있는데, 문장이 중간에 잘리면 'segment'가 새로이 시작되는 부분에서 문제가 심각하다. 문장의 앞부분에 대한 배경이 없기 때문이다. 이런 문제를 논문에서는 'context fragmentation'이라고 한다. 문맥이 조각난다는 의미이다.

그뿐만 아니라, 트랜스포머는 모델을 사용할 때에도 (evaluation phase) 인풋 길이가 넘어가는 순간부터 한 토큰을 출력할때마다 모든 인풋에 대해서 모델 연산을 해야한다. 효율적이지 않은 것은 사실이다.

Transformer-XL이 어떻게 이 문제를 해결하는지 살펴봅시다.

# Model

토큰 시퀀스 $$\bold{x}=(x_1,\dots,x_T)$$가 있을 떄, 언어모델은 이 토큰의 결합확률분($$P(\bold{x})$$)를 추정하는 것으로 이해할 수 있는데, 언어이다 보니 순서가 중요하다. 즉, 자기회귀적(autoregressive)으로 분해를 하여 생각한다. $$P(\bold{x})=\prod_tP(x_t\mid\bold{x}_{<t})$$ 이기 때문에, 결국 조건부 항을 각각 추정하면 결합확률분포를 추정할 수 있게 된다. 뉴럴넷을 이용해 $$\bold{x}_{<t}$$를 정해진 길이의 벡터(hidden state)로 표현하고, 단어 임베딩과 곱해져서 나온 로짓을 소프트맥스에 넣어서 다음 토큰을 예측하는 방식이다.

트랜스포머에 순환형 구조를 더하고 이를 가능하게 하기 위한 새로운 positional embedding방식을 제안한다. 먼저 바닐라 트랜스포머 모델부터 보자.

## Vanilla Transformer

이상적으로, 임의의 시퀀스를 정해진 길이의 hidden state로 임베딩하기 위해서는 엄청 긴 시퀀스를 한번에 트랜스포머에 넣어야 하기 때문에 메모리와 연산량이 무한으로 많이 필요하다. 하지만 실제로는 그렇지 않기 때문에 앞서 살펴본대로 시퀀스를 조각내서 모델에 넣는다. 이렇게 할 경우, 조각 사이에서 정보 공유가 불가능하다. 이 때문에 일어나는 문제가 크게 두 가지가 있다.

첫 번째는 토큰 간의 의존성을 나타낼 수 있는 시퀀스 길이가 일정 값 이하로 제한된다는 것이다. RNN의 vanishing gradient 현상을 타개하기 위해 제안된 트랜스포머지만 결국에는 RNN과 비슷하게 순차적으로 연산을 진행해야 하는 때가 있고, 트랜스포머 모델 구조를 낭비하는 결과로 이어진다. 두 번째로, 시퀀스 길이가 길면 모델이 받아들일 수 있는 길이를 기준으로 입력 시퀀스를 잘라서 넣기때문에 문장이 중간에 잘려서 들어가게 되고, 앞에서 언급한 context fragmentation 현상이 일어나게 된다.

Evaluation phase에도 한 토큰을 예측하기 위해 모델 입력의 최댓값에 해당하는 토큰들을 모두 사용하여 연산을 진행하게 되어서 상당히 비효율적이다.

이 두 가지 문제를 해결하기 위해 제안된 두 가지 요소가 아래 내용이다.

## Segment-Level Recurrence with State Reuse

트랜스포머 모델은 고정된 길이의 인풋밖에 받아들이지 못하기 때문에 recurrence개념을 도입한다. 먼저, 인풋을 조각낸 다음 각 조각을 트랜스포머에 넣어 hidden state를 계산한다. 다음 조각이 트랜스포머에 들어가서 연산이 진행될 때 이전 조각의 hidden state가 쓰이는 구조이다. RNN과 비슷하다. $$\bold{h}^{n}_{\tau}\in\mathbb{R}^{L\times d}$$가 $$\tau$$번째 조각 $$\bold{s}_{\tau}$$의 $$n$$번째 레이어 hidden state라고 하면,

$$\tilde\bold{h}^{n-1}_{\tau+1}=[SG(\bold{h}^{n-1}_{\tau})\circ\bold{h}^{n-1}_{\tau+1}],\newline \bold{q}^{n}_{\tau+1}, \bold{k}^{n}_{\tau+1}, \bold{v}^{n}_{\tau+1},= \bold{h}^{n-1}_{\tau+1}\bold{W}^\top_{q}, \tilde\bold{h}^{n-1}_{\tau+1}\bold{W}^\top_{k}, \tilde\bold{h}^{n-1}_{\tau+1}\bold{W}^\top_{v}\newline \bold{h}^{n}_{\tau+1}=\text{Transformer}(\bold{q}^{n}_{\tau+1}, \bold{k}^{n}_{\tau+1}, \bold{v}^{n}_{\tau+1})$$

위와 같이 $$\tau+1$$번째 조각에 대한 $$n$$번째 레이어 hidden state가 계산된다. Key와 value벡터를 구할 때 이전 hidden state와 지금 hidden state를 이어붙인 행렬을 사용하는 것이 다른점이다. 이 hidden state는 크기를 보면 알 수 있듯 각 토큰에 대한 정보를 저장하고 있다. 각 원소가 그 위치의 토큰을 나타낸다고 이해하면 된다.

이런 방식으로 모델은 매번 두 개 조각에 대한 정보를 처리하게 되는데, 직전 조각의 hidden state에는 그 전 조각에 대한 정보도 담겨있기 때문에 긴 시퀀스의 의존성까지 모델링이 가능하다는 주장이다. 또한, RNN과는 다르게 $$n$$번째 레이어의 hidden state는 이전 조각의 전 레이어($$n-1$$번째)의 hidden state의 정보를 활용한다. 각 레이어마다 hidden state가 있어서, 레이어의 개수에 따라서 얼마나 긴 시퀀스에 대한 정보를 모델링할 수 있는지가 결정된다.

이 방법을 사용하면 evaluation phase에서도 연산적인 이점이 작용한다. 모든 연산을 처음부터 다 수행하는게 아니라 이전 조각에 대한 정보는 hidden state에 캐시하여 저장하고 있기 때문이다.

## Relative Positional Encodings

트랜스포머에서는 토큰의 위치 정보를 모델에 반영하기 위해서 positional encoding을 수행하여 토큰에 더해준다. 기존 트랜스포머는 시퀀스 길이가 정해져있기때문에 토큰의 ‘절대적 위치’를 기반으로 positional encoding을 수행하면 됐었다. 하지만 이제는 다르다. 기존 Positional encoding 행렬 $$\bold{U}\in \mathbb{R}^{L_{max}\times d}$$가 있다고 하면, ($$L_{max}$$는 모델이 받을 수 있는 최대 시퀀스 길이이다) 이 행렬을 그대로 적용할 경우

$$\bold{h}_{\tau+1}=f(\bold{h}_{\tau},\bold{E}_{\bold{s}_{\tau+1}}+\bold{U}_{1:L})\newline \bold{h}_{\tau}=f(\bold{h}_{\tau-1},\bold{E}_{\bold{s}_{\tau}}+\bold{U}_{1:L})$$

처럼 나타낼 수 있다. $$\bold{E}_{\bold{s}_{\tau}} \in \mathbb{R}^{L \times d}$$는 $$\bold{s}_{\tau}$$의 임베딩을 나타내는 행렬이다. 위에서 볼 수 있듯, positional encoding을 그대로 쓰게된다면 $$\bold{E}_{\bold{s}_{\tau}}$$와 $$\bold{E}_{\bold{s}_{\tau+1}}$$에 동일한 positional encoding이 적용되는 문제가 발생한다. 토큰이 어떠한 조각에서 왔는지에 대한 정보가 손실되는 것이다.

이를 해결하기 위해서 absolute positional encoding이 아니라 relative positional encoding을 사용한다. 이전에도 relative positional encoding을 사용하려는 시도는 있었는데, 이 논문은 조금 다른 방향으로 구현한다. 이전 연구들은 positional encoding을 인풋에 직접적으로 적용하였으나 Transformer-XL에서는 이 정보를 각 레이어의 어텐션 스코어에 주입하는 방식을 택했다. Relative positional encoding $$\bold{R}\in \mathbb{R}^{L_{max}\times d}$$를 만든다. 각 행 $$i$$는 은 쿼리 입장에서 봤을 때 키가 $$i$$만큼 떨어져있을 때 적용되어야 할 temporal 정보이다. 이 방법을 사용하면 위에서 제기된 문제를 해결할 수 있다. 어느 조각에서 나온 토큰인지 구별할 수 있게 된다.

수식을 살펴보자. 기존 positional encoding을 적용한 트랜스포머의 셀프 어텐션을 전개하면 아래와 같은 식이 된다.

$$\text{A}^{\text{abs}}_{i,j}=\underbrace{\bold{E}^\top_{x_i}\bold{W}^\top_{q}\bold{W}_{k}\bold{E}_{x_j}}_\text{(a)}+\underbrace{\bold{E}^\top_{x_i}\bold{W}^\top_{q}\bold{W}_{k}\bold{U}_{j}}_\text{(b)}+\underbrace{\bold{U}^\top_{i}\bold{W}^\top_{q}\bold{W}_{k}\bold{E}_{x_j}}_\text{(c)}+\underbrace{\bold{U}^\top_{i}\bold{W}^\top_{q}\bold{W}_{k}\bold{U}_{j}}_\text{(d)}$$

이제 relative positional encoding을 적용한 식을 보자.

$$\text{A}^{\text{rel}}_{i,j}=\underbrace{\bold{E}^\top_{x_i}\bold{W}^\top_{q}\bold{W}_{k,E}\bold{E}_{x_j}}_\text{(a)}+\underbrace{\bold{E}^\top_{x_i}\bold{W}^\top_{q}\bold{W}_{k,R}\bold{R}_{i-j}}_\text{(b)}+\underbrace{u^\top\bold{W}_{k,E}\bold{E}_{x_j}}_\text{(c)}+\underbrace{v^\top\bold{W}_{k,R}\bold{R}_{i-j}}_\text{(d)}$$

크게 변화를 준 것이 세 가지 있다.

먼저, (b)와 (d)에 나오는 absolute positional encoding을 모두 relative positional encoding으로 바꿔준다.

다음으로는 쿼리의 위치 정보에 해당하는 $$\bold{U}^\top_{i}\bold{W}^\top_{q}$$를 하나의 벡터로 바꾸어 주는 것이다. Relative positional encoding을 쓴다는 것은 쿼리의 관점에서 key와 value를 바라본다는 의미이기 때문에 쿼리의 위치정보는 고정되는 것이 맞기 때문이다. (c)에 있는 쿼리는 $$u\in \mathbb{R}^d$$로, (d)에 있는 쿼리는 $$v\in \mathbb{R}^d$$로 두었다.

마지막으로, key의 weight matrix $$\bold{W}_{k}$$를 $$\bold{W}_{k,E}$$와 $$\bold{W}_{k,R}$$로 분리했다. 각각 임베딩에 관련된 항과 위치에 관련된 항에 들어가는 weight matrix이다.

직관적으로 의미를 부여해보면, (a)는 단어의 의미를 나타내는 부분, (b)는 단어의 의미에 관련된 위치 정보를 나타내는 부분, (c)는 시퀀스의 전체적인 의미를 나타내는 부분, (d)는 시퀀스의 위치정보를 나타내는 부분이다.

결론적으로 종합해보면 Transformer-XL은 아래와 같은 식으로 정리할 수 있다.

$$\tilde\bold{h}^{n-1}_{\tau}=[SG(\bold{m}^{n-1}_{\tau})\circ\bold{h}^{n-1}_{\tau}],\newline \bold{q}^{n}_{\tau}, \bold{k}^{n}_{\tau}, \bold{v}^{n}_{\tau},= \bold{h}^{n-1}_{\tau}{\bold{W}^{n}_{q}}^\top, \tilde\bold{h}^{n-1}_{\tau}{\bold{W}^{n}_{k,E}}^\top, \tilde\bold{h}^{n-1}_{\tau}{\bold{W}^{n}_{v}}^\top\newline \bold{A}^{n}_{\tau,i,j}={\bold{q}^{n}_{\tau,i}}^\top\bold{k}^{n}_{\tau,j}+{\bold{q}^{n}_{\tau,i}}^\top\bold{W}^{n}_{k,R}\bold{R}_{i-j}\newline+u^\top\bold{k}_{\tau,j}+v^\top\bold{W}^{n}_{k,R}\bold{R}_{i-j}\newline\bold{a}^{n}_{\tau}=\text{Masked-Softmax}(\bold{A}^{n}_{\tau})\bold{v}^{n}_{\tau}\newline\bold{o}^{n}_{\tau}=\text{LayerNorm}(\text{Linear}(\bold{a}^{n}_{\tau})+\bold{h}^{n-1}_{\tau})\newline\bold{h}^{n}_{\tau}=\text{Positionwise-Feed-Forward}(\bold{o}^{n}_{\tau})$$

메모리에 $$m$$을 쓴 것은 메모리의 길이를 임의로 정할 수 있기 때문이다. 구현할 때는 $$\bold{W}^{n}_{k,R}\bold{R}_{i-j}$$의 연산이 상당히 복잡하기 때문에 $$i-j$$가 일정 범위 이내의 값만을 가진다는 것에 착안하여 효율적인 계산법을 제안했다. 논문의 appendix에 나와있는데, 간단한 행렬 연산을 한 뒤 인덱스를 바꿔주는 계산법을 사용하였다.

# Conclusion

오랜만에 제대로 읽은 NLP논문인데 활용할 수 있는 곳이 많을 것 같다!
