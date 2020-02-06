---
title:  "Stock Returns and Future Tense Language in 10-K Reports"
excerpt: "Karapandza, Rasa. \"Stock returns and future tense language in 10-K reports.\" Journal of Banking & Finance 71 (2016): 50-61."
toc: true
toc_sticky: true

categories:
  - Paper Review
tags:
  - Text Mining
  - Finance
  - Market
  - Trading
last_modified_at: 2020-01-31T01:49:00+09:00
---

논문 링크([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1910011))

세계적인 금융 정보 서비스 월스트리트저널과 블룸버그에도 소개가 되었던 논문.
[블룸버그 기사](https://www.bloomberg.com/news/articles/2019-06-25/past-present-future-tense-of-stock-talk-weighs-on-returns)

EBS대학교(그 EBS가 아니고 European Business School이 대학교로 승격된 기관) 교수 Rasa Karapandza의 연구 성과로, 한줄로 요약하자면

> 10-K에 미래 시제 표현이 적게 들어간 기업의 주식은 연평균 5%의 추가수익을 낸다.

추가로 정리하자면, 사업 보고서에 미래 시제 표현이 적다는 것은 그만큼 투자자들이 기업의 향후 방향에 대한 정보가 적다는 뜻이고, 정보 부재로부터 오는 리스크에 대한 보상으로 추가수익이 발생한다는 주장이다. Market Efficiency와 관련된 연구. 함 살펴봅시다.

# Introduction
전통적으로는 기업의 수익성, 발생회계의 양, 신주발행이나 자사주매입 등의 리스크팩터에 딸려오는 anomalous return을 identify하는 방향으로 연구가 진행되었다. 모두 기업의 정량적인 지표들. 하지만 저자는 정성적인 분석을 활용한다고 말한다. 이를 통해 사업보고서가 내포한 '기업의 미래 정보'의 정보량을 계산해 낼 수 있다는 것.

물론 이전에도 기업관련 unstructured data에 대한 정성적인 분석 기법은 존재했다. 가장 유명한것이 아마 Loughran & McDonald (2011)의 영단어 금융 sentiment dictionary. 금융 관련 글에 자주 쓰이는 단어에 대한 감성지표를 부여하였다. 신기한 건 그들 자신은 해당 지표와 실제 자산의 수익성 사이 상관성을 주장하지는 않았다는 모양. 하지만 이 사전을 바탕으로 뉴스기사를 감성분석해 수익률과 연관시키려는 시도는 있었다.(Tetlock et al., 2008) 이 외에도 media coverage(Fang & Peress, 2009)나 10-K에 등장하는 지명을 기반으로 한 geographic diversification(Garcia & Norly, 2010) 등 생소하고 기발한 정보를 팩터로 삼으려는 노력은 쭉 존재했음.

근데 Rasa가 제안하는 방법은 위의 사례와는 좀 다르다고 주장한다. 사전을 사용하는 기법처럼 tone을 보는 것도 아니고, media coverage의 예와는 다르게 (둘다 기업의 정보가 시장에 공개되느냐에 집중한건 비슷함) 정보 공개의 정도를 연구한 것이다. 그것도 아주 간단한 방법으로. 바로 본인이 설정한 '미래 시제 단어' 3개를 정해 놓고 그 단어의 발생 빈도를 센 것이다. 정말 간단하다.

이 3개의 표현이라 함은 'will', 'shall', 그리고 'going to'. 영어는 strong FTR(Future Time Reference)언어이기 때문에 미래 내용을 말하기 위해선 꼭 미래 시제 표현이 들어가야 한다고 한다. 즉, 언어적 특성으로 인해 단순히 미래 시제 표현을 세는 것만으로 기업의 계획에 대한 정보량을 알 수 있다는 것이다. 얼핏 보면 단순한 방법을 통해 나온 결과인 것 같지만 실제로는 언어학적으로도 이론적 뒷받침이 있는 주장이었다!

# Experiments
## Data
10-K 리포트는 SEC(미국의 금감원) 홈페이지에서, 종목의 수익률은 WRDS의 CRSP에서 구했다. 없어진 기업도 있고 사명을 바꾼 기업도 있고 데이터 소스를 여러개 사용하다보면 index문제도 생기고 해서 금융 데이터 다룰 때 골치아픈게 한두가지가 아니다. 쨌든 저자는 이전에 선배들이 쓴 방법을 그대로 차용했다고 한다. 특히 Fama-French와 Loughran & McDonald의 방법을 많이 따랐는데, 아니나다를까 이 논문의 draft 수정을 도와준게 후자의 듀오라고 한다. Fama-French는 워낙 거장이고.
역시나 다양한 전처리를 했음. 일단 금융사들은 모조리 제외했고, book equity(주주지분, 장부가치)가 음수인 경우 제외, Compustat에 2년 연속으로 기록되기 전까지의 기록 제외 등.. NYSE, AMEX, NASDAQ의 주식을 다 모아서 결국엔 37,287개의 사업보고서를 모았다.

## Variable
단어의 개수를 센다고 표현했는데, 아무 사업보고서나 갖다놓고 그저 한개, 두개 손으로 세서 숫자를 써놓으면 끝이 아니다. 일단 모든 기업이 사업보고서를 약속하고 한날한시에 발표하는 것이 아니다. 각자 정한 회계년도가 끝나는 날로부터 90일 이내에 발표하면 되는 것이다. 저자는 역시 Fama-French와 같이 7월 1일을 기준일로 잡고 이전 1년동안 발표된 사업보고서를 이용해 이후 1년동안 투자하는 방식을 취하였다. 매년 6월 30일에 발표되는 Moody's manual에 따라서 book equity를 사용하기 때문에 7월 1일을 기준일로 삼아야 lookahead bias가 없다는 것. 그렇다면 그들이 사용한 변수의 계산식을 보자.

$$Frequency \ of \ Future \ Tense_{i,t-1} = \left\{
                \begin{array}{cc}
                  \frac{1+\log{(Number \ of \ will,shall,going \ to_{i,t-1}})}{1+\log{(Number \ of \ Words_{i,t-1})}} & if \quad (Number \ of \ will,shall,going \ to_{i,t-1}) \ge 1\\
                  0 & otherwise\\
                \end{array}
              \right.$$

단순히 개수를 세기보단 전체 문서의 길이에서 해당 글자가 차지하는 비율을 나타내기 위해 전체 문서의 글자 수로 나누어 준 모습이다. Robustness check을 위해서 분모에 글자수 대신 10-K text파일의 용량을 쓰기도 했다. ($$Filesize_{i,t-1}$$ instead of $$Number \ of \ Words_{i,t-1}$$) 글자 수와 크게 다르지 않을 것 같지만 쨌든 했다고 한다.

위의 지표를 계산해 수치가 높게 나온 것들로 꾸린 포트폴리오를 High Future Tense Portfolio, 수치가 낮은 것들로 꾸린 포트폴리오를 Low Future Tense Portfolio라고 칭했는데, 기존의 통념과 저자가 주장하는 것은 Low Future Tense Portfolio가 High Future Tense Portfolio보다 수익률이 유의미하게 높다는 것이다.

# Empirical Results
## Raw Returns
위의 지표로 50-50 long-short portfolio를 만들어 결과를 봤는데 아주 잘나온단다. 비교를 위해서 현재 stock universe에 Fama-French factor 중 HML과 SMB, 그리고 전체 universe의 value weighted portfolio의 초과수익도 같이 그렸다. 모든 시기에 대해서 HML과 SMB를 압도하지만, Mkt-rf에는 못미친다. 원래 시장 전체가 젤 어렵긴 하다. 하지만 Sharpe ratio에서는 모든 비교대상을 압도한다. 놀라운 결과.

회사의 크기에 따라서 영향이 있나 하고 표를 그려봤는데, Market Cap기준 하위 20%, 20%-50%, 그리고 median보다 상위종목들만 따로 봤을 때에도 역시 Low 가 High보다 평균적으로 높다. Value-weighted와 Equal-weighted 모두에서 같은 결과가 나왔다. 기본 세팅은 microcap(하위 20%) 종목은 제거한 것으로 보인다.

## Risk-Adjusted Returns
위의 결과가 혹시 이미 알려진 리스크에 대한 보상으로서 나타난 수익률은 아닐까? 라는 의문을 품은 듯 하다. 타당한 말이다. Future tense expression개수 세가지고 유레카! 했는데 알고보니 사람들이 이미 알고있는 리스크에 대한 proxy였다면? 새로운 정보를 추가하는게 아니기 때문에 저자의 주장은 근거를 잃게 된다. 그리고 우울감에 빠질수도.

이를 위해 저자는 선배들의 방법을 따라하는데, Future Tense로 만들어진 수익률(long-short 수익률)이 얼마나 기존 팩터로 설명이 되나 보기 위해 Fama-French를 참고하여 선형회귀(!!)를 했다. 알려진 팩터를 independent variable로 두고 회귀를 한 다음에 intercept의 통계적 유의성을 보는 것. 기존 팩터의 입장에선 mispricing이라고 주장할 수 있을 것이고 저자의 입장에선 새로운 팩터를 발견했을지도 모른다고 주장할 수 있는 것이다. 이 '알려진 팩터'를 위해 저자는 다음의 4가지 모델을 사용하였다: CAPM, Fama-French 3 factor model, Carhart 4 factor model, Carhart 6 factor model

결과는.. 99% 신회구간에서 유의하다는 것! 가장 좋은 결과인 Carhart 6 factor model과의 대비에선 연평균 7%의 추가 수익이 발생했다.  
이 외에도 Fama-French(2008)에서 선보인 비모수 기법도 사용했다는데, 각 팩터의 matching portfolio를 사용하는 방법인 것 같은데 이부분은 공부가 더 필요하다.. 뭐 이쪽 방법으로도 저자가 원하는 결과가 나왔다고 한다.

# Risk vs Mispricing
저자는 위의 과정을 통해 본인이 발견한 이 수익률이 기존의 팩터로 설명이 안된다는 것을 알아냈다. 그렇다면 이부분은 risk일까 mispricing일까, 즉 리스크에 대한 합당한 보상인 것일까 아니면 리스크 없이 얻을 수 있는 공짜 점심인 것일까? 저자는 이것을 risk, 곧 새로운 팩터의 발견이라고 주장한다. 이를 뒷받침하기 위해 저자는 아주 많은 공을 들인다. 먼저 이 주장에 힘을 싣기 위해 다른 팩터들과 비교를 하는데, 크게 1) Time-series Regression 과 2) GRS F-test를 진행한다.

첫번째 방법은 Richardson, Tuna, Wysocki(2010)에도 잘 정리가 된 방법으로서, 기존의 리스크 팩터로 만든 포트폴리오들 + Low-High 포트폴리오들(10등분, 5등분, 3등분을 하여 long-short portfolio들을 만드는 것) 의 수익률을 가지고 한번에 time-series regression을 실시하는 것이다. 그 후 각 포트폴리오의 *adjusted* $$R^{2}$$, 이상 수익률, Low-High 포트폴리오 팩터의 계수 등을 비교하였더니만 해당 지표가 낮아짐에 따라, 1) *adjusted* $$R_{2}$$는 높아졌고, 2) 이상 수익률도 사라지고, 3) 계수도 대부분의 경우에서 통계적으로 유의하다는 결과를 얻었다.

두번째 방법은 Gibbons, Ross, Shanken(1989)가 제안한 포트폴리오에서의 F-test방법인데, Carhart 4 factor model이 위 포트폴리오의 수익률을 설명한다는 가설에 대한 p-value는 각각 0.072, 0.081, 0.035인데 반해, 그 가설에 Low-High factor가 추가된 모델로 모델을 바꾸면 p-value가 0.218, 0.195, 0.334로 확연히 차이가 나는 것을 관찰 할 수 있다. 또한, Low만으로 만든 포트폴리오의 수익률을 가지고 비슷한 가설 검정을 진행하였는데, Carhart 4 factor model이 Low-only 포트폴리오의 수익률을 설명한다는 가설의 p-value는 무려 0.0068, Carhart 4 factor model에 Low-High 포트폴리오를 추가하니 p-value가 0.199로 치솟았다. 분명히 뭔가가 있긴 있는거다.

# Alternative Explanations
그 다음 부분은 위 부분의 연장으로 정말 정말 노력을 많이 들인 부분이라고 할 수 있다. 본인의 가설을 끊임없이 의심하며, 본인이 제기한 의심을 하나씩 격파해나가는 모습을 보여준다. 심지어 대단한 노가다까지 섞어서.

- 널리 알려진 Anomalies, 여타 기업정보량 관련 변수, 텍스트 감성 점수
- 성장형 기업, 가치형 기업
- 산업-특화 언어
- 영어가 문제 아니냐
- 'will','shall','going to', 어쩌다 찾은 단어의 조합은 아닌지?  
이를 위해서 10-K에서 가장 빈번하게 쓰인 100개의 단어와 그 중 세 단어의 모든 조합을 가지고 포트폴리오를 만들어서 수익률을 비교했다. 논문에서 쓴 표현은 'I try all of them.' \$wag. 근성이 대단하다. 총 100 + 161,700 = 161,800개의 포트폴리오를 구축한 것. 박수박수. 결국 'will, shall, going to'의 조합을 이길 수 있는 단어 조합은 한 개도 없었고, 뿐만 아니라 저자가 제시한 4개의 robustness check을 통과한 포트폴리오조차 한개도 없었다. (당연히 future tense 포트폴리오는 4가지 모두 통과) 특히, shall은 Loughran & McDonald에서 520위, going과 to는 심지어 사전에 등록되어 있지도 않다. 어쩌다 얻어걸린 조합일 수가 없다는 주장.

이부분은 나중에 추가하겠음

# My Conclusion
정말 대단한 논문. 다른게 아니고 본인의 주장을 뒷받침하기 위해 얼마나 많은 생각을 하고 자신의 가설을 부정하는 노력을 했을지 짐작이 조금 간다. 연구 애새\*인 나는 아직 완전히 이해하지는 못할 것이다. 정말 간단한 주장이라 신문에도 실릴 정도의 내용이지만, 그 속을 들여다보면 탄탄한 논리가 있다. 영어의 언어학적 성질, 금융 커뮤니티의 consensus 등 본인이 생각할 수 있는 모든 의심을 통한 검증의 결과물이라고 생각한다.

난 평생 이런거 쓸 수 있을런지.
