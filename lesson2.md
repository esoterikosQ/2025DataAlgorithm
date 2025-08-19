## 1. 선형회귀

### 1.1. 다중선형회귀를 행렬식으로 표현

```math
\boldsymbol y = \boldsymbol {X \beta} + \boldsymbol \epsilon 
```

이때, $\boldsymbol y$와 $\boldsymbol \epsilon$는 $(n \times 1)$, $\boldsymbol X$는 $(n \times p)$, $\boldsymbol \beta$는 $(p \times 1)$ 차원의 벡터와 행렬이며, $x_{11} = x_{21} = \cdots = x_{n1} = 1$는 절편항인 모형으로, 절편까지 포함한 변수의 개수가 $p$개, 관측치의 수는 $n$개이다.

```math
\boldsymbol y = \begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}

\qquad

\boldsymbol X = \begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
&&\vdots \\
x_{n1} & x_{n2} & \cdots & x_{np} \\
\end{pmatrix}

\qquad

\boldsymbol \beta = \begin{pmatrix}
\beta_1 \\
\beta_2 \\
\vdots \\
\beta_p
\end{pmatrix}

\qquad

\boldsymbol \epsilon = \begin{pmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{pmatrix}
```

---

### 1.2. 행렬식을 이용한 회귀계수 계산

정규방정식을 이용하여 최소제곱법으로 회귀계수 벡터 $\boldsymbol \beta$의 추정치 $\hat {\boldsymbol \beta}$를 구하면 아래와 같다.

```math
\boldsymbol \epsilon = \boldsymbol y - \boldsymbol {X \beta}
```

```math
\begin{aligned}
\boldsymbol {\epsilon^2} = S &=  (\boldsymbol y - \boldsymbol {X \beta})'(\boldsymbol y - \boldsymbol {X \beta}) \\

&= \boldsymbol {y'y} - \underbrace{\boldsymbol {y'X\beta}}_{scalr값} - \underbrace{\boldsymbol {\beta'X'y}}_{scalr값} + \boldsymbol {\beta'X'X\beta} \\

\\

&= \boldsymbol {y'y} - 2\boldsymbol {\beta'X'y} + \boldsymbol {\beta'X'X\beta}\\

\end{aligned}
```

이때, $\boldsymbol {\epsilon^2}$을 최소화하기 위한 $\boldsymbol {\beta}$를 구하기 위해서는 위 식을 $\boldsymbol {\beta}$로 미분한 값이 0이 되면 된다.

```math
\begin{aligned}

\frac { \partial {\boldsymbol S} } { \partial {\boldsymbol \beta}} &= 0 - 2\boldsymbol {X'y} + \boldsymbol { X'X\beta} + \boldsymbol { \beta'X'X} \\

\\

&= - 2\boldsymbol {X'y} + 2\boldsymbol { X'X\beta} = 0 \tag{normal  equation}\\

\end{aligned}
```

```math
\boldsymbol { X'X\beta} = \boldsymbol {X'y} \\

\\

\Rightarrow \boldsymbol {\hat \beta} = \boldsymbol {(X'X)^{-1}X'y}
```

- 회귀모형은 $\beta_n$의 값에 따라 달라진다. $\Rightarrow$ 회귀모형의 모수(parameter)는 회귀계수.

- 회귀계수를 구하는 과정에서 선형회귀분석의 5대 가정(독립변수의 독립성, 모형의 선형성, 오차 상호간의 비상관성, 오차의 정규성, 오차의 등분산성)이 필요없음 $\Rightarrow $ 선형회귀분석의 가정은 모형의 타당성을 진단하고 검정통계량을 구하는 과정에서 활용

> 참고자료 : 파이썬 노트북 2.1.

---

### 1.3. 다변량 데이터에서 선형회귀분석이 어려운 이유1 : 정칙행렬

정규방정식을 통해 계산한 해인 $\hat {\boldsymbol \beta}$가 존재하기 위해서는 $\boldsymbol{(X'X)^{-1}}$가 존재해야 한다. 그런데 $n \ll p$인 다변량 데이터에서는 $\boldsymbol{(X'X)^{-1}}$가 존재하지 않는다. 따라서, 최소제곱법(ordinary least square, OLS)에 근거한 선형회귀분석의 해는 다변량 데이터에서는 존재하지 않는다.

- $\boldsymbol {(X'X)^{-1}}$가 존재하지 않는 이유 : 행렬 $\boldsymbol X$가 full-rank가 아니어서 $rank(\boldsymbol X) \leq n < p$이기 때문에 $(p \times p)$차원의 정방행렬인 $\boldsymbol {(X'X)}$는 $rank(\boldsymbol {(X'X)}) \leq n < p$인 정칙행렬(singular matrix)이다. 따라서, $\boldsymbol {(X'X)}$의 역행렬은 존재하지 않는다.

    - 행렬의 성질로부터 유도(정칙행렬, 행렬의 rank)

    - 직관적으로 이해하면, $(n \times m)$차원의 행렬을 $n$차원의 데이터를 $m$차원으로 변환하는 함수의 집합이라고 이해하면, 역행렬은 $(m \times n)$으로 변환된 $m$차원의 데이터를 원래의 $n$차원의 데이터로 환원하기 위한 함수의 집합 (수학적으로 $n \neq m$이면 역행렬이 존재하지 않음)

    - 정칙행렬 : $(n \times n)$ 차원의 정방행렬의 행렬식(determinant) 값이 0이거나 rank가 $n$보다 작은 경우 정칙행렬(singular matrix)이라고 한다. 정칙행렬은 $n$차원의 입력 정보를 변환하는데, 정칙행렬의 변환차원은 $n$보다 작기 때문에 변환 과정에서 $a$개($a>1$) 이상의 차원에서 정보를 손실하고, 이 때문에 변환 후의 $(n-a)$ 차원으로부터 $n$차원의 정보를 복원할 수 없다. 따라서, 정칙행렬은 역행렬이 존재하지 않는다.

    - 그런데, 다변량 데이터를 입력값으로 주고 파이썬에서 회귀분석을 돌리면 회귀계수 추정값을 내준다. 그것은 패키지에서 사용하는 방법은 역행렬을 이용한 OLS가 아니라 의사역행렬(pseudo inverse matrix)을 이용하기 때문

    - 역행렬이 존재하지 않는다는 말은 역행렬이 무수히 많다는 뜻. 의사역행렬은 무수히 많은 역행렬 중 기하학적으로 진정한 역행렬의 값에 가장 가까운 행렬을 의미.

---

### 1.4. 다변량 데이터에 선형회귀분석이 어려운 이유2 : 추정통계의 불안정성

적은 수의 데이터로 모수를 추정했을 때에 나타날 수 있는 일반적인 한계로 추정량의 불안정성이 있다. 표준정규분포 $N(0, 1)$을 따르는 확률변수 $X$를 추정하기 위해 관측치가 1개인 경우와 100개인 경우의 예를 생각해보면, 동일한 관측을 10번 반복한다고 했을 때에 관측치가 1개인 경우에 $X$의 추정량이 관측치가 100개인 경위 $X$의 추정량보다 분산이 커질 수 밖에 없다. 

> 참고자료 : 파이썬 노트북 2.1.

비슷한 문제가 다변량 데이터의 선형회귀분석에서 나타난다. 즉, 변수의 수보다 적은 수의 관측치만을 가지고 있기 때문에 필연적으로 1개 이상의 변수는 수준 분리가 일어나지 않는다. 단순하게 모든 변수가 {0, 1}로 이루어진 범주형 변수라고 할 때, 모든 변수에 대해 0/1 수준을 교차하여 관측하기 위해 필요한 최소한의 관측치 수는 $2^p > p$인데, 이보다도 적은 수의 관측치를 가진 데이터에서는 많은 변수가 관측치는 가지고 있지만 수준에서 차별화가 되지 않는 문제에 직면한다. 따라서, 분리의 수준이 낮은 변수에 대해서는 회귀계수 추정에 필요한 데이터가 실제로는 n개가 아니라 0 또는 1개인 상황이 발생할 수 있다. 이는 앞서 본 것과 같은 1개를 관측하여 모수를 추정하는 것과 유사한 문제로, 결국 추정통계량의 불안정성이 커지게 된다.

이를 수학적으로 표현하면 아래와 같다.

회귀계수 추정량 벡터 $\boldsymbol {\hat \beta}$의 분산은 아래 과정을 거쳐 구할 수 있다.

```math
\begin{aligned}

\boldsymbol {\hat \beta} &= \boldsymbol { (X'X)^{-1}X'y } \\

&= \boldsymbol { (X'X)^{-1}X'(X\beta + \epsilon) } \\

&= \boldsymbol { \underbrace{ (X'X)^{-1}X'X}_{= I} \beta + (X'X)^{-1}X'\epsilon } \\

&= \boldsymbol { \beta + (X'X)^{-1}X'\epsilon }

\end{aligned}
```

따라서,

```math
\boldsymbol {\hat \beta - \beta} = \boldsymbol { (X'X)^{-1}X'\epsilon }
```

양변에 분산을 취하면,

```math
Var ( \boldsymbol {\hat \beta - \beta} ) = Var( 
\boldsymbol { (X'X)^{-1}X'\epsilon } )
```

```math
\begin{aligned}

\Rightarrow Var ( \boldsymbol {\hat \beta} ) &= Var( \boldsymbol { (X'X)^{-1}X'\epsilon } ) \\

&= \boldsymbol{(X'X)^{-1} X'} \boxed {Var(\boldsymbol \epsilon) }_{=\sigma^2\boldsymbol I} \boldsymbol{X(X'X)^{-1}} \\

&= \sigma^2 \boldsymbol{(X'X)^{-1}} \underbrace { \boldsymbol{X'X (X'X)^{-1}} }_{= \boldsymbol I} \\

&= \sigma^2 \boldsymbol{(X'X)^{-1}}

\end{aligned}
```

따라서, 회귀계수 추정치와 마찬가지로 $\boldsymbol{(X'X)^{-1}}$가 존재하지 않으면 분산도 존재하지 않는다. 역행렬 대신 의사역행렬을 이용하는 경우 분산을 구할 수는 있으나 의사역행렬에는 특잇값의 역수가 사용된다. 그런데 $n \ll p$인 경우에는 $p-n$개 이상의 특잇값이 0이 되고, 0이 아닌 경우에도 0에 가까운 특잇값을 많이 가지게 된다. 이 가중치 행렬을 포함하는 의사역행렬은 실제 데이터 간의 관계를 필요 이상으로 변환하게 되고 이것이 "큰 분산 = 추정통계량의 불안정성"으로 이어지게 된다.

> 의사역행렬이란? : 역행렬이 존재하지 않는 경우 다음 4가지 조건을 만족하는 유일한 행렬을 의사역행렬이라고 한다.  
> 1. $\boldsymbol {AA^+A} = \boldsymbol A$
> 2. $\boldsymbol {A^+AA^+} = \boldsymbol {A^+}$
> 3. $\boldsymbol {(AA^+)'} = \boldsymbol {AA^+}$
> 4. $\boldsymbol {(A^+A)'} = \boldsymbol {A^+A}$  
> 
> 의사역행렬을 구하기 위해서는 특잇값 분해(singular value decomposition, SVD)의 과정을 역으로 활용한다. 즉, $\boldsymbol A$를 다음과 같이 분해한다면,
> ```math
> \boldsymbol A = \boldsymbol {U \Sigma V'}
> ```
> 의사역행렬은 다음과 같이 구할 수 있다.  
> ```math
> \boldsymbol {A^+} = \boldsymbol {V \Sigma^+ U}
> ```
> 여기서 $\boldsymbol \Sigma^+$는 $\boldsymbol \Sigma$에서 0이 아닌 원소는 역수를 취하고 0인 원소는 0을 취한 대각행렬이다.

---

### 1.5. 다변량 데이터에 선형회귀분석이 어려운 이유3 : 과적합

실용적인 관점에서 다변량 데이터 처리가 어려운 이유는 수학적으로 엄밀한 해의 부존재보다 과적합이다. 수준이 분리되는 변수에 대해서만 분석 모형의 적합이 이루어지기 때문에 그 결과 추정한 모형의 모수는 제한적인 범위에서만 예측 효용성을 가진다. 이 때문에 입력 데이터와 동일한 p차원의 새로운 데이터를 이용하여 반응변수를 예측하면 적합 데이터의 범위를 벗어나는 문제가 생긴다.

수학적인 예를 들면, <가로>, <세로>, <높이>를 설명변수로 가지는 3차원 데이터셋이 있다고 할 때에, <가로>에 해당하는 변수가 모두 10이라는 값만을 가지면 이 데이터의 차원은 3차원이지만 실제로 데이터를 통해 확인할 수 있는 변수의 변화는 2차원에 불과하기 때문에 <가로>가 10이 아닌 경우에 반응변수가 어떻게 변화하는지를 적절하게 예측할 수 없다.
