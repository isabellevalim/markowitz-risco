# markowitz-risco
Modelo de otimização para otimização de portfólio de investimentos que utiliza inteligência artificial para prever o retorno financeiro esperado

## Estrutura do código

Class ReturnsPredictor – contém os métodos necessários para construção do modelo de predição dos valores de retorno para o ano 2021
Class Optimizer – contém os métodos necessários para otimização da carteira utilizando redes neurais
Class Markowitz – contém os métodos necessários para alocação de carteiras utilizando a teoria de Markowitz


## Sobre o projeto

A Teoria de Markowitz calcula o binômio risco-retorno com base na média e covariância dos dados. O problema disso é que, para um conjunto grande de ativos, os cálculos ficam muito complexos e demanda muito esforço computacional.

Proposta de solução: utilização de uma rede neural do tipo recorrente para predição dos valores de retorno do próximo ano. O cálculo do binômio risco- retorno é feito com base nessa predição de retorno futuro.


## Criando a fronteira eficiente usando a Teoria de Markowitz

Para construção da fronteira eficiente, foram criados aleatoriamente 1000 possíveis portifólios de alocação. Todos foram calculados com base no valor médio dos retornos, levando em consideração os anos entre 1928 e 2020. O portifólio escolhido como mais adequado é o que apresenta maior índice Sharpe.

Para o cálculo do índice Sharpe, foi considerado que a taxa do ativo livre de risco é zero.

Considerando que o cálculo do índice Sharpe para todas as carteiras e identificação do melhor portifólio é um processo custoso, foi implementado um otimizador baseado em LSTM para otimização do processo de alocação. Em ambos os casos, o valor utilizado como retorno esperado para 2021 foi o valor do retorno médio de todos os anos.

## Otimizador de portfólios baseado em LSTM

Para utilização deste modelo de otimização de portifólios, foi necessário realizar a transformação dos retornos anuais dos ativos em séries de preços que respeitam esses retornos. Para isso, foi criado um ano base (1927) e considerado que, para esse ano, o preço unitário para todos os ativos era 1. Dessa forma, o preço dos ativos dos próximos anos, foram calculados como sendo𝑃 = 𝑃 .(1+𝑅 ),sendo P opreço e R o retorno.

O otimizador baseado em LSTM possui 4 neurônios na camada de entrada (correspondente aos retornos dos 4 ativos), 4 neurônios na camada intermediária e 4 neurônios na camada de saída (correspondente aos pesos de cada ativo no portifólio). Foram utilizadas as funções de ativação “sigmoid” e “softmax” nas camadas intermediária e de saída, respectivamente. A função objetivo a ser minimizada pelo otimizador “RMSProp” é 𝑓 = 𝑒−𝑠h𝑎𝑟𝑝𝑒 . Dessa forma, é possível maximizar o índice Sharpe.

Além disso, uma camada do tipo Dropout() foi adicionada à rede com o objetivo de prevenir a ocorrência de overfitting. Com a utilização do Dropout(), o processo de treinamento do modelo é modificado. Alguns neurônios da camada intermediária da rede são eliminados aleatoriamente. Isso significa que a rede aprenderá um conjunto de informações sob a condição de descarte de neurônios, que seria como treinar várias redes neurais diferentes. Esse procedimento de eliminação é semelhante a calcular a média dos efeitos de uma grande quantidade de topologias diferentes. 

Isso confere maior robustez ao modelo, já que essas redes neurais diferentes se adaptam de maneiras diferentes.

A Figura 1 apresenta a fronteira eficiente construída com a metodologia descrita. Os pontos destacados referem-se ao portifólio com maior índice Sharpe identificados pela teoria clássica (círculo) e pelo otimizador (xis).

<img width="537" alt="Captura de Tela 2023-07-17 às 10 42 07" src="https://github.com/testedaisa/markowitz-risco/assets/121257762/f0fb777d-0b1f-453a-935a-99221f49fa66">

A Figura 2 apresenta a sugestão de alocação dos ativos correspondente aos dois pontos destacados na fronteira de decisão.

<img width="619" alt="Captura de Tela 2023-07-17 às 10 42 39" src="https://github.com/testedaisa/markowitz-risco/assets/121257762/0fa8bfb3-9cf4-4ed2-9a3b-ce9902c1c6c0">


## Criando a fronteira eficiente utilizando redes neurais

Para este desafio, foi escolhida uma rede neural recorrente, capaz de armazenar informações ao processar novas entradas. Essa memória a torna ideal para tarefas de processamento, onde as entradas anteriores precisam ser consideradas (problemas com séries temporais, por exemplo).

A solução proposta foi implementada utilizando a abordagem da memória de curto e longo prazo (Long Short-Term Memory – LSTM) em aprendizagem profunda. A escolha foi baseada nos trabalhos de Cao et al., 2020, Ta et al., 2020, Ma et al., 2020 e Mahlawat et al., 2020. Todos estes autores utilizaram uma rede de aprendizagem profunda do tipo LSTM para otimização de portifólios.

Para predição do valor de retorno para o ano de 2021, foram considerados os retornos dos últimos 5 anos. A topologia utilizada apresenta 4 neurônios na camada de entrada (correspondente aos retornos dos 4 ativos), 4 neurônios na camada intermediária e 4 neurônios na camada de saída (referente aos retornos preditos para cada ativo). A função de ativação “sigmoid” foi utilizada na camada intermediária e na camada de saída. O otimizador utilizado para o modelo foi o “RMSProp”. 

Assim como descrito para o otimizador LSTM, no modelo preditivo também foi utilizada a função Dropout() para evitar a ocorrência de overfitting. O modelo preditivo foi capaz de prever o retorno esperado dos ativos para o ano de 2021 com 76.47% de acurácia.

Assim como no caso anterior, foram criados aleatoriamente 1000 possíveis carteiras para construção da fronteira eficiente. A diferença é que agora o índice Sharpe é calculado com base no valor de retorno predito para 2021, e não mais com base no valor médio dos retornos entre 1928 e 2020.

A Figura 3 apresenta a fronteira eficiente determinada a partir da utilização da predição do retorno de cada ativo para o ano de 2021. Os pontos destacados se referem ao método utilizado para determinação do maior índice Sharpe (círculo – teoria de Markowitz e xis – otimizador LSTM).

<img width="645" alt="Captura de Tela 2023-07-17 às 10 43 47" src="https://github.com/testedaisa/markowitz-risco/assets/121257762/4e31b3a2-a82f-4183-b05d-188729df6150">

A Figura 4 apresenta a sugestão de alocação dos ativos correspondente aos dois pontos destacados na fronteira de decisão.

<img width="633" alt="Captura de Tela 2023-07-17 às 10 44 06" src="https://github.com/testedaisa/markowitz-risco/assets/121257762/541a6d71-d70d-40e1-854a-2ea8e0ad6c18">

## Comparando os resultados

### Quanto à utilização de um modelo otimizador baseado em LSTM ao invés da utilização da metodologia clássica de Markowitz

A simulação de N portifólios, envolvendo cálculo de matriz de covariância para determinação do risco e retorno de cada um deles, com o objetivo final de determinar o portfólio com maior índice Sharpe é considerado um processo computacionalmente custoso. Sendo assim, é válida a implementação de um modelo otimizador baseado em LSTM para desempenho dessa tarefa.

O otimizador baseado em LSTM apresentou resultado semelhante à metodologia clássica de Markowitz, o que indica que sua utilização pode ser adotada sem perda de qualidade na determinação do portifólio ótimo.

### Quanto à utilização dos retornos preditos para o ano de 2021 ao invés da utilização do valor médio dos retornos

A utilização do valor médio dos retornos, calculado como sendo a média dos retornos de todos os anos, acaba levando em conta fatores históricos adversos.

No banco de dados disponibilizado, por exemplo, são considerados para o cálculo do valor de retorno médio, os retornos obtidos no ano de 1929, onde os EUA sofreram com a quebra da bolsa de Nova York. Indiscutivelmente, esse fato influenciou os retornos relatados neste ano e, desta forma, esse fator continua impactando a escolha de um portifólio ótimo para o ano de 2021.

Com a utilização deste modelo neural, o portifólio ótimo para 2021 pode ser determinado levando em consideração apenas os acontecimentos dos últimos 5 anos, o que proporcionaria uma análise mais atual para uma tomada de decisão mais assertiva.


## Referências

Cao H.K., Cao H.K., Nguyen B.T. (2020) DELAFO: An Efficient Portfolio Optimization Using Deep Neural Networks. In: Lauw H., Wong RW., Ntoulas A., Lim EP., Ng SK., Pan S. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2020. Lecture Notes in Computer Science, vol 12084. Springer
 
Ta, & Liu, Chuan-Ming & Tadesse. (2020). Portfolio Optimization-Based Stock Prediction Using Long-Short Term Memory Network in Quantitative Trading. Applied Sciences. 10. 437. 10.3390/app10020437.
 
Y. Ma, R. Han and W. Wang, (2020). Prediction-Based Portfolio Optimization Models Using Deep Neural Networks, vol. 8, pp. 115393-115405, IEEE Access

Mahlawat, Sumit and Prabhakar, Utkarsh and Goyal, Nishank and Parth, Praket and Ramamohan, Varun, (2020). A Long Short-Term Memory Approach Towards Stock Selection and Portfolio Optimization.
 







