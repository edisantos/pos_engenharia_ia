# Treinamento de uma Rede Neural com TensorFlow.js

## 1. Objetivo

Este projeto implementa uma rede neural utilizando a biblioteca **TensorFlow.js** para realizar uma tarefa de classificação multiclasse. O modelo recebe dados numéricos que representam características de pessoas e aprende a classificá-las em três categorias: `premium`, `medium` e `basic`.

O treinamento é realizado utilizando exemplos de entrada e suas respectivas categorias esperadas, permitindo que a rede ajuste seus parâmetros para minimizar os erros de previsão.

## 2. Arquitetura da Rede Neural

A rede neural utiliza uma arquitetura do tipo *Sequential*, na qual as camadas são organizadas em sequência.

### Camada de entrada e camada oculta

```javascript
model.add(tf.layers.dense({
    inputShape: [7],
    units: 80,
    activation: 'relu'
}));
```

Essa camada recebe um vetor com **7 características numéricas** e possui 80 neurônios.

As características utilizadas no exemplo são:

1. Idade normalizada.
2. Indicador da cor azul.
3. Indicador da cor vermelha.
4. Indicador da cor verde.
5. Indicador da localização São Paulo.
6. Indicador da localização Rio de Janeiro.
7. Indicador da localização Curitiba.

As cores e localizações são representadas numericamente por meio da codificação *one-hot encoding*. A idade é normalizada para facilitar o processamento pela rede.

A função de ativação `ReLU` (Rectified Linear Unit) introduz não linearidade ao modelo, permitindo que ele aprenda relações mais complexas entre as características de entrada.

### Camada de saída

```javascript
model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
}));
```

A camada de saída possui **3 neurônios**, um para cada categoria que o modelo deve prever:

- `premium`
- `medium`
- `basic`

A função `softmax` transforma as saídas em valores que representam uma distribuição de probabilidades entre as três classes. A classe com maior valor previsto pode ser selecionada como a classificação final.

## 3. Compilação do Modelo

```javascript
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});
```

A compilação define como a rede será treinada.

- **Adam:** otimizador responsável por atualizar os pesos e vieses da rede durante o treinamento.
- **categoricalCrossentropy:** função de perda utilizada para classificação multiclasse com rótulos codificados em *one-hot encoding*.
- **accuracy:** métrica que mede a proporção de classificações corretas.

## 4. Dados de Treinamento

Os dados de entrada são representados por uma matriz de sete características por pessoa:

```javascript
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0],
    [0,    0, 1, 0, 0, 1, 0],
    [1,    0, 0, 1, 0, 0, 1]
];
```

Cada linha representa uma pessoa, e cada coluna representa uma característica.

As saídas esperadas são definidas por meio de *one-hot encoding*:

```javascript
const tensorLabels = [
    [1, 0, 0], // premium
    [0, 1, 0], // medium
    [0, 0, 1]  // basic
];
```

Cada linha identifica a categoria correta para o respectivo exemplo de entrada.

Os dados são convertidos em tensores bidimensionais com `tf.tensor2d()`, formato utilizado pelo TensorFlow.js para alimentar o modelo.

## 5. Processo de Treinamento

O treinamento é executado por meio do método `model.fit()`:

```javascript
await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    batchSize: 1,
    shuffle: true
});
```

Os principais parâmetros são:

- **epochs: 100:** o modelo percorre o conjunto de treinamento até 100 vezes.
- **batchSize: 1:** os pesos são atualizados após o processamento de cada exemplo.
- **shuffle: true:** os exemplos são embaralhados a cada época.
- **verbose: 0:** desativa o registro padrão detalhado do treinamento.

O callback `onEpochEnd` permite acompanhar a evolução da função de perda e da acurácia ao final de cada época.

## 6. Fluxo de Funcionamento

1. Preparação e normalização dos dados.
2. Codificação das características categóricas.
3. Conversão dos dados em tensores.
4. Definição das camadas da rede neural.
5. Compilação do modelo.
6. Treinamento com os exemplos de entrada e saída.
7. Utilização do modelo treinado para realizar previsões.

## 7. Considerações

O conjunto de dados apresentado possui apenas três exemplos. Portanto, ele serve como demonstração didática da construção e do treinamento de uma rede neural, mas não é suficiente para validar a capacidade de generalização do modelo.

Para obter resultados confiáveis, é necessário utilizar mais exemplos representativos, avaliar o desempenho com dados não utilizados no treinamento e verificar se as características fornecidas realmente permitem distinguir as categorias.

Além disso, o número de neurônios, as épocas e os demais parâmetros são escolhas iniciais que podem ser ajustadas conforme os resultados dos experimentos.


onEpochEnd: (epoch, logs) => {
    const accuracy = logs.acc ?? logs.accuracy ?? 0;

    console.log(
        `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${accuracy.toFixed(4)}`
    );
}