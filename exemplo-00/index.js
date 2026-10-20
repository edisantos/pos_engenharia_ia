import * as tf from '@tensorflow/tfjs';

async function trinarModelo(inputXs, outputYs) {
    // Criamos o modelo sequencial
    const model = tf.sequential(); 
    
    //Primeira camada de rede
    // entrada de 7 posicoes (idade, cor e localização) e saída de 10 posicoes
    model.add(tf.layers.dense({inputShape: [7], units: 80, activation: 'relu'})); // Camada de entrada

    //saida da primeira camada de 80 posicoes e entrada da segunda camada de 80 posicoes
    model.add(tf.layers.dense({units: 3, activation: 'softmax'})); // Camada de saída

    // Compilamos o modelo
    // Otimizador Adam, função de perda categoricalCrossentropy e métrica de acurácia
    // categoricalCrossentropy é usada para problemas de classificação com múltiplas classes
    // softmax é usada na camada de saída para normalizar os valores de saída em uma distribuição de probabilidade
    // A função de perda categoricalCrossentropy mede a diferença entre a distribuição de probabilidade prevista pelo modelo e a distribuição de probabilidade real (labels)
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(inputXs, outputYs, {
        verbose: 0, // 0 para não exibir logs de treinamento
        epochs: 100, // Número de épocas (quantas vezes o modelo verá todo o dataset)
        batchSize: 1, // Tamanho do lote (quantos exemplos serão processados antes de atualizar os pesos)'
        shuffle: true, // Embaralha os dados a cada época para melhorar a generalização do modelo
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Treinamos o modelo com os dados de entrada e saída
// O modelo é retornado após o treinamento
//quanto mais dados de entrada e saída, melhor será o modelo treinado
const model = trinarModelo(inputXs, outputYs);

//inputXs.print();
//outputYs.print();