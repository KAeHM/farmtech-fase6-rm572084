# Resultados dos experimentos

## Protocolo

- Hardware da execução: NVIDIA GeForce RTX 3060.
- Semente: `572084`.
- Imagens de teste: 8, nunca usadas no treinamento.
- YOLO customizada: YOLOv8n, imagens de 320 px e pesos iniciais iguais nas duas durações.
- YOLO padrão: pesos COCO, sem novo treinamento, mantendo apenas `dog` e `car`.
- CNN: quatro blocos convolucionais, entrada 128 × 128 e pesos iniciados do zero.
- Avaliação manual de detecção: confiança mínima de 0,25 e IoU mínimo de 0,50.

## Comparação quantitativa

| Abordagem | Acurácia de classe | Precisão | Recall | F1 | mAP50 | mAP50-95 | Treino | Inferência média |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO customizada — 30 épocas | 1,000 | 1,000 | 0,889 | 0,941 | 0,895 | **0,866** | 77,5 s | 15,7 ms |
| YOLO customizada — 60 épocas | 1,000 | 1,000 | 0,889 | 0,941 | 0,896 | 0,852 | 105,7 s | 2,1 ms |
| YOLO padrão COCO | 0,750 | 1,000 | 0,667 | 0,800 | — | — | 0 s | 5,6 ms |
| CNN do zero | 0,750 | 0,833 | 0,750 | 0,733 | — | — | 33,3 s | 4,6 ms |

Os tempos de inferência foram medidos em uma única execução curta e são sensíveis a aquecimento da GPU, cache e tamanho da amostra. Eles não devem ser tratados como benchmark absoluto.

## Interpretação

### YOLO customizada: 30 × 60 épocas

O mAP50 permaneceu praticamente estável ao passar de 30 para 60 épocas. Ao mesmo tempo, o mAP50-95 caiu de 0,866 para 0,852 e o custo de treinamento aumentou. O resultado indica convergência por volta das 30 épocas e ausência de ganho de generalização ao prolongar o treinamento nesta base pequena.

### YOLO padrão

A solução padrão exigiu a menor preparação e nenhum novo treinamento. Ela acertou a classe predominante em seis das oito imagens, mas recuperou somente seis dos nove objetos rotulados. É uma boa linha de base, porém menos adaptada ao recorte selecionado.

### CNN do zero

A CNN acertou seis das oito imagens. Todos os carros foram classificados corretamente, enquanto dois dos quatro cachorros foram confundidos com carros. Ela é simples e rápida, mas não devolve caixas delimitadoras e sofre mais com a pequena quantidade de dados.

## Recomendação

A solução recomendada é a **YOLO customizada treinada por 30 épocas**. Ela localiza os objetos, obteve 100% de acerto da classe predominante no conjunto de teste e apresentou o melhor mAP50-95 com menor custo que o experimento de 60 épocas.

Os dados completos para auditoria estão em:

- [`comparacao_final.json`](resultados/comparacao_final.json);
- [`comparacao_final.csv`](resultados/comparacao_final.csv);
- [`metricas_cnn.json`](resultados/metricas_cnn.json);
- [`avaliacao_yolo_custom_30.json`](resultados/avaliacao_yolo_custom_30.json);
- [`avaliacao_yolo_custom_60.json`](resultados/avaliacao_yolo_custom_60.json);
- [`avaliacao_yolo_padrao.json`](resultados/avaliacao_yolo_padrao.json).
