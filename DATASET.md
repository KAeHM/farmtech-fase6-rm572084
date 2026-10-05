# Dataset — origem, preparação e licenças

## Composição

O dataset é um recorte reproduzível da divisão de validação do **Open Images V7**. Ele contém 80 imagens distribuídas igualmente entre as classes `dog` e `car`.

Por classe, a divisão fixa é de 32 imagens para treino, quatro para validação e quatro para teste. Não há repetição de imagens entre as divisões.

## Preparação

O script [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py):

1. obtém o catálogo oficial de classes e as anotações de caixas delimitadoras;
2. seleciona apenas imagens que contenham uma das duas classes do projeto;
3. remove caixas marcadas como grupo, representação gráfica ou visão interna;
4. exige que o maior objeto da classe ocupe ao menos 12% da imagem;
5. faz uma seleção determinística com a semente `572084`;
6. converte as coordenadas para o formato YOLO;
7. grava a origem de cada item em [`manifest.csv`](data/openimages_dog_car/manifest.csv).

Os rótulos resultantes podem ser importados ou revisados no Make Sense AI porque seguem o padrão `classe x_centro y_centro largura altura`, com coordenadas normalizadas.

## Validação

O script [`scripts/validate_dataset.py`](scripts/validate_dataset.py) verifica:

- correspondência entre imagens e arquivos de rótulo;
- integridade dos arquivos de imagem;
- identificadores de classe válidos;
- coordenadas dentro do intervalo de 0 a 1;
- totais esperados de 64 imagens de treino, oito de validação e oito de teste.

A validação foi concluída sem erros. Uma amostra visual está em [`evidencias/dataset_teste_rotulado.jpg`](evidencias/dataset_teste_rotulado.jpg).

## Fontes oficiais

- [Página de download do Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
- [Descrição e licenças do Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [Catálogo de classes delimitáveis](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
- [Anotações de caixas da divisão de validação](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv)

## Licenças e atribuição

Segundo a documentação oficial do Open Images, as anotações são disponibilizadas sob **CC BY 4.0**. As imagens listadas pelo projeto são apresentadas sob **CC BY 2.0**, mas o próprio Open Images recomenda verificar o status de cada imagem antes de reutilização além deste contexto acadêmico.

O `manifest.csv` mantém o identificador e a URL pública de cada imagem para auditoria e atribuição. A licença MIT do código deste repositório não substitui nem modifica as licenças das imagens e anotações.
