# TRABALHO PRÁTICO 01

## DataOps e Implantação de Aplicações de Machine Learning

Professor: Jean Carlos Alves

Alunos: Tiago Henrique Pereira Fonseca

### Objetivo: Orquestrar todo o fluxo de processamento necessários para execução de um modelo de machine learning utilizando o airflow.

#### Etapas a serem desenvolvidas:

- Criar uma task etl responsável pelo download do arquivo iris.txt disponível no [link](https://drive.google.com/uc?export=download&id=1rZgVuwYon_3QogTr0-v480PRpi-2l1-v)

- Criar uma task pre-processamento para validar se os dados se encontram no formato correto:
    1. sepal_length range(4.3, 7.9)
    2. sepal_width range(2.0, 4.4)
    3. petal_length range(1.0, 6.9)
    4. petal_width range(0.1, 2.5)
    5. classEncoder range(0, 2)
    6. class ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

OBS: Todos os dados que não se encontram no formato ou range correto devem ser excluídos do dataset.

OBS2: Criar um arquivo com os registros removidos contendo a mensagem do erro conforme exemplo:

- sepal_length,sepal_width,petal_length,petal_width,class,classEncoder,messageError 5.1,3.5,1.4,0.2,Iris-setosa,0, sepal_width maior que 4.4

#### Criar uma task de grupo que contenha:
- Execução de 1 algoritmo de ML para o dataset Iris
- Execução de 1 algoritmo de ML para o dataset Iris

OBS1: O processamento deve ser realizado de forma paralela

OBS2: Utilizar o MLFlow para gerenciar os modelos criados


### Conclusão

Modelo pickle selecionado foi o do RandomForest, que para nosso pequeno conjunto de dados e usando os hiperparametros ajustados, ficou com um accuracy_score maior que o gerado pelo algoritmo de random forest com seus hiperparametros diferentes.

---

# Parte 2

Arquivo *deployApi* contém modelo .pkl a ser usado em atividade 2

Para mais informações link para documentação da segunda parte [aqui](deployApi/README.md)