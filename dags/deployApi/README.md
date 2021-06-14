# Como Rodar

## Configuração do ambiente 

- Python 3.6+
- Flask
- Postman / Insomnia / cURL

### Para usar o cURL 
```
curl --location --request POST 'localhost:5000/api/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "sepal_length": 3.2,
    "sepal_width": 2.2,
    "petal_length": 4.3,
    "petal_width": 3.4
}
```

### Para usar o Postman ou Insomnia 

```
METHOD: Post
BODY: {
    "sepal_length": 3.2,
    "sepal_width": 2.2,
    "petal_length": 4.3,
    "petal_width": 3.4
}
```

### Rodando o projeto

Dentro da pasta *deployApi* usar o comando `python server.py` para iniciar o servidor Flask

Então fazer a execução.

### Rodando o Teste

Arquivo teste usando pacote de teste unitário do Flask para testar as rotas
para iniciar rodar o comando `python test.py` dentro da pasta *deployApi*

