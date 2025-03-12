FROM python:3.12.1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Instalar o Poetry
RUN pip install poetry

# Desabilitar criação de ambiente virtual pelo Poetry
RUN poetry config virtualenvs.create false

# Limpar cache do Poetry
RUN poetry cache clear --all pypi

# Copiar os arquivos do projeto
COPY . /src
WORKDIR /src

# Instalar dependências do projeto com mais detalhes
RUN poetry install --no-interaction --no-root

# Expôr a porta
EXPOSE 8501

# Definir o comando de inicialização
ENTRYPOINT ["poetry", "run", "streamlit", "run", "src/baseline.py", "--server.port=8501", "--server.address=0.0.0.0"]
