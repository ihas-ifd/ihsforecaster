# 📊 IHS Forecaster

IHS Forecaster é um projeto de previsão utilizando **Python**, **Poetry** e **Streamlit**.

## 🚀 Como rodar o projeto

### 1️⃣ Clone o repositório  
```bash
git clone https://github.com/ihas-ifd/ihsforecaster.git
cd ihsforecaster
```

2️⃣: Construa a imagem Docker
```bash
docker build -t forecaster-image .
```
3️⃣ Rode o contêiner

```bash
docker run -d -p 8501:8501 --name forecaster-container forecaster-image
```
Agora, acesse o aplicativo em http://localhost:8501 🚀