# PDF Knowledge Base - RAG with Gemini & PGVector

[Portugues](#portugues) | [English](#english)

---

<a id="portugues"></a>
## Portugues

Aplicacao de Retrieval-Augmented Generation (RAG) que ingere documentos PDF, armazena embeddings no PostgreSQL com PGVector e responde perguntas via Gemini (Google AI).

### Arquitetura

```
PDF -> PyPDFLoader -> Chunks -> Gemini Embeddings -> PGVector (PostgreSQL)
                                                          |
Pergunta -> Embedding -> Similarity Search -> Contexto -> Gemini LLM -> Resposta
```

### Pre-requisitos

- Python 3.12+
- Docker e Docker Compose
- Chave de API do Google AI (Gemini)

### Instalacao

1. Clone o repositorio e entre na pasta do projeto:

```bash
cd mba-ia-desafio-ingestao-busca
```

2. Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
```

3. Instale as dependencias:

```bash
pip install -r requirements.txt
```

4. Configure as variaveis de ambiente. Copie o `.env.example` para `.env` e preencha:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=sua-chave-aqui
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=gemini_collection
PDF_PATH=/caminho/completo/para/document.pdf
```

5. Suba o banco de dados:

```bash
docker compose up -d
```

### Uso

#### 1. Ingestao do PDF

```bash
python src/ingest.py
```

O script divide o PDF em chunks, gera embeddings com o Gemini e armazena no PGVector. Os documentos sao enviados em 3 lotes com intervalo de 60 segundos entre cada um para respeitar os limites de rate da API.

#### 2. Busca interativa

```bash
python src/search.py
```

Ao iniciar, voce escolhe o template de prompt. Use as setas para navegar e ENTER para confirmar. Pressione ESC para sair a qualquer momento.

### Templates

#### Default

Template simples. O LLM recebe o contexto e responde com base nele. Se a informacao nao estiver no contexto, responde que nao tem informacoes suficientes.

![Template Default em funcionamento](images/default.gif)

#### Melhorado (com protecoes)

Alem das regras do Default, inclui:

- Descricao da aplicacao para o LLM (responde quando perguntam "o que voce faz?")
- Bloqueio explicito de perguntas comparativas, rankings, agregacoes e contagens
- Exemplos positivos e negativos para guiar o comportamento

![Template Melhorado em funcionamento](images/melhorado.gif)

### Limitacoes

Esta aplicacao usa RAG com busca por similaridade semantica. Isso significa que:

1. **A busca retorna apenas os trechos mais relevantes, nao todos os dados.** Perguntas que precisam de todos os registros para serem corretas (como "qual empresa faturou mais?") podem gerar respostas incompletas ou erradas.

2. **Tabelas sao mal interpretadas.** O PDF e dividido em blocos de texto. Tabelas perdem sua estrutura (cabecalhos, colunas, linhas) e viram texto corrido fragmentado.

3. **Perguntas agregadas nao funcionam bem.** Somas, medias, rankings e contagens dependem de acesso completo aos dados. A busca por similaridade nao garante isso.

4. **A qualidade depende do chunk.** Se o chunk size for muito grande, a busca fica imprecisa. Se for muito pequeno, o contexto perde sentido.

5. **Rate limits da API.** O plano gratuito do Gemini tem limites de requisicoes por minuto. A ingestao usa batching com delays para contornar isso.

---

<a id="english"></a>
## English

Retrieval-Augmented Generation (RAG) application that ingests PDF documents, stores embeddings in PostgreSQL with PGVector, and answers questions using Gemini (Google AI).

### Architecture

```
PDF -> PyPDFLoader -> Chunks -> Gemini Embeddings -> PGVector (PostgreSQL)
                                                          |
Question -> Embedding -> Similarity Search -> Context -> Gemini LLM -> Answer
```

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Google AI API key (Gemini)

### Setup

1. Clone the repository and navigate to the project folder:

```bash
cd mba-ia-desafio-ingestao-busca
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables. Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your-key-here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=gemini_collection
PDF_PATH=/full/path/to/document.pdf
```

5. Start the database:

```bash
docker compose up -d
```

### Usage

#### 1. PDF Ingestion

```bash
python src/ingest.py
```

The script splits the PDF into chunks, generates embeddings with Gemini, and stores them in PGVector. Documents are sent in 3 batches with 60-second intervals to respect API rate limits.

#### 2. Interactive Search

```bash
python src/search.py
```

On startup, you choose a prompt template. Use arrow keys to navigate and ENTER to confirm. Press ESC to exit at any time.

### Templates

#### Default

Simple template. The LLM receives the context and answers based on it. If the information is not in the context, it replies that it does not have enough information.

![Default template in action](images/default.gif)

#### Melhorado (Enhanced)

In addition to the Default rules, it includes:

- Application description for the LLM (answers when asked "what do you do?")
- Explicit blocking of comparative questions, rankings, aggregations, and counts
- Positive and negative examples to guide behavior

![Enhanced template in action](images/melhorado.gif)

### Limitations

This application uses RAG with semantic similarity search. This means:

1. **The search returns only the most relevant excerpts, not all data.** Questions that require all records to be correct (such as "which company had the highest revenue?") may produce incomplete or wrong answers.

2. **Tables are poorly interpreted.** The PDF is split into text blocks. Tables lose their structure (headers, columns, rows) and become fragmented plain text.

3. **Aggregate questions do not work well.** Sums, averages, rankings, and counts depend on complete data access. Similarity search does not guarantee this.

4. **Quality depends on chunk size.** If the chunk size is too large, the search becomes imprecise. If too small, the context loses meaning.

5. **API rate limits.** The free Gemini plan has per-minute request limits. Ingestion uses batching with delays to work around this.

