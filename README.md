# ConsultAI — Consulta inteligente a documentos com LLM e RAG

Sistema de perguntas e respostas em linguagem natural sobre documentos institucionais, com **citação das fontes**. O caso de uso inicial é a **Resolução nº 1892-CONSEPE/2019**, que aprova as Normas Regulamentadoras dos Cursos de Graduação da UFMA, mas o administrador pode enviar outros PDFs.

Software com registro no INPI (BR512025005061-5). Desenvolvido em grupo na disciplina de Projeto de Desenvolvimento de Software (UFMA, 2025.1).

<!-- Adicione aqui um print ou GIF do chat respondendo uma pergunta, com as fontes visíveis -->

## Como funciona

```
PDF → extração de texto (+ OCR para páginas digitalizadas) → chunks → embeddings → Pinecone
                                                                                    │
Pergunta → embedding → busca semântica (top-k) → reordenação → prompt com contexto → LLM (Groq) → resposta + fontes
```

1. **Ingestão (administrador):** o PDF é lido com PyMuPDF/pdfplumber. Páginas sem texto passam por OCR (Tesseract). O texto é dividido em chunks com LangChain e convertido em embeddings (Sentence Transformers).
2. **Indexação:** os vetores e os metadados (documento, ordem do chunk) ficam no Pinecone.
3. **Consulta:** a pergunta vira embedding, o sistema busca os trechos mais próximos, reordena por relevância e monta o contexto dentro de um limite de tamanho.
4. **Resposta:** o LLM (Groq) responde apenas com base no contexto recuperado, e o sistema devolve as fontes que de fato foram enviadas ao modelo.

## Funcionalidades

- **Chat** com respostas fundamentadas e citação de fontes, com escolha do documento a consultar
- **Autenticação** por e-mail/senha (JWT) e **login com Google**
- **Perfis de acesso:** usuário comum e administrador, com fluxo de solicitação para virar admin e painel de gestão de administradores
- **Gestão de documentos (admin):** upload, download, remoção e sincronização de metadados
- **Histórico** de consultas por usuário
- **Avaliação das respostas** com botões de feedback

## Tecnologias

| Camada | Tecnologias |
|---|---|
| Backend | Python, FastAPI, Uvicorn/Gunicorn |
| LLM | Groq API |
| Busca semântica | Pinecone, Sentence Transformers |
| Processamento de documentos | PyMuPDF, pdfplumber, pypdf, Tesseract OCR (pytesseract), LangChain Text Splitters |
| Autenticação | JWT (python-jose), bcrypt, Google OAuth |
| Frontend | React, Axios, Lucide |
| Infraestrutura | Docker, Docker Compose, Nginx, PostgreSQL |

## Como executar (Docker)

Pré-requisitos: Docker e Docker Compose, e contas gratuitas na Groq e na Pinecone.

```bash
git clone https://github.com/Hissa-Barbara/G4_CONSULTA_DOCUMENTOS.git
cd G4_CONSULTA_DOCUMENTOS
cp .env.example .env        # no PowerShell: Copy-Item .env.example .env
```

Edite o `.env` e preencha, no mínimo:

| Variável | Para quê |
|---|---|
| `GROQ_API_KEY`, `GROQ_MODEL`, `LLM_PROVIDER` | LLM (use `LLM_PROVIDER=groq`) |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_HOST` | Índice vetorial |
| `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | Modelo de embeddings e a dimensão do índice |
| `SECRET_KEY` | Chave de assinatura do JWT (use um valor longo e aleatório) |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Apenas se for usar login com Google |

```bash
docker compose up -d --build
docker compose ps           # backend, frontend e db devem estar "Up"
```

- Frontend: http://localhost:10000
- API (Swagger): http://localhost:8000/docs
- Logs: `docker compose logs -f backend`

Detalhes de cada parte: [Backend](Codigo/BackEnd/README.md) · [Frontend](Codigo/FrontEnd/README.md)

> **Segurança:** nunca faça commit do `.env`. Se uma chave for exposta, revogue e gere outra.

## Estrutura do repositório

```
Codigo/
├── BackEnd/     API FastAPI (rotas de login, chat, histórico, admin e processamento de documentos)
└── FrontEnd/    Interface React (chat, histórico e painéis de administração)
Diagramas/       Casos de uso, atividade, estado, classe e sequência (PlantUML)
Documentos/      Documentação do projeto
Gestão/          Material de gestão do projeto (board)
docker-compose.yml
```

## Endpoints principais

| Rota | Função |
|---|---|
| `POST /chat` | Pergunta ao sistema e recebe resposta com fontes |
| `POST /login/`, `/login/google`, `/login/register` | Autenticação |
| `GET/POST /history` | Histórico de consultas |
| `POST /admin/upload`, `GET /admin/documents`, `DELETE /admin/document/{id}` | Gestão de documentos |
| `POST /admin-requests/request`, `/admin-requests/requests/{id}/review` | Solicitação e aprovação de perfil admin |

A documentação completa e interativa está no Swagger (`/docs`).

## Equipe

Euderlan Freire da Silva Abreu, Hissa Bárbara Oliveira, Yasmin Serejo Lima, Anderson Rodrigo Diniz Oliveira e Yuram Almeida Santos.

Orientação: Prof. Dr. Thales Levi Azevedo Valente (UFMA).

### Minha contribuição (Hissa Bárbara)

<!-- Ajuste conforme o que você de fato fez; os itens abaixo vêm do histórico de commits -->

- Desenvolvimento full stack (frontend React e backend FastAPI)
- Correções e melhorias na seleção de documentos e no resumo salvo dos documentos
- Processamento visual de documentos e melhoria da busca documental
- Diagramas de sequência, atividade e estado (PlantUML) e documentação do projeto

## Limitações e próximos passos

- O sistema responde apenas com base nos documentos enviados e não substitui a consulta oficial à universidade.
- Próximos passos: avaliação automática da qualidade das respostas (usando o feedback dos usuários) e suporte a mais tipos de documento.

## Licença

Licença MIT (adicione o arquivo `LICENSE` na raiz do repositório). Projeto acadêmico da Universidade Federal do Maranhão (UFMA).
