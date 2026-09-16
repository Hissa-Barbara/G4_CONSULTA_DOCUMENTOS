# Guia Rapido para Rodar o Projeto (Branch `teste`)

Este arquivo explica o passo a passo para baixar, configurar e executar o projeto localmente com Docker.

## 1) Clonar o repositorio

```bash
git clone https://github.com/Hissa-Barbara/G4_CONSULTA_DOCUMENTOS.git
cd G4_CONSULTA_DOCUMENTOS
```

## 2) Ir para a branch de teste

```bash
git fetch origin
git switch teste
```

Se a branch local ainda nao existir:

```bash
git switch -c teste --track origin/teste
```

## 3) Criar e configurar o arquivo `.env`

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

No Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Depois, edite o `.env` e preencha os campos necessarios.

### Campos importantes

- `GROQ_API_KEY`: sua chave da Groq
- `GROQ_MODEL`: modelo a usar (exemplo: `qwen/qwen3-32b`)
- `LLM_PROVIDER`: `groq` para usar Groq primeiro
- `PINECONE_API_KEY`: chave da Pinecone
- `PINECONE_INDEX_NAME`: nome do indice
- `PINECONE_HOST`: host do indice
- `SECRET_KEY`: chave JWT da aplicacao
- `GOOGLE_CLIENT_ID` e `GOOGLE_CLIENT_SECRET`: se for usar login Google

Observacao: os valores de banco no `.env.example` ja estao prontos para o Docker Compose local.

## 4) Subir os containers

```bash
docker compose up -d --build
```

## 5) Verificar se subiu tudo

```bash
docker compose ps
```

Esperado:
- `backend` em `Up`
- `frontend` em `Up`
- `db` em `Up (healthy)`

## 6) Acessar o sistema

- Frontend: http://localhost:10000
- Backend (Swagger): http://localhost:8000/docs

## 7) Ver logs se algo falhar

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f db
```

## 8) Atualizar codigo depois

Quando houver novas alteracoes no GitHub:

```bash
git switch teste
git pull
docker compose up -d --build
```

## 9) Boas praticas

- Nunca subir `.env` para o repositorio.
- Rotacionar chaves se alguma credencial for exposta.
- Fazer commit somente do que e codigo/config segura.

-----------------







# Sistema Inteligente de Consultas via LLM - UFMA

Um sistema de consultas inteligente baseado em LLM (Large Language Model) desenvolvido para facilitar o acesso às informações da RESOLUÇÃO Nº 1892-CONSEPE de 28 de junho de 2019, que aprova as Normas Regulamentadoras dos Cursos de Graduação da Universidade Federal do Maranhão.

## 📋 Descrição do Projeto

Este sistema oferece uma interface de chat inteligente que permite aos usuários realizarem consultas em linguagem natural sobre o conteúdo da resolução acadêmica da UFMA. Através de processamento de linguagem natural, o sistema interpreta perguntas e retorna respostas precisas com citação das fontes específicas.

## ✨ Funcionalidades Principais

### 🔐 1. Sistema de Autenticação
- Login seguro com diferentes níveis de acesso
- Controle de permissões (usuário comum/administrador)
- Gestão segura de credenciais

### 💬 2. Consultas via LLM
- Interface de chat intuitiva
- Processamento de perguntas em linguagem natural
- Respostas precisas com citação de fontes
- Foco na RESOLUÇÃO Nº 1892-CONSEPE

### 📚 3. Histórico de Consultas
- Visualização de perguntas anteriores
- Histórico personalizado por usuário
- Facilita continuidade nas pesquisas acadêmicas

### ⚙️ 4. Gestão de Documentos (Administrador)
- Upload de versões atualizadas da resolução
- Edição e remoção de documentos
- Processamento automático de arquivos PDF
- Acesso restrito a administradores

### ⭐ 5. Sistema de Avaliação
- Botões de feedback para qualidade das respostas
- Opção para reportar erros
- Análise contínua para melhoria do sistema

### 🚪 6. Logout Seguro
- Encerramento seguro da sessão
- Limpeza adequada dos dados de autenticação
- Redirecionamento para tela de login

## 🛠️ Tecnologias Utilizadas

- **Backend**: [Especificar tecnologias utilizadas]
- **Frontend**: [Especificar tecnologias utilizadas]
- **LLM**: [Especificar modelo de linguagem utilizado]
- **Base de Dados**: [Especificar banco de dados]
- **Processamento de PDF**: [Especificar biblioteca utilizada]

## 🚀 Como Executar

```bash
# Clone o repositório
# Segue link do Readme BackEnd
https://github.com/Euderlan/G4_CONSULTA_DOCUMENTOS/blob/main/Codigo/BackEnd/README.md
#Segue link do Readme FrontEnd
https://github.com/Euderlan/G4_CONSULTA_DOCUMENTOS/blob/main/Codigo/FrontEnd/README.md
```

## 📖 Como Usar

1. **Login**: Acesse o sistema com suas credenciais
2. **Fazer Pergunta**: Digite sua consulta sobre a resolução no chat
3. **Visualizar Resposta**: Receba a resposta com citação da fonte
4. **Avaliar**: Use os botões de feedback para avaliar a qualidade
5. **Histórico**: Acesse suas consultas anteriores quando necessário

## 🎯 Público-Alvo

- Estudantes da UFMA
- Professores e coordenadores
- Funcionários administrativos
- Pesquisadores acadêmicos

## 📄 Documentação

O sistema é baseado na **RESOLUÇÃO Nº 1892-CONSEPE** de 28 de junho de 2019, que estabelece as Normas Regulamentadoras dos Cursos de Graduação da Universidade Federal do Maranhão.

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

## 📞 Suporte

Para dúvidas ou suporte técnico, entre em contato através dos canais oficiais da UFMA ou abra uma issue neste repositório.

---

## 📝 Reconhecimentos e Direitos Autorais

**@autor**: Euderlan Freire da Silva Abreu,
            Hissa Bárbara Oliveira,
            Yasmin Serejo Lima,
            Anderson Rodrigo Diniz Oliveira,
            Yuram Almeida Santos       
**@contato**: [Seus Emails - se quiserem]  
**@data última versão**: 12 de junho de 2025  
**@versão**: 1.0  
**@outros repositórios**: [URLs - apontem para os seus Gits AQUI]  
**@Agradecimentos**: Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.

### Copyright/License

Este material é resultado de um trabalho acadêmico para a disciplina **PROJETO DE DESENVOLVIMENTO DE SOFTWARE**, sob a orientação do professor Dr. **THALES LEVI AZEVEDO VALENTE**, semestre letivo 2025.1, curso Engenharia da Computação, na Universidade Federal do Maranhão (UFMA). 

Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da Licença MIT, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito.

### Licença MIT

Licenciado sob a Licença MIT. Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:

Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software.

**O SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA. EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.**

Para mais informações sobre a Licença MIT: https://opensource.org/licenses/MIT

---

**Desenvolvido com ❤️ para UFMA**
