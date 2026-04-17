# Módulo responsável por gerenciar as interações de chat.
# Este componente coordena a recuperação de informações de documentos e a geração de respostas
# utilizando um modelo de linguagem grande (LLM) da Groq.

from fastapi import APIRouter, Body, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging

# Importa as funções auxiliares necessárias para o pipeline RAG (Retrieval-Augmented Generation):
#   - generate_embedding: Para converter texto em vetores numéricos.
#   - get_pinecone_index: Para acessar a instância do índice Pinecone.
from routes.utils import generate_embedding, get_pinecone_index, generate_llm_response

# Importa autenticação e função para salvar histórico
from routes.login import get_current_active_user

# Carrega as variáveis de ambiente definidas no arquivo .env do projeto.
load_dotenv()

# Configura o logger específico para este módulo para facilitar o rastreamento de eventos e erros.
logger = logging.getLogger(__name__)

# Cria um APIRouter, que permite organizar rotas relacionadas ao chat de forma modular.
router = APIRouter()

# Define o modelo de dados para a requisição de chat.
# Utiliza Pydantic para validação automática da entrada.
class ChatRequest(BaseModel):
    question: str # O único campo esperado na requisição é a pergunta do usuário.
    selected_document: str = None  #Campo opcional para documento selecionado

@router.post("")
async def send_message(
    request: ChatRequest = Body(...),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint principal para o processamento de mensagens de chat.
    Implementa o fluxo de trabalho de Retrieval-Augmented Generation (RAG):
    1.  A pergunta do usuário é convertida em um embedding vetorial.
    2.  Este embedding é usado para buscar documentos relevantes em um banco de dados vetorial (Pinecone).
    3.  Os trechos de documentos recuperados são combinados com a pergunta original
        para formar um prompt contextualizado para o LLM.
    4.  O modelo de linguagem da Groq processa este prompt e gera uma resposta coerente e informada.
    5.  A resposta do LLM, juntamente com as fontes consultadas, é retornada ao cliente.
    6.  A conversa é automaticamente salva no histórico do usuário autenticado.

    Args:
        request (ChatRequest): Objeto contendo a pergunta do usuário e documento selecionado opcional.
        current_user (dict): Dados do usuário autenticado.

    Returns:
        dict: Um dicionário contendo a resposta gerada (`answer`),
              as fontes dos documentos utilizados (`sources`), um trecho do contexto completo (`context`),
              e informações de debug (`debug_info`).

    Raises:
        HTTPException: Erros HTTP são levantados para cenários como perguntas ausentes,
                       ou falhas na inicialização/acesso a serviços externos (Pinecone, Groq).
    """
    try:
        question = request.question
        selected_document = request.selected_document  
        
        if not question:
            raise HTTPException(status_code=400, detail='A pergunta do usuário não foi fornecida.')

        # Etapa 1: Geração do embedding da pergunta do usuário.
        # Este vetor numérico é a representação semântica da pergunta.
        question_embedding = generate_embedding(question)

        # Etapa 2: Busca de contexto relevante no Pinecone.
        # Verifica se a instância do índice Pinecone está disponível.
        pinecone_index_instance = get_pinecone_index()
        if not pinecone_index_instance:
            raise HTTPException(status_code=500, detail="O índice Pinecone não foi inicializado ou está inacessível. O serviço de busca de documentos está inoperante.")

        #  Busca com filtro opcional por documento
        query_params = {
            "vector": question_embedding,
            "top_k": 10,
            "include_metadata": True
        }
        
        #  Aplica filtro se um documento específico foi selecionado
        if selected_document and selected_document != "all":
            query_params["filter"] = {"filename": selected_document}
            logger.info(f"Busca filtrada para documento: {selected_document}")
        else:
            logger.info("Busca em todos os documentos")

        query_results = pinecone_index_instance.query(**query_params)

        context_parts = [] # Lista para armazenar o conteúdo dos chunks recuperados.
        sources = []       # Lista para armazenar informações das fontes para o frontend.

        # Processa cada resultado (match) retornado pelo Pinecone.
        for match in query_results.matches:
            # Extrai o conteúdo e os metadados do chunk. Um valor padrão é usado se a chave não existir.
            content = match.metadata.get('content', 'Conteúdo do chunk não encontrado.')
            filename = match.metadata.get('filename', 'N/A')
            score = match.score # A pontuação de similaridade do Pinecone.
            
            # Verifica se o conteúdo não está vazio antes de adicionar
            if content.strip():
                # Adiciona identificação do documento para melhor contexto
                context_parts.append(f"[DOCUMENTO: {filename}]\n{content}")
                sources.append({
                    'filename': filename,
                    'score': score,
                    'conteudo': content[:200] + "..." if len(content) > 200 else content # Trecho do conteúdo para exibição como fonte.
                })

        # Sistema de fallback: se temos poucos resultados, busca mais agressivamente
        if len(context_parts) < 5 and (not selected_document or selected_document == "all"):
            logger.info("Poucos chunks encontrados, executando busca expandida")
            
            expanded_query_params = {
                "vector": question_embedding,
                "top_k": 15,
                "include_metadata": True
            }
            
            # Aplica o mesmo filtro se necessário
            if selected_document and selected_document != "all":
                expanded_query_params["filter"] = {"filename": selected_document}
            
            expanded_query = pinecone_index_instance.query(**expanded_query_params)
            
            # Adiciona resultados adicionais sem threshold muito restritivo
            for match in expanded_query.matches[len(context_parts):]:
                content = match.metadata.get('content', '')
                filename = match.metadata.get('filename', 'Documento')
                score = match.score
                
                if content.strip() and len(context_parts) < 12:  # Limita a 12 chunks totais
                    context_parts.append(f"[FONTE: {filename} - Score: {score:.2f}]\n{content}")
                    sources.append({
                        'filename': filename,
                        'score': score,
                        'conteudo': content[:150] + "..."
                    })
        
        # Concatena todos os conteúdos dos chunks relevantes para formar o contexto completo para o LLM.
        context = "\n\n".join(context_parts)

        # Evita payload grande para a Groq (limite de TPM na conta on_demand)
        max_context_chars = int(os.getenv("GROQ_MAX_CONTEXT_CHARS", "8000"))
        if len(context) > max_context_chars:
            logger.info(f"Contexto truncado de {len(context)} para {max_context_chars} caracteres")
            context = context[:max_context_chars]

        # Log detalhado para monitoramento e debug da qualidade da busca
        logger.info(f"Pergunta recebida: {question}")
        logger.info(f"Documento selecionado: {selected_document or 'Todos'}")  
        logger.info(f"Chunks encontrados: {len(context_parts)}")
        logger.info(f"Tamanho do contexto gerado: {len(context)} caracteres")
        if query_results.matches:
            scores = [f'{m.score:.3f}' for m in query_results.matches[:3]]
            logger.info(f"Principais scores de similaridade: {scores}")

        # Etapa 3: Construção do prompt muito flexível e otimizado.
        # O prompt é formatado para instruir o LLM a ser maximamente útil
        # priorizando qualquer informação que possa ajudar o usuário.
        document_context = f" do documento '{selected_document}'" if selected_document and selected_document != "all" else ""
        
        prompt = f"""Você é um assistente especializado em documentos da UFMA (Universidade Federal do Maranhão).

Pergunta do usuário: {question}
{f"Contexto: Respondendo especificamente com base{document_context}" if document_context else ""}

Contexto dos documentos:
{context}

Instruções:
- Use QUALQUER informação relevante do contexto, mesmo que seja parcial
- Se não há resposta exata, forneça informações relacionadas que possam ajudar
- Seja proativo em explicar conceitos relacionados encontrados nos documentos
- Se encontrar procedimentos similares ou regras gerais, mencione-os
- Sempre tente ser útil, mesmo com informações incompletas
- Cite especificamente quais documentos você está consultando
{f"- Foque sua resposta nas informações{document_context}" if document_context else ""}
- Formatação obrigatória da resposta:
    1) Não use títulos Markdown (ex.: ###), tabelas, linhas --- ou emojis.
    2) Responda em texto corrido com, no máximo, 6 bullets simples quando necessário.
    3) Evite repetir o mesmo conteúdo e seja objetivo.
    4) Finalize com "Fontes:" seguido apenas dos nomes dos arquivos usados.

Resposta detalhada:"""
        
        # Etapa 4: Geração da resposta utilizando o modelo de linguagem da Groq.
        try:
            answer = generate_llm_response(
                prompt=prompt,
                max_tokens=900,
                temperature=0.05,
                top_p=0.95,
            )

            logger.info(f"Resposta gerada: {answer[:100]}...")
            
        except Exception as e:
            logger.error(f"Erro ao processar a requisição com o modelo Groq: {e}", exc_info=True)
            answer = f"Ocorreu um erro ao processar a resposta do modelo de linguagem: {str(e)}"
        
        # Etapa 5: Salvar no histórico do usuário
        try:
            # Importa a função para salvar histórico (evita importação circular)
            from routes.history import add_chat_entry
            
            user_email = current_user["email"]
            add_chat_entry(
                user_email=user_email,
                question=question,
                answer=answer,
                sources=sources
            )
            logger.info(f"Conversa salva no histórico para usuário {user_email}")
        except Exception as history_error:
            # Não falha a resposta se houver erro ao salvar histórico
            logger.warning(f"Erro ao salvar no histórico: {history_error}")

        return {
            'answer': answer,
            'sources': sources,
            'context': context[:800] + "..." if len(context) > 800 else context, # Trecho maior do contexto para visualização.
            'selected_document': selected_document,  #  Retorna documento selecionado
            'debug_info': {  # Informações de debug para monitoramento da qualidade
                'chunks_found': len(context_parts),
                'context_length': len(context),
                'similarity_scores': [f"{s['score']:.3f}" for s in sources[:5]],
                'total_results': len(query_results.matches),
                'document_filter': selected_document or 'all'  
            }
        }
        
    except Exception as e:
        logger.error(f"Erro interno ao processar a mensagem do chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")