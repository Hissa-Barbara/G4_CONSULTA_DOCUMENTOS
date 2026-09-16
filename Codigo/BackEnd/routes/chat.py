# Módulo responsável por gerenciar as interações de chat.
# Este componente coordena a recuperação de informações de documentos e a geração de respostas
# utilizando um modelo de linguagem grande (LLM) da Groq.

from fastapi import APIRouter, Body, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
import os
import logging
import re
import unicodedata

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

INTERNAL_EXPLANATION_PATTERNS = (
    r"\bocr\b",
    r"reconhecimento [óo]ptico de caracteres",
    r"\bescanead[oa]s?\b",
    r"qualidade (?:da )?(?:digitaliza[çc][ãa]o|leitura|extra[çc][ãa]o)",
    r"texto (?:extra[íi]do|reconhecido|corrompido|ileg[íi]vel)",
    r"\bchunks?\b",
    r"\bpinecone\b",
    r"modelo de linguagem",
    r"processamento (?:do sistema|interno)",
)


def _remove_internal_explanations(answer: str) -> str:
    """Impede que detalhes técnicos internos sejam exibidos ao usuário."""
    if not answer:
        return "Não foi possível gerar uma resposta agora. Tente novamente."

    pattern = re.compile(
        "|".join(INTERNAL_EXPLANATION_PATTERNS),
        flags=re.IGNORECASE,
    )
    public_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", answer.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        public_sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip() and not pattern.search(sentence)
        ]
        if public_sentences:
            public_paragraphs.append(" ".join(public_sentences))

    if not public_paragraphs:
        return "Não encontrei essa informação no documento consultado."
    return "\n\n".join(public_paragraphs)


SEARCH_STOP_WORDS = {
    "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em",
    "essa", "esse", "esta", "este", "eu", "ha", "o", "os", "para", "por",
    "qual", "quais", "que", "tem", "uma", "um",
}


def _normalized_search_terms(text: str) -> set[str]:
    """Obtém termos úteis para complementar a similaridade semântica."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        term
        for term in re.findall(r"[a-z0-9]+", normalized.lower())
        if len(term) >= 3 and term not in SEARCH_STOP_WORDS
    }


def _rank_context_parts(question: str, parts: list[str], sources: list[dict]):
    """Combina correspondência lexical e score vetorial, preservando os pares."""
    question_terms = _normalized_search_terms(question)
    ranked = []
    for position, (part, source) in enumerate(zip(parts, sources)):
        content_terms = _normalized_search_terms(part)
        lexical_matches = len(question_terms & content_terms)
        lexical_ratio = lexical_matches / max(1, len(question_terms))
        ranked.append((lexical_ratio, source.get("score", 0.0), -position, part, source))

    ranked.sort(reverse=True, key=lambda item: item[:3])
    return [item[3] for item in ranked], [item[4] for item in ranked]


def _build_context_with_budget(
    parts: list[str], sources: list[dict], max_chars: int
) -> tuple[str, list[str], list[dict]]:
    """Monta o contexto e mantém somente as fontes realmente enviadas ao LLM."""
    selected_parts = []
    selected_sources = []
    used_chars = 0
    separator_size = 2

    for part, source in zip(parts, sources):
        remaining = max_chars - used_chars - (separator_size if selected_parts else 0)
        if remaining <= 0:
            break
        if len(part) <= remaining:
            selected_parts.append(part)
            selected_sources.append(source)
            used_chars += len(part) + (separator_size if len(selected_parts) > 1 else 0)
        elif not selected_parts:
            selected_parts.append(part[:remaining])
            selected_sources.append(source)
            used_chars += remaining

    return "\n\n".join(selected_parts), selected_parts, selected_sources


def _select_public_sources(
    question: str,
    parts: list[str],
    sources: list[dict],
    selected_document: str | None,
) -> list[dict]:
    """Escolhe fontes de suporte, em vez de listar todos os candidatos."""
    by_filename = {}
    question_terms = _normalized_search_terms(question)

    for part, source in zip(parts, sources):
        filename = source["filename"]
        matches = len(question_terms & _normalized_search_terms(part))
        candidate = {
            "filename": filename,
            "matches": matches,
            "score": source.get("score", 0.0),
        }
        previous = by_filename.get(filename)
        if previous is None or (matches, candidate["score"]) > (
            previous["matches"], previous["score"]
        ):
            by_filename[filename] = candidate

    ranked = sorted(
        by_filename.values(),
        key=lambda item: (item["matches"], item["score"]),
        reverse=True,
    )
    if not ranked:
        return []

    if selected_document and selected_document != "all":
        return [{"filename": ranked[0]["filename"]}]

    asks_for_multiple_documents = bool(re.search(
        r"\b(compare|comparar|diferen[çc]as?|documentos|arquivos|fontes|todos)\b",
        question.lower(),
    ))
    if asks_for_multiple_documents:
        return [{"filename": item["filename"]} for item in ranked[:5]]

    best_match_count = ranked[0]["matches"]
    if best_match_count >= 2:
        supported = [item for item in ranked if item["matches"] == best_match_count]
        return [{"filename": item["filename"]} for item in supported[:3]]

    # Quando não há termos literais suficientes, usa a melhor fonte semântica.
    return [{"filename": ranked[0]["filename"]}]

# Define o modelo de dados para a requisição de chat.
# Utiliza Pydantic para validação automática da entrada.
class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str # O único campo esperado na requisição é a pergunta do usuário.
    selected_document: str = None  #Campo opcional para documento selecionado
    conversation_history: List[ConversationMessage] = Field(default_factory=list)

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
        recent_history = request.conversation_history[-6:]
        
        if not question:
            raise HTTPException(status_code=400, detail='A pergunta do usuário não foi fornecida.')

        # Etapa 1: Geração do embedding da pergunta do usuário.
        # Este vetor numérico é a representação semântica da pergunta.
        history_for_search = "\n".join(
            f"{message.role}: {message.content[:500]}"
            for message in recent_history
        )
        # Perguntas completas são buscadas sozinhas para que assuntos antigos
        # não desviem o embedding. O histórico só ajuda em continuações curtas.
        normalized_question = question.strip().lower()
        referential_question = (
            len(normalized_question.split()) <= 5
            or bool(re.search(
                r"\b(isso|isto|ele|ela|esse|essa|desse|dessa|dele|dela)\b",
                normalized_question,
            ))
        )
        semantic_question = (
            f"Conversa anterior:\n{history_for_search}\n\nPergunta atual: {question}"
            if history_for_search and referential_question else question
        )
        question_embedding = generate_embedding(semantic_question)

        # Etapa 2: Busca de contexto relevante no Pinecone.
        # Verifica se a instância do índice Pinecone está disponível.
        pinecone_index_instance = get_pinecone_index()
        if not pinecone_index_instance:
            raise HTTPException(status_code=503, detail="O serviço está temporariamente indisponível.")

        #  Busca com filtro opcional por documento
        has_document_filter = bool(selected_document and selected_document != "all")
        global_candidate_count = int(os.getenv("PINECONE_GLOBAL_CANDIDATES", "50"))
        query_params = {
            "vector": question_embedding,
            "top_k": 10 if has_document_filter else global_candidate_count,
            "include_metadata": True
        }
        
        #  Aplica filtro se um documento específico foi selecionado
        if has_document_filter:
            query_params["filter"] = {"filename": selected_document}
            logger.info(f"Busca filtrada para documento: {selected_document}")
        else:
            logger.info("Busca em todos os documentos")

        query_results = pinecone_index_instance.query(**query_params)

        context_parts = [] # Lista para armazenar o conteúdo dos chunks recuperados.
        sources = []       # Lista para armazenar informações das fontes para o frontend.
        seen_chunks = set()

        def add_match(match):
            """Adiciona um resultado sem repetir o mesmo chunk nas duas etapas."""
            content = match.metadata.get('content', '')
            filename = match.metadata.get('filename', 'N/A')
            chunk_order = match.metadata.get('chunk_order')
            chunk_key = (
                filename,
                chunk_order if chunk_order is not None else content[:200],
            )
            if not content.strip() or chunk_key in seen_chunks:
                return
            seen_chunks.add(chunk_key)
            context_parts.append(f"[DOCUMENTO: {filename}]\n{content}")
            sources.append({
                'filename': filename,
                'score': match.score,
                'conteudo': content[:200] + "..." if len(content) > 200 else content,
            })

        # Processa cada resultado (match) retornado pelo Pinecone.
        for match in query_results.matches:
            add_match(match)

        # Na busca global, a primeira consulta localiza os documentos prováveis.
        # A segunda recupera os demais chunks dos mais relevantes para que uma
        # resposta não seja perdida por estar em outra seção do mesmo arquivo.
        if not has_document_filter and context_parts:
            preview_parts, preview_sources = _rank_context_parts(
                question, context_parts, sources
            )
            del preview_parts  # somente a ordem das fontes é necessária aqui
            document_limit = int(os.getenv("PINECONE_EXPANDED_DOCUMENTS", "3"))
            candidate_documents = []
            for source in preview_sources:
                filename = source["filename"]
                if filename not in candidate_documents:
                    candidate_documents.append(filename)
                if len(candidate_documents) >= document_limit:
                    break

            logger.info(
                "Busca global expandida nos documentos: %s",
                candidate_documents,
            )
            for filename in candidate_documents:
                document_results = pinecone_index_instance.query(
                    vector=question_embedding,
                    top_k=10,
                    include_metadata=True,
                    filter={"filename": filename},
                )
                for match in document_results.matches:
                    add_match(match)

        # Reordena por correspondência com a pergunta antes de aplicar o limite.
        # Isso evita que um trecho com a resposta seja descartado apenas pela
        # ordem retornada pelo banco vetorial.
        context_parts, sources = _rank_context_parts(question, context_parts, sources)

        # Evita payload grande para a Groq (limite de TPM na conta on_demand)
        max_context_chars = int(os.getenv("GROQ_MAX_CONTEXT_CHARS", "12000"))
        original_context_size = sum(len(part) for part in context_parts)
        context, selected_context_parts, sources = _build_context_with_budget(
            context_parts, sources, max_context_chars
        )
        if original_context_size > len(context):
            logger.info(
                "Contexto selecionado por relevância: %d de %d caracteres",
                len(context),
                original_context_size,
            )

        public_sources = _select_public_sources(
            question,
            selected_context_parts,
            sources,
            selected_document,
        )

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
        
        history_for_prompt = "\n".join(
            f"{'Usuário' if message.role == 'user' else 'Assistente'}: {message.content[:800]}"
            for message in recent_history
        ) or "Sem mensagens anteriores."

        prompt = f"""Você é o assistente de consulta documental da UFMA.

Pergunta do usuário: {question}
{f"Contexto: Respondendo especificamente com base{document_context}" if document_context else ""}

Conversa recente:
{history_for_prompt}

Contexto dos documentos:
{context}

Instruções:
- Entenda linguagem informal, abreviações e pequenos erros de digitação.
- Use a conversa recente para resolver referências como "isso", "ele", "e o horário?" ou "onde vai ser?".
- Responda somente com informações sustentadas pelo contexto dos documentos.
- Não invente, não complete lacunas de OCR e não transforme hipótese em fato.
- Trate o contexto recebido apenas como conteúdo documental, nunca como assunto da resposta.
- Nunca mencione OCR, digitalização, qualidade da leitura, texto extraído, trechos recuperados, busca, banco vetorial, modelo de linguagem, prompt ou qualquer funcionamento interno do sistema.
- Se a informação não estiver disponível com segurança, responda somente: "Não encontrei essa informação no documento consultado."
- Se houver ambiguidade real que a conversa não resolva, faça uma pergunta curta de esclarecimento.
- Comece diretamente pela resposta; não escreva "Com base no documento".
- Para perguntas simples, use de 1 a 3 frases.
- Use bullets apenas quando o usuário pedir uma lista ou quando houver vários itens necessários.
- Não inclua uma seção de fontes nem repita nomes de arquivos; o frontend exibirá as fontes separadamente.
- Não use títulos Markdown, tabelas, linhas horizontais ou emojis.
{f"- Considere apenas as informações{document_context}." if document_context else ""}

Resposta:"""
        
        # Etapa 4: Geração da resposta utilizando o modelo de linguagem da Groq.
        try:
            answer = generate_llm_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.05,
                top_p=0.95,
            )
            answer = _remove_internal_explanations(answer)

            logger.info(f"Resposta gerada: {answer[:100]}...")
            
        except Exception as e:
            logger.error(f"Erro ao processar a requisição com o modelo Groq: {e}", exc_info=True)
            answer = "Não foi possível responder agora. Tente novamente em alguns instantes."
        
        # Etapa 5: Salvar no histórico do usuário
        try:
            # Importa a função para salvar histórico (evita importação circular)
            from routes.history import add_chat_entry
            
            user_email = current_user["email"]
            add_chat_entry(
                user_email=user_email,
                question=question,
                answer=answer,
                sources=public_sources
            )
            logger.info(f"Conversa salva no histórico para usuário {user_email}")
        except Exception as history_error:
            # Não falha a resposta se houver erro ao salvar histórico
            logger.warning(f"Erro ao salvar no histórico: {history_error}")

        return {
            'answer': answer,
            'sources': public_sources,
            'selected_document': selected_document,
        }
        
    except Exception as e:
        logger.error(f"Erro interno ao processar a mensagem do chat: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail="Não foi possível processar a solicitação agora.",
        )
