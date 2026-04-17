# Módulo para encapsular funcionalidades centrais do Retrieval-Augmented Generation (RAG).
# Inclui a geração de embeddings, interação com o banco de dados vetorial Pinecone,
# e funcionalidades de busca simples para fallback.

import os
import logging
import re
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index, PodSpec, ServerlessSpec
from groq import Groq
import time 

# Configuração do logger para monitoramento e depuração
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Armazenamento em memória para documentos processados
document_store = []


def _sanitize_model_output(text: str) -> str:
    """Remove blocos de raciocínio interno (ex.: <think>...</think>) da resposta final."""
    if not text:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()

# --- Configuração de LLM (Groq com fallback para Ollama) ---
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        logger.info("Cliente Groq inicializado para geração de respostas.")
    except Exception as e:
        logger.warning(f"Falha ao inicializar cliente Groq: {e}")


def _generate_with_ollama(prompt: str, max_tokens: int, temperature: float, top_p: float | None = None) -> str:
    """Gera resposta via servidor local (Ollama nativo ou API compatível OpenAI)."""
    base_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
    ollama_max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
    request_timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
    local_max_tokens = min(max_tokens, ollama_max_tokens)

    errors = []

    # 1) Ollama nativo: /api/generate
    try:
        generate_payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": local_max_tokens,
            },
        }
        if top_p is not None:
            generate_payload["options"]["top_p"] = top_p

        response = requests.post(f"{base_url}/api/generate", json=generate_payload, timeout=request_timeout)
        if response.status_code < 400:
            data = response.json()
            text = (data.get("response") or "").strip()
            if text:
                return _sanitize_model_output(text)
            errors.append("/api/generate retornou resposta vazia")
        else:
            errors.append(f"/api/generate status {response.status_code}: {response.text[:180]}")
    except Exception as e:
        errors.append(f"/api/generate erro: {e}")

    # 2) Ollama chat API: /api/chat
    try:
        chat_payload = {
            "model": ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": local_max_tokens,
            },
        }
        if top_p is not None:
            chat_payload["options"]["top_p"] = top_p

        response = requests.post(f"{base_url}/api/chat", json=chat_payload, timeout=request_timeout)
        if response.status_code < 400:
            data = response.json()
            msg = data.get("message", {}) if isinstance(data, dict) else {}
            text = (msg.get("content") or "").strip()
            if text:
                return _sanitize_model_output(text)
            errors.append("/api/chat retornou resposta vazia")
        else:
            errors.append(f"/api/chat status {response.status_code}: {response.text[:180]}")
    except Exception as e:
        errors.append(f"/api/chat erro: {e}")

    # 3) Compatível OpenAI: /v1/chat/completions
    try:
        openai_payload = {
            "model": ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": local_max_tokens,
        }
        if top_p is not None:
            openai_payload["top_p"] = top_p

        response = requests.post(f"{base_url}/v1/chat/completions", json=openai_payload, timeout=request_timeout)
        if response.status_code < 400:
            data = response.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            text = ""
            if choices:
                message = choices[0].get("message", {})
                text = (message.get("content") or "").strip()
            if text:
                return _sanitize_model_output(text)
            errors.append("/v1/chat/completions retornou resposta vazia")
        else:
            errors.append(f"/v1/chat/completions status {response.status_code}: {response.text[:180]}")
    except Exception as e:
        errors.append(f"/v1/chat/completions erro: {e}")

    raise RuntimeError("Falha no fallback local (Ollama/API compatível): " + " | ".join(errors))


def _should_fallback_to_ollama(exc: Exception) -> bool:
    msg = str(exc).lower()
    fallback_markers = [
        "model_decommissioned",
        "decommissioned",
        "rate_limit_exceeded",
        "request too large",
        "tokens per minute",
        "requested",
    ]
    return any(marker in msg for marker in fallback_markers)


def generate_llm_response(prompt: str, max_tokens: int, temperature: float, top_p: float | None = None) -> str:
    """
    Gera resposta de LLM com priorização em Groq e fallback para Ollama local.

    Configurações por ambiente:
    - GROQ_MODEL (default: qwen/qwen3-32b)
    - OLLAMA_URL (default: http://host.docker.internal:11434)
    - OLLAMA_MODEL (default: qwen2.5:latest)
    - LLM_PROVIDER (groq|ollama, default: groq)
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

    if provider == "ollama":
        return _generate_with_ollama(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    if groq_client is None:
        logger.warning("Groq indisponível (sem chave ou inicialização). Usando Ollama.")
        return _generate_with_ollama(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    try:
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else 1,
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Groq retornou resposta vazia")
        return _sanitize_model_output(text)
    except Exception as groq_error:
        if _should_fallback_to_ollama(groq_error):
            logger.warning(f"Falha no Groq ({groq_model}). Tentando Ollama fallback: {groq_error}")
            return _generate_with_ollama(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        raise

# --- Configuração do Modelo de Embeddings ---
# O modelo de embedding é inicializado uma única vez na carga do módulo para eficiência.
embedding_model = None # Inicializado como None para controle de estado
try:
    # Lê o modelo das variáveis de ambiente, com fallback para o modelo padrão
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(model_name)
    logger.info(f"Modelo de embedding '{model_name}' carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo de embedding SentenceTransformer: {e}", exc_info=True)
    # Falhas na carga do modelo impactam diretamente a funcionalidade de geração de embeddings.

def generate_embedding(text: str) -> list[float]:
    """
    Gera um vetor numérico (embedding) para o texto fornecido.
    Estes embeddings são cruciais para a busca de similaridade em bancos de dados vetoriais.

    Raises:
        RuntimeError: Se o modelo de embedding não foi inicializado corretamente.
    """
    if embedding_model is None:
        raise RuntimeError("Modelo de embedding não inicializado. Verifique os logs de inicialização e as configurações.")
    return embedding_model.encode(text).tolist()

# --- Configuração do Pinecone ---
pinecone_client = None
pinecone_index = None

try:
    # Inicializa o cliente Pinecone utilizando as chaves de API e ambiente das variáveis de ambiente.
    pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT") # Mantém o environment para compatibilidade de API
    )
    logger.info("Cliente Pinecone inicializado.")

    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Conecta-se diretamente ao índice existente ou o cria se não existir
    try:
        pinecone_index = pinecone_client.Index(index_name)
        
        # Testa a conexão
        stats = pinecone_index.describe_index_stats()
        logger.info(f"Conectado ao índice Pinecone '{index_name}' com sucesso. Estatísticas: {stats}")
        
    except Exception as index_error:
        logger.warning(f"Não foi possível conectar ao índice existente '{index_name}': {index_error}")
        
        # Se não conseguir conectar, verifica se o índice existe
        try:
            available_indexes = pinecone_client.list_indexes()
            logger.info(f"Índices disponíveis: {available_indexes}")
            
            # Ajuste aqui para verificar se o nome do índice existe na lista retornada
            if index_name not in [idx['name'] for idx in available_indexes]: # Pinecone v3 retorna uma lista de dicionários
                logger.info(f"Índice '{index_name}' não encontrado. Criando novo índice...")
                
                # Cria um novo índice com as dimensões corretas
                # A dimensão do índice é lida das variáveis de ambiente, com fallback para 384
                embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
                pinecone_client.create_index(
                    name=index_name,
                    dimension=embedding_dimensions,  # Dimensão configurável
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws', 
                        region='us-east-1' 
                    )
                )
                logger.info(f"Índice '{index_name}' criado com sucesso com {embedding_dimensions} dimensões.")
                
                # Aguarda criação
                time.sleep(10) # Importante aguardar a criação do índice

            # Tenta conectar novamente
            pinecone_index = pinecone_client.Index(index_name)
            logger.info(f"Conectado ao índice Pinecone '{index_name}' após verificação.")
            
        except Exception as create_error:
            logger.error(f"Erro ao verificar/criar índice: {create_error}")
            pinecone_index = None

except Exception as e:
    logger.error(f"Erro crítico ao inicializar o Pinecone: {e}. Funcionalidades RAG podem ser afetadas.", exc_info=True)
    pinecone_index = None # Garante que a variável permaneça None em caso de falha.

def get_pinecone_index() -> Index:
    """
    Fornece a instância do índice Pinecone para outros módulos.
    Isso centraliza o acesso ao índice, garantindo uma única fonte de verdade.

    Raises:
        RuntimeError: Se o índice Pinecone não foi inicializado corretamente.
    """
    if pinecone_index is None:
        raise RuntimeError("Pinecone index não inicializado. Verifique logs e variáveis de ambiente.")
    return pinecone_index

# --- Função de Divisão de Texto (Text Splitter Customizado) ---
def create_documents_from_text_manual(text: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """
    Implementa um divisor de texto customizado para segmentar documentos longos em "chunks" menores.
    Inclui sobreposição entre chunks para preservar o contexto.
    Essa abordagem substitui a necessidade de bibliotecas de text splitting mais pesadas como Langchain.

    Args:
        text (str): O conteúdo textual completo do documento.
        filename (str): O nome do arquivo original, usado nos metadados dos chunks.
        chunk_size (int): O tamanho máximo desejado para cada chunk.
        overlap (int): O número de caracteres de sobreposição entre chunks consecutivos.

    Returns:
        list[dict]: Uma lista de dicionários, onde cada dicionário representa um chunk
                    e contém seu 'content' e 'metadata'.
    """
    chunks_data = []
    text_length = len(text)
    start = 0
    chunk_order = 0 # Mantém a ordem dos chunks para metadados

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_content = text[start:end]
        
        if not chunk_content.strip(): # Ignora chunks vazios ou contendo apenas espaços em branco
            # Ajusta o ponto de partida para o próximo chunk, evitando loops em texto residual vazio
            start = end - overlap if end - overlap >= 0 else 0
            if start >= text_length:
                break
            continue

        # Associa o conteúdo do chunk com metadados relevantes para indexação no Pinecone.
        chunks_data.append({
            "content": chunk_content,
            "metadata": {
                "filename": filename,
                "chunk_order": chunk_order,
                "start_char": start,
                "end_char": end,
                # Metadados adicionais, como número da página, podem ser incluídos se extraídos.
            }
        })
        
        chunk_order += 1
        start = end - overlap
        if start < 0: start = 0

    return chunks_data

def simple_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Implementação alternativa de divisor de texto para otimizar processamento de documentos grandes.
    Divide texto em chunks com sobreposição para manter contexto entre segmentos.
    Retorna apenas o conteúdo dos chunks (sem metadados).
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length) # Garante que 'end' não exceda o tamanho do texto
        chunk = text[start:end]
        
        if chunk:
            chunks.append(chunk)
        
        # Ajusta o 'start' para o próximo chunk com sobreposição
        start = end - overlap
        if start < 0: # Evita start negativo no início
            start = 0
    
    return chunks

def simple_search(query: str, documents: list[dict], max_results: int = 3) -> list[dict]:
    """
    Sistema de busca por relevância baseado em contagem de palavras-chave.
    Implementa scoring simples mas eficaz para recuperação de informações.
    Funciona como fallback quando o Pinecone não está disponível.
    
    Args:
        query (str): A pergunta do usuário para a busca.
        documents (list[dict]): Lista de documentos (chunks) no formato {'content': '...', 'filename': '...'}.
        max_results (int): Número máximo de resultados a serem retornados.

    Returns:
        list[dict]: Lista de documentos relevantes ordenados por score.
    """
    query_words = query.lower().split()
    results = []
    
    for doc in documents:
        score = 0
        content_lower = doc['content'].lower()
        
        for word in query_words:
            score += content_lower.count(word)
        
        if score > 0:
            results.append({
                'content': doc['content'],
                'filename': doc['filename'],
                'score': score
            })
    
    # Ordenação por relevância para retornar os resultados mais pertinentes
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]