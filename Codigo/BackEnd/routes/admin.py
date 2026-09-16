from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
import os
import mimetypes
import uuid
import json
from datetime import datetime
from pathlib import Path
import logging
from dotenv import load_dotenv
from routes.document_processor import process_and_index_pdf
from routes.utils import get_pinecone_index

# Configuração básica
load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter()

# Configurações otimizadas
UPLOAD_DIR = "uploads"
METADATA_FILE = "document_metadata.json"  # Arquivo para persistir metadados
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}

# Armazenamento em memória para metadados dos documentos (carregado do arquivo)
documents_metadata = {}

def safe_upload_path(file_id: str) -> str:
    """Resolve um arquivo dentro de uploads e bloqueia path traversal."""
    if not file_id or Path(file_id).name != file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de arquivo inválido"
        )

    upload_root = Path(UPLOAD_DIR).resolve()
    candidate = (upload_root / file_id).resolve()
    if candidate.parent != upload_root:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de arquivo inválido"
        )
    return str(candidate)

def load_metadata():
    """Carrega metadados do arquivo JSON"""
    global documents_metadata
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                documents_metadata = json.load(f)
            logger.info(f"Metadados carregados: {len(documents_metadata)} documentos")
        else:
            documents_metadata = {}
            logger.info("Arquivo de metadados não encontrado, iniciando vazio")
    except Exception as e:
        logger.error(f"Erro ao carregar metadados: {e}")
        documents_metadata = {}

def save_metadata():
    """Salva metadados no arquivo JSON"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents_metadata, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Metadados salvos: {len(documents_metadata)} documentos")
    except Exception as e:
        logger.error(f"Erro ao salvar metadados: {e}")

def sync_metadata_with_files():
    """Sincroniza metadados com arquivos existentes no diretório"""
    try:
        existing_files = set()
        for filename in os.listdir(UPLOAD_DIR):
            if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                existing_files.add(filename)
                
                # Se arquivo existe mas não tem metadados, cria entrada básica
                if filename not in documents_metadata:
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    stat = os.stat(file_path)
                    
                    original_name = filename
                    if '_' in filename:
                        parts = filename.split('_')
                        if len(parts) > 2:
                            original_name = '_'.join(parts[2:])
                    
                    documents_metadata[filename] = {
                        'original_name': original_name,
                        'summary': 'Resumo não disponível - documento carregado antes da implementação de resumos',
                        'upload_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'file_size': stat.st_size
                    }
        
        # Remove metadados de arquivos que não existem mais
        files_to_remove = []
        for filename in documents_metadata.keys():
            if filename not in existing_files:
                files_to_remove.append(filename)
        
        for filename in files_to_remove:
            del documents_metadata[filename]
            logger.info(f"Removido metadado órfão: {filename}")
        
        # Salva alterações
        if files_to_remove or len(existing_files) != len(documents_metadata):
            save_metadata()
            
    except Exception as e:
        logger.error(f"Erro na sincronização de metadados: {e}")

# Carrega metadados ao inicializar o módulo
load_metadata()
sync_metadata_with_files()

@router.get("/documents")
async def list_documents():
    """Lista todos os documentos armazenados com metadados"""
    documents = []
    try:
        for filename in os.listdir(UPLOAD_DIR):
            if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                file_path = os.path.join(UPLOAD_DIR, filename)
                stat = os.stat(file_path)
                
                # Extrai o nome original se estiver no formato correto
                original_name = filename
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) > 2:
                        original_name = '_'.join(parts[2:])
                
                # Pega o resumo dos metadados se disponível
                metadata = documents_metadata.get(filename, {})
                summary = metadata.get('summary', 'Resumo não disponível')
                
                documents.append({
                    "id": filename,
                    "original_name": original_name,
                    "size": stat.st_size,
                    "upload_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "summary": summary
                })
        
        return JSONResponse(content=documents)
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao listar documentos"
        )

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload de PDF ou imagem com processamento automático e indexação
    """
    # Validações 
    original_filename = Path((file.filename or "").replace("\\", "/")).name
    file_extension = Path(original_filename).suffix.lower()
    if not original_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nome de arquivo inválido"
        )
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formatos aceitos: PDF, PNG, JPG, JPEG, WEBP, TIF e TIFF"
        )

    file_path = None
    try:
        # Gera nome único preservando a extensão 
        file_id = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}_{original_filename}"
        file_path = safe_upload_path(file_id)
        
        # Salva em stream com verificação de tamanho 
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Arquivo excede o tamanho máximo de {MAX_FILE_SIZE//(1024*1024)}MB"
                    )
                buffer.write(chunk)
        
        # PROCESSAMENTO AUTOMÁTICO
        logger.info(f"Iniciando processamento automático de {original_filename}")
        try:
            processing_result = await process_and_index_pdf(file_path, file_id)
            logger.info(f"Processamento concluído: {processing_result}")
            
            # Salva metadados incluindo resumo E persiste no arquivo
            documents_metadata[file_id] = {
                'original_name': original_filename,
                'summary': processing_result.get('summary', 'Resumo não disponível'),
                'upload_date': datetime.now().isoformat(),
                'file_size': file_size
            }
            
            # Persiste metadados no arquivo
            save_metadata()
            
            # Retorna metadados completos + resultado do processamento
            return {
                "id": file_id,
                "original_name": original_filename,
                "size": file_size,
                "upload_date": datetime.now().isoformat(),
                "file_path": file_path,
                "processing_result": processing_result,
                "summary": processing_result.get('summary', 'Resumo não disponível'),
                "status": "success",
                "message": f"Documento '{original_filename}' carregado e indexado com sucesso!"
            }
            
        except Exception as processing_error:
            # Se falhar no processamento, remove o arquivo e informa erro
            logger.error(f"Erro no processamento de {original_filename}: {processing_error}")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Arquivo removido após falha no processamento: {file_path}")
            
            # Re-lança erro para informar o usuário
            if isinstance(processing_error, HTTPException):
                raise processing_error
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Falha no processamento do documento: {str(processing_error)}"
                )

    except HTTPException:
        raise
    except Exception as e:
        # Remove arquivo em caso de erro geral
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha no processamento do documento"
        )

@router.get("/download/{file_id}")
async def download_document(file_id: str):
    """Endpoint para download direto de documentos"""
    try:
        file_path = safe_upload_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Documento não encontrado"
            )
        
        # Obtém o nome original do arquivo se disponível
        metadata = documents_metadata.get(file_id, {})
        original_filename = metadata.get('original_name', file_id)
        
        # Fallback para extração do nome se não estiver nos metadados
        if not original_filename or original_filename == file_id:
            if '_' in file_id:
                parts = file_id.split('_')
                if len(parts) > 2:
                    original_filename = '_'.join(parts[2:])
        
        media_type = mimetypes.guess_type(original_filename)[0] or 'application/octet-stream'

        return FileResponse(
            path=file_path,
            filename=original_filename,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao recuperar documento"
        )

@router.delete("/document/{file_id}")
async def delete_document(file_id: str):
    """Remove um documento permanentemente"""
    try:
        file_path = safe_upload_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Documento não encontrado"
            )
            
        # Remove primeiro os vetores; se falhar, mantém arquivo e metadados
        # para que a operação possa ser repetida sem deixar dados órfãos.
        pinecone_index = get_pinecone_index()
        if pinecone_index is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Índice vetorial indisponível"
            )
        pinecone_index.delete(filter={"filename": {"$eq": file_id}})

        os.remove(file_path)
        
        # Remove metadados também e persiste a alteração
        if file_id in documents_metadata:
            del documents_metadata[file_id]
            save_metadata()
            
        return {"success": True, "message": "Documento removido com sucesso"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao remover documento"
        )

@router.get("/metadata/sync")
async def sync_metadata():
    """Endpoint para sincronizar metadados manualmente"""
    try:
        sync_metadata_with_files()
        return {
            "success": True, 
            "message": "Metadados sincronizados com sucesso",
            "total_documents": len(documents_metadata)
        }
    except Exception as e:
        logger.error(f"Sync error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao sincronizar metadados"
        )
