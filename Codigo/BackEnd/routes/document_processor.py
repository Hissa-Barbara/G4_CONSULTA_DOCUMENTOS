import os
import io
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException

# Imports para processamento de PDF
import fitz
import pdfplumber
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader
from difflib import SequenceMatcher

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_AVAILABLE = False

# Imports do sistema RAG
from routes.utils import generate_embedding, get_pinecone_index, generate_llm_response

# LangChain para chunking inteligente
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain não disponível, usando chunking manual")

logger = logging.getLogger(__name__)

class HybridDocumentProcessor:
    """Processador HÍBRIDO: LangChain chunking + Sentence Transformers embeddings + Resumo com LLM"""
    
    def __init__(self):
        # Configurações otimizadas para máxima preservação do contexto
        self.chunk_size = 2500      # Tamanho para manter contexto
        self.chunk_overlap = 600    # Overlap para excelente continuidade
        self.batch_size = 32        # Processamento em lotes grandes
        self.min_text_chars_for_ocr = 80
        self.max_pages = 100
        self.ocr_render_scale = float(os.getenv("OCR_RENDER_SCALE", "3.5"))
        self.ocr_min_quality = float(os.getenv("OCR_MIN_QUALITY", "62"))
        self.gemini_enabled = os.getenv("GEMINI_VISUAL_ENABLED", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self.gemini_models = [
            model.strip()
            for model in os.getenv(
                "GEMINI_VISUAL_MODELS",
                "gemini-3.5-flash,gemini-3.6-flash",
            ).split(",")
            if model.strip()
        ]
        self.gemini_retries = max(1, int(os.getenv("GEMINI_RETRIES_PER_MODEL", "1")))
        self.gemini_retry_delay = max(0.0, float(os.getenv("GEMINI_RETRY_DELAY", "2")))
        self.gemini_max_inline_bytes = int(
            float(os.getenv("GEMINI_MAX_INLINE_MB", "18")) * 1024 * 1024
        )

        # Permite configurar outro caminho e detecta a instalação padrão no Windows.
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if not tesseract_cmd and os.name == "nt":
            program_files = os.getenv("ProgramFiles", r"C:\Program Files")
            default_tesseract = Path(program_files) / "Tesseract-OCR" / "tesseract.exe"
            if default_tesseract.exists():
                tesseract_cmd = str(default_tesseract)

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _gemini_is_configured(self) -> bool:
        """Confirma se a extração visual pode ser usada sem expor a chave."""
        return bool(
            self.gemini_enabled
            and GEMINI_AVAILABLE
            and os.getenv("GEMINI_API_KEY")
            and self.gemini_models
        )

    @staticmethod
    def _is_transient_gemini_error(exc: Exception) -> bool:
        """Identifica indisponibilidade temporária ou limite momentâneo."""
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None
        return status_code in {429, 500, 502, 503, 504}

    def _extract_with_gemini(self, file_path: str, mime_type: str) -> str:
        """Transcreve documentos visuais com troca automática de modelo."""
        if not self._gemini_is_configured():
            return ""

        file_size = os.path.getsize(file_path)
        if file_size > self.gemini_max_inline_bytes:
            logger.warning(
                "Documento visual excede o limite configurado do Gemini (%d bytes)",
                file_size,
            )
            return ""

        document_bytes = Path(file_path).read_bytes()
        if mime_type == "application/pdf":
            prompt = """Transcreva fielmente todo o documento em português.
Leia tanto o texto digital quanto todo texto presente em páginas escaneadas,
imagens, quadros e tabelas. Preserve números, datas, cargas horárias, nomes,
títulos, artigos e a relação entre linhas e colunas. Converta tabelas para
Markdown sem separar uma célula de seu cabeçalho. Marque cada página como
--- Página N ---. Retorne somente a transcrição. Não resuma, não explique,
não corrija e não invente informações ausentes."""
        else:
            prompt = """Transcreva fielmente todo o conteúdo textual desta imagem
em português. Preserve títulos, números, datas, horários e a relação entre
linhas e colunas. Converta tabelas e programações para Markdown. Comece com
--- Imagem 1 ---. Retorne somente a transcrição. Não resuma, não explique,
não corrija e não invente informações ausentes."""

        last_error = None
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        try:
            for model in self.gemini_models:
                for attempt in range(1, self.gemini_retries + 1):
                    try:
                        logger.info(
                            "Extração visual com Gemini: modelo=%s tentativa=%d",
                            model,
                            attempt,
                        )
                        response = client.models.generate_content(
                            model=model,
                            contents=[
                                genai_types.Part.from_bytes(
                                    data=document_bytes,
                                    mime_type=mime_type,
                                ),
                                prompt,
                            ],
                        )
                        text = self._clean_text(response.text or "")
                        if text:
                            logger.info(
                                "Extração visual concluída com %s: %d caracteres",
                                model,
                                len(text),
                            )
                            return text
                        last_error = RuntimeError("O Gemini retornou conteúdo vazio")
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "Falha no Gemini (%s, tentativa %d): %s",
                            model,
                            attempt,
                            exc,
                        )
                        if not self._is_transient_gemini_error(exc):
                            break
                        if attempt < self.gemini_retries:
                            time.sleep(self.gemini_retry_delay * attempt)
        finally:
            client.close()

        logger.warning("Gemini indisponível; usando extração local: %s", last_error)
        return ""

    def _pdf_requires_visual_extraction(self, file_path: str) -> bool:
        """Detecta scans e páginas híbridas sem consumir a API desnecessariamente."""
        if not self._gemini_is_configured():
            return False

        document = None
        try:
            document = fitz.open(file_path)
            for page_num in range(min(len(document), self.max_pages)):
                page = document[page_num]
                digital_text = self._clean_text(page.get_text("text") or "")
                if len(digital_text) < self.min_text_chars_for_ocr:
                    return True

                page_area = page.rect.width * page.rect.height
                if page_area <= 0:
                    continue
                for image_info in page.get_images(full=True):
                    rects = page.get_image_rects(image_info[0])
                    if any((rect.width * rect.height) / page_area >= 0.12 for rect in rects):
                        return True
            return False
        except Exception as exc:
            logger.warning("Não foi possível classificar o PDF visualmente: %s", exc)
            return False
        finally:
            if document is not None:
                document.close()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normaliza espaços sem destruir as quebras de parágrafo."""
        if not text:
            return ""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _format_table(table: List[List[Any]], table_number: int) -> str:
        """Converte uma tabela em Markdown, preservando linhas e colunas."""
        rows = []
        for row in table or []:
            cleaned_row = [
                str(cell).replace("\n", " ").replace("|", "\\|").strip()
                if cell is not None else ""
                for cell in row
            ]
            if any(cleaned_row):
                rows.append(cleaned_row)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        rows = [row + [""] * (column_count - len(row)) for row in rows]
        header = rows[0]
        body = rows[1:]

        lines = [
            f"[Tabela {table_number}]",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    def _extract_tables(self, page: Any) -> List[str]:
        """Extrai tabelas digitais da página."""
        extracted = []
        try:
            for table_number, table in enumerate(page.extract_tables(), start=1):
                formatted = self._format_table(table, table_number)
                if formatted:
                    extracted.append(formatted)
        except Exception as exc:
            logger.warning(f"Não foi possível extrair tabela: {exc}")
        return extracted

    @staticmethod
    def _otsu_threshold(image: Image.Image) -> int:
        """Calcula um limiar automático para separar texto e fundo."""
        histogram = image.histogram()
        total = sum(histogram)
        weighted_sum = sum(index * count for index, count in enumerate(histogram))
        background_weight = 0
        background_sum = 0
        best_variance = -1.0
        best_threshold = 160

        for threshold, count in enumerate(histogram):
            background_weight += count
            if background_weight == 0:
                continue
            foreground_weight = total - background_weight
            if foreground_weight == 0:
                break
            background_sum += threshold * count
            background_mean = background_sum / background_weight
            foreground_mean = (weighted_sum - background_sum) / foreground_weight
            variance = (
                background_weight
                * foreground_weight
                * (background_mean - foreground_mean) ** 2
            )
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold

        return best_threshold

    def _prepare_ocr_images(self, image: Image.Image) -> List[Image.Image]:
        """Cria versões adequadas a scans apagados, manchados ou pouco nítidos."""
        grayscale = ImageOps.exif_transpose(image).convert("L")
        enhanced = ImageOps.autocontrast(grayscale, cutoff=(1, 1)).filter(
            ImageFilter.UnsharpMask(radius=1.4, percent=180, threshold=2)
        )
        threshold = self._otsu_threshold(enhanced)
        binary = enhanced.point(
            lambda pixel: 255 if pixel > threshold else 0,
            mode="1",
        ).convert("L")
        return [enhanced, binary]

    @staticmethod
    def _rebuild_ocr_lines(data: Dict[str, List[Any]]) -> str:
        """Reconstrói as linhas produzidas pelo Tesseract sem perder listas."""
        lines: List[str] = []
        current_key = None
        current_words: List[str] = []

        for index, raw_word in enumerate(data.get("text", [])):
            word = str(raw_word).strip()
            if not word:
                continue
            key = (
                data["block_num"][index],
                data["par_num"][index],
                data["line_num"][index],
            )
            if current_key is not None and key != current_key and current_words:
                lines.append(" ".join(current_words))
                current_words = []
            current_key = key
            current_words.append(word)

        if current_words:
            lines.append(" ".join(current_words))
        return "\n".join(lines)

    def _run_ocr(self, image: Image.Image, psm: int) -> tuple[str, float]:
        """Executa uma tentativa e devolve texto e qualidade média reconhecida."""
        config = (
            f"--oem 3 --psm {psm} --dpi 300 "
            "-c preserve_interword_spaces=1"
        )

        def read_data(language: str | None) -> Dict[str, List[Any]]:
            kwargs = {
                "output_type": pytesseract.Output.DICT,
                "config": config,
            }
            if language:
                kwargs["lang"] = language
            return pytesseract.image_to_data(image, **kwargs)

        try:
            data = read_data("por+eng")
        except pytesseract.TesseractError:
            data = read_data(None)

        text = self._clean_text(self._rebuild_ocr_lines(data))
        weighted_confidence = 0.0
        confidence_weight = 0
        useful_words = 0
        for raw_word, raw_confidence in zip(data["text"], data["conf"]):
            word = str(raw_word).strip()
            if not word:
                continue
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if confidence < 0:
                continue
            weight = max(1, len(word))
            weighted_confidence += confidence * weight
            confidence_weight += weight
            if re.search(r"[A-Za-zÀ-ÿ]{3,}", word):
                useful_words += 1

        average_confidence = (
            weighted_confidence / confidence_weight if confidence_weight else 0.0
        )
        lines = [line.split() for line in text.splitlines() if line.strip()]
        single_word_ratio = (
            sum(len(line) == 1 for line in lines) / len(lines) if lines else 1.0
        )
        quality = (
            average_confidence
            + min(useful_words, 200) * 0.04
            - single_word_ratio * 30
        )
        return text, quality

    def _ocr_image(self, image: Image.Image, psm: int = 6) -> str:
        """Executa OCR adaptativo e mantém a tentativa de melhor qualidade."""
        prepared_images = self._prepare_ocr_images(image)
        attempts: List[tuple[str, float]] = []
        primary_text, primary_quality = self._run_ocr(prepared_images[0], psm)
        attempts.append((primary_text, primary_quality))

        if primary_quality < self.ocr_min_quality:
            attempts.append(self._run_ocr(prepared_images[1], psm))
            alternative_psm = 11 if psm == 6 else 6
            attempts.append(self._run_ocr(prepared_images[0], alternative_psm))

        best_text, best_quality = max(attempts, key=lambda attempt: attempt[1])
        logger.info(
            "OCR concluído com qualidade %.1f em %d tentativa(s)",
            best_quality,
            len(attempts),
        )
        return best_text

    def _ocr_full_page(self, page: fitz.Page) -> str:
        """Renderiza uma página escaneada e aplica OCR."""
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(self.ocr_render_scale, self.ocr_render_scale),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return self._ocr_image(image)

    def _extract_image_text(self, document: fitz.Document, page: fitz.Page) -> List[str]:
        """Aplica OCR às imagens relevantes incorporadas em uma página digital."""
        image_texts = []
        processed_xrefs = set()

        for image_info in page.get_images(full=True):
            xref = image_info[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            image_rects = page.get_image_rects(xref)
            page_area = page.rect.width * page.rect.height

            if not image_rects or page_area <= 0:
                continue

            largest_rect = max(
                image_rects,
                key=lambda rect: rect.width * rect.height,
            )
            largest_image_ratio = (
                largest_rect.width * largest_rect.height
            ) / page_area

            # Ignora logotipos, ícones e outras imagens decorativas pequenas
            if largest_image_ratio < 0.12:
                continue

            try:
                # Renderiza como a imagem aparece na página. Isso respeita
                # rotação, recorte e escala aplicados pelo próprio PDF.
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    clip=largest_rect,
                    alpha=False,
                )
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                text = self._ocr_image(image, psm=11)
                useful_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
                is_duplicate = any(
                    SequenceMatcher(None, text, existing_text).ratio() >= 0.85
                    for existing_text in image_texts
                )

                if len(useful_words) >= 5 and not is_duplicate:
                    image_texts.append(text)
            except Exception as exc:
                logger.warning(f"Não foi possível processar uma imagem: {exc}")

        return image_texts
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extrai texto digital, tabelas e conteúdo obtido por OCR."""
        fitz_document = None
        try:
            reader = PdfReader(file_path)
            fitz_document = fitz.open(file_path)
            extracted_pages = []

            with pdfplumber.open(file_path) as plumber_document:
                max_pages = min(
                    len(reader.pages),
                    len(fitz_document),
                    len(plumber_document.pages),
                    self.max_pages,
                )

                for page_num in range(max_pages):
                    try:
                        digital_text = self._clean_text(
                            reader.pages[page_num].extract_text() or ""
                        )
                        tables = self._extract_tables(plumber_document.pages[page_num])
                        page_parts = [f"--- Página {page_num + 1} ---"]

                        if len(digital_text) < self.min_text_chars_for_ocr:
                            ocr_text = self._ocr_full_page(fitz_document[page_num])
                            if ocr_text:
                                page_parts.append(ocr_text)
                            elif digital_text:
                                page_parts.append(digital_text)
                        else:
                            page_parts.append(digital_text)
                            image_texts = self._extract_image_text(
                                fitz_document, fitz_document[page_num]
                            )
                            for image_number, image_text in enumerate(image_texts, start=1):
                                if image_text not in digital_text:
                                    page_parts.append(image_text)

                        if tables:
                            page_parts.append("\n".join(tables))

                        if len(page_parts) > 1:
                            extracted_pages.append("\n\n".join(page_parts))
                    except Exception as exc:
                        logger.warning(f"Erro na página {page_num + 1}: {exc}")

            all_text = "\n\n".join(extracted_pages)
            if not all_text.strip():
                raise ValueError("Nenhum texto extraído do PDF")

            logger.info(
                f"Texto extraído: {len(all_text):,} caracteres de {max_pages} páginas"
            )
            return all_text

        except Exception as e:
            logger.error(f"Erro ao extrair texto: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao processar PDF: {str(e)}")
        finally:
            if fitz_document is not None:
                fitz_document.close()

    def extract_text_from_image(self, file_path: str) -> str:
        """Extrai texto de uma imagem enviada diretamente."""
        try:
            with Image.open(file_path) as image:
                text = self._ocr_image(image, psm=11)

            if not text:
                raise ValueError("Nenhum texto reconhecido na imagem")

            return f"--- Imagem 1 ---\n\n{text}"
        except Exception as e:
            logger.error(f"Erro ao extrair texto da imagem: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao processar imagem: {str(e)}",
            )

    def extract_text_from_document(self, file_path: str) -> str:
        """Seleciona o extrator adequado conforme a extensão do arquivo."""
        extension = Path(file_path).suffix.lower()
        if extension == ".pdf":
            if self._pdf_requires_visual_extraction(file_path):
                visual_text = self._extract_with_gemini(file_path, "application/pdf")
                if visual_text:
                    return visual_text
            return self.extract_text_from_pdf(file_path)
        if extension in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
            }
            visual_text = self._extract_with_gemini(file_path, mime_types[extension])
            if visual_text:
                return visual_text
            return self.extract_text_from_image(file_path)
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado")

    def generate_document_summary(self, text: str, filename: str) -> str:
        """Gera um resumo do documento usando Groq LLM com fallback robusto"""
        try:
            # Pega primeiros caracteres para o resumo
            text_for_summary = text[:8000] if len(text) > 8000 else text
            
            prompt = f"""Analise o seguinte documento da UFMA e crie um resumo conciso e informativo:

DOCUMENTO: {filename}

CONTEÚDO:
{text_for_summary}

Crie um resumo de 3-4 parágrafos que inclua:
1. Tipo de documento e seu propósito principal
2. Principais pontos, regras ou decisões abordadas
3. Quem é afetado por este documento (estudantes, professores, etc.)
4. Informações práticas importantes

Mantenha o resumo claro, objetivo e útil para quem precisa consultar este documento."""

            prompt += '''

Não mencione OCR, digitalização, qualidade da leitura, extração de texto,
processamento, indexação, modelo ou qualquer funcionamento interno do sistema.
Se o conteúdo não permitir um resumo confiável, responda somente: "Resumo não disponível."'''

            summary = generate_llm_response(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
            )
            
            # Validação do resumo gerado
            if not summary or len(summary.strip()) < 50:
                logger.warning(f"Resumo muito curto gerado para {filename}, usando fallback")
                return self._generate_fallback_summary(text, filename)
            
            logger.info(f"Resumo gerado para {filename}: {len(summary)} caracteres")
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo com Groq: {e}")
            return self._generate_fallback_summary(text, filename)
    
    def _generate_fallback_summary(self, text: str, filename: str) -> str:
        """Retorna um estado neutro quando não há resumo confiável."""
        logger.warning("Resumo confiável indisponível para %s", filename)
        return "Resumo não disponível."
    
    def create_smart_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Chunking inteligente usando LangChain para melhor qualidade de contexto"""
        
        if LANGCHAIN_AVAILABLE:
            # Usa LangChain para chunking inteligente com separadores otimizados
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],  # Separadores inteligentes
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            # Cria documentos com metadados
            metadatas = [{"filename": filename}]
            langchain_docs = text_splitter.create_documents([text], metadatas=metadatas)
            
            # Converte para o formato do sistema
            chunks = []
            for i, doc in enumerate(langchain_docs):
                if len(doc.page_content.strip()) > 50:  # Pula chunks muito pequenos
                    chunks.append({
                        "content": doc.page_content.strip(),
                        "metadata": {
                            "filename": filename,
                            "chunk_order": i,
                            "char_count": len(doc.page_content),
                            "source": "langchain_recursive"
                        }
                    })
            
            logger.info(f"LangChain criou {len(chunks)} chunks inteligentes")
            
        else:
            # Fallback para chunking manual otimizado
            chunks = self._manual_chunking(text, filename)
            logger.info(f"Chunking manual criou {len(chunks)} chunks")
            
        return chunks
    
    def _manual_chunking(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Chunking manual de backup com configurações otimizadas"""
        chunks = []
        text_length = len(text)
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Tenta quebrar em final de frase ou parágrafo
            if end < text_length:
                # Procura por quebras naturais
                for separator in ["\n\n", "\n", ".", "!", "?"]:
                    sep_pos = text.rfind(separator, max(start + self.chunk_size//2, start), end)
                    if sep_pos > start + self.chunk_size//2:
                        end = sep_pos + len(separator)
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:  # Só adiciona chunks úteis
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "chunk_order": chunk_num,
                        "start_char": start,
                        "end_char": end,
                        "char_count": len(chunk_text),
                        "source": "manual"
                    }
                })
                chunk_num += 1
            
            start = end - self.chunk_overlap
            if start >= text_length:
                break
                
        return chunks
    
    def batch_generate_embeddings(self, contents: List[str]) -> List[List[float]]:
        """Embeddings em lote otimizado para alta performance"""
        try:
            # Carrega modelo uma vez só para eficiência
            if not hasattr(self, '_model'):
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                
                logger.info(f"Carregando modelo {model_name}...")
                self._model = SentenceTransformer(
                    model_name,
                    device='cpu',
                    trust_remote_code=False
                )
                logger.info("Modelo carregado e otimizado!")
            
            # Processa em lotes otimizados
            logger.info(f"Gerando {len(contents)} embeddings em lote...")
            
            embeddings = self._model.encode(
                contents,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device='cpu'  # Força CPU para estabilidade
            )
            
            logger.info(f"{len(embeddings)} embeddings gerados com sucesso!")
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Erro nos embeddings: {str(e)}")
    
    def optimized_pinecone_insert(self, chunks: List[Dict], embeddings: List[List[float]], filename: str, summary: str) -> Dict[str, Any]:
        """Inserção otimizada no Pinecone com controle de qualidade"""
        try:
            pinecone_index = get_pinecone_index()
            if not pinecone_index:
                raise RuntimeError("Pinecone não inicializado")
            
            vectors_to_insert = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # ID único mais simples
                chunk_id = f"{Path(filename).stem}_{i}_{uuid.uuid4().hex[:6]}"
                
                vectors_to_insert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "content": chunk["content"],
                        "filename": filename,
                        "chunk_order": chunk["metadata"]["chunk_order"],
                        "char_count": chunk["metadata"]["char_count"],
                        "source": chunk["metadata"].get("source", "unknown"),
                        "indexed_at": datetime.now().isoformat(),
                        "summary": summary  # Inclui o resumo nos metadados
                    }
                })
            
            # Inserção em lotes grandes para melhor performance
            batch_size = 100
            total_inserted = 0
            
            logger.info(f"Inserindo {len(vectors_to_insert)} vetores...")
            
            for i in range(0, len(vectors_to_insert), batch_size):
                batch = vectors_to_insert[i:i + batch_size]
                try:
                    pinecone_index.upsert(vectors=batch)
                    total_inserted += len(batch)
                    logger.info(f"Lote {i//batch_size + 1}: {len(batch)} vetores")
                except Exception as e:
                    logger.error(f"Erro no lote: {e}")
                    continue
            
            return {
                "success": True,
                "filename": filename,
                "total_chunks": len(chunks),
                "vectors_inserted": total_inserted,
                "chunking_method": "langchain" if LANGCHAIN_AVAILABLE else "manual",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "summary": summary
            }
                
        except Exception as e:
            logger.error(f"Erro na indexação: {e}")
            raise HTTPException(status_code=500, detail=f"Falha na indexação: {str(e)}")
    
    async def process_pdf_hybrid(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Pipeline completo: processamento + resumo garantido + indexação"""
        start_time = datetime.now()
        
        try:
            logger.info(f"=== PROCESSAMENTO HÍBRIDO DE {filename} ===")
            
            # Etapa 1: Extração de texto otimizada
            logger.info("Extraindo texto...")
            text_content = self.extract_text_from_document(file_path)
            
            # Etapa 2: Geração de resumo com fallback garantido
            logger.info("Gerando resumo...")
            summary = self.generate_document_summary(text_content, filename)
            
            # Garantia: sempre ter um resumo válido
            if not summary or len(summary.strip()) < 20:
                summary = self._generate_fallback_summary(text_content, filename)
            
            # Etapa 3: Chunking inteligente
            logger.info("Chunking inteligente...")
            chunks = self.create_smart_chunks(text_content, filename)
            
            if not chunks:
                raise ValueError("Nenhum chunk válido criado")
            
            # Etapa 4: Embeddings em lote otimizado
            logger.info("Gerando embeddings...")
            contents = [chunk["content"] for chunk in chunks]
            embeddings = self.batch_generate_embeddings(contents)
            
            # Etapa 5: Indexação otimizada
            logger.info("Indexando...")
            index_result = self.optimized_pinecone_insert(chunks, embeddings, filename, summary)
            
            # Resultado final com métricas detalhadas
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                **index_result,
                "text_length": len(text_content),
                "processing_time_seconds": round(processing_time, 2),
                "chunks_per_second": round(len(chunks) / processing_time, 2),
                "optimization": "hybrid_langchain_sentence_transformers",
                "avg_chunk_size": round(sum(len(chunk["content"]) for chunk in chunks) / len(chunks), 2)
            }
            
            logger.info(f"=== CONCLUÍDO EM {processing_time:.1f}s ===")
            return final_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Erro após {processing_time:.1f}s: {e}")
            
            # Em caso de erro, retorna resumo básico para não perder completamente
            summary = self._generate_fallback_summary("", filename)
            return {
                "success": False,
                "filename": filename,
                "error": str(e),
                "summary": summary,
                "processing_time_seconds": round(processing_time, 2)
            }

# Instância global
hybrid_processor = HybridDocumentProcessor()

# Função wrapper
async def process_and_index_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    """Versão HÍBRIDA: melhor qualidade de contexto e performance + Resumo garantido"""
    return await hybrid_processor.process_pdf_hybrid(file_path, filename)
