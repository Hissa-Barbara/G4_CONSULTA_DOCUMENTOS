# Importações necessárias do FastAPI e bibliotecas auxiliares
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # Para carregar variáveis de ambiente do .env
import logging  # Para registro de logs
from routes import admin, chat, history, login, admin_requests, admin_management  # Importação das rotas
import os

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Cria a aplicação FastAPI
app = FastAPI()

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

# Configuração do middleware CORS (Cross-Origin Resource Sharing)
# Isso permite que o frontend (mesmo hospedado em outro domínio) acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Permite origens específicas via variável de ambiente
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os headers
    expose_headers=["Content-Disposition"]  # Exposição de headers específicos (útil para downloads)
)

# Configuração do sistema de logging (registro de eventos e erros)
logging.basicConfig(
    level=logging.INFO,  # Nível mínimo de log (INFO, DEBUG, WARNING, ERROR, etc.)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Formato do log
)
logger = logging.getLogger(__name__)  # Criação do logger para este módulo

# Inclusão das rotas da aplicação com seus respectivos prefixos e tags
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(login.router, prefix="/api/login", tags=["Login"])
app.include_router(admin_requests.router, prefix="/api/admin-requests", tags=["Admin Requests"])
app.include_router(admin_management.router, prefix="/api/admin-management", tags=["Admin Management"])

# Rota de verificação de status da API
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "services": ["admin", "chat", "history", "login", "admin-requests", "admin-management"]
    }

# Bloco para execução local do servidor com Uvicorn
# Este bloco só será executado se o script for rodado diretamente (ex: python main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Inicia o servidor na porta 8000 acessível de qualquer IP
