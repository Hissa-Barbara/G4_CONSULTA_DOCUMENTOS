import React, { useState } from 'react';
import { MessageSquare, User, } from 'lucide-react';
import GoogleLoginButton from '../GoogleLoginButton/GoogleLoginButton';
import './LoginView.css';

const LoginView = ({ 
  API_BASE_URL,
  onLoginSuccess,
  isLoading,
  setIsLoading
}) => {
  // === ESTADOS LOCAIS DO COMPONENTE ===
  // Estado para credenciais de login tradicional
  const [credentials, setCredentials] = useState({ 
    email: '', 
    password: '' 
  });
  // Estado para dados de registro de novo usuário
  const [registerData, setRegisterData] = useState({ 
    name: '', 
    email: '', 
    password: '', 
    confirmPassword: '' 
  });
  // Estado para controlar qual formulário está ativo (login ou registro)
  const [loginMode, setLoginMode] = useState('traditional'); // 'traditional' ou 'register'

  // === FUNÇÕES DE VALIDAÇÃO ===
  // Faz parse seguro de resposta para evitar erro quando backend retorna corpo vazio.
  const parseResponseData = async (response) => {
    const rawText = await response.text();
    if (!rawText) return {};

    try {
      return JSON.parse(rawText);
    } catch {
      return { detail: rawText };
    }
  };

  // Valida formato de email usando regex
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // Valida formulário de login antes do envio
  const validateLoginForm = () => {
    if (!credentials.email || !credentials.password) {
      alert('Por favor, preencha email e senha!');
      return false;
    }

    if (!validateEmail(credentials.email)) {
      alert('Por favor, insira um email válido!');
      return false;
    }

    return true;
  };

  // Valida formulário de registro com múltiplas verificações
  const validateRegisterForm = () => {
    // Validação de campos obrigatórios
    if (!registerData.name || !registerData.email || !registerData.password) {
      alert('Preencha todos os campos!');
      return false;
    }
    
    // Validação de nome
    if (registerData.name.trim().length < 2) {
      alert('O nome deve ter pelo menos 2 caracteres!');
      return false;
    }
    
    // Validação de email
    if (!validateEmail(registerData.email)) {
      alert('Por favor, insira um email válido!');
      return false;
    }
    
    // Validação de senha
    if (registerData.password.length < 6) {
      alert('A senha deve ter pelo menos 6 caracteres!');
      return false;
    }
    
    // Validação de confirmação de senha
    if (registerData.password !== registerData.confirmPassword) {
      alert('As senhas não coincidem!');
      return false;
    }

    return true;
  };

  // === FUNÇÃO DE LOGIN TRADICIONAL ===
  // Processa login com email e senha
  const handleLogin = async () => {
    if (!validateLoginForm()) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/login/`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/x-www-form-urlencoded' 
        },
        body: new URLSearchParams({ 
          username: credentials.email, 
          password: credentials.password 
        }).toString(),
      });
      
      const data = await parseResponseData(response);
      
      if (response.ok) {
        // Sucesso - chama callback do App.js para atualizar estado global
        onLoginSuccess(data.user, data.access_token);
        
        // Limpa formulário após sucesso
        setCredentials({ email: '', password: '' });
      } else {
        // Erro - mostra mensagem de erro específica
        alert('Erro no login: ' + (data.detail || 'Email ou senha incorretos.'));
      }
    } catch (error) {
      console.error('Erro no login:', error);
      alert('Erro de conexão. Verifique sua internet e tente novamente.');
    } finally {
      setIsLoading(false);
    }
  };

  // === FUNÇÃO DE REGISTRO ===
  // Processa criação de nova conta de usuário
  const handleRegister = async () => {
    if (!validateRegisterForm()) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/login/register`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({
          name: registerData.name.trim(),
          email: registerData.email.toLowerCase(),
          password: registerData.password
        }),
      });
      
      const data = await parseResponseData(response);
      
      if (response.ok) {
        // Sucesso - mostra mensagem e redireciona para login
        alert('🎉 Cadastro realizado com sucesso!\n\nAgora você pode fazer login com suas credenciais.');
        
        // Limpa formulário de registro
        setRegisterData({ 
          name: '', 
          email: '', 
          password: '', 
          confirmPassword: '' 
        });
        
        // Volta para tela de login
        setLoginMode('traditional');
        
        // Pré-preenche email no formulário de login
        setCredentials({ 
          email: registerData.email, 
          password: '' 
        });
      } else {
        // Erro - mostra mensagem específica do backend
        const errorMessage = data.detail || 'Erro desconhecido no cadastro.';
        alert('Erro no cadastro: ' + errorMessage);
      }
    } catch (error) {
      console.error('Erro no cadastro:', error);
      alert('Erro de conexão. Verifique sua internet e tente novamente.');
    } finally {
      setIsLoading(false);
    }
  };

  // === FUNÇÕES DO GOOGLE LOGIN ===
  // Processa login bem-sucedido via Google OAuth
  const handleGoogleSuccess = async (userData) => {
    setIsLoading(true);
    try {
      console.log('Google Login Success:', userData);
      
      const response = await fetch(`${API_BASE_URL}/api/login/google`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify(userData),
      });
      
      const data = await parseResponseData(response);
      
      if (response.ok) {
        // Sucesso - chama callback do App.js
        onLoginSuccess(data.user, data.access_token);
      } else {
        alert('Erro no login com Google: ' + (data.detail || 'Erro desconhecido.'));
      }
    } catch (error) {
      console.error('Erro no login com Google:', error);
      alert('Erro de conexão com Google. Tente novamente.');
    } finally {
      setIsLoading(false);
    }
  };

  // Trata erros do Google OAuth
  const handleGoogleError = (error) => {
    console.error('Google Login Error:', error);
    alert('Falha no login com Google. Tente novamente.');
  };

  // === FUNÇÃO PARA TESTE RÁPIDO (DESENVOLVIMENTO) ===
  // Preenche credenciais rapidamente para testes
  const handleQuickLogin = (email, password) => {
    setCredentials({ email, password });
    // Pequeno delay para mostrar que preencheu
    setTimeout(() => {
      handleLogin();
    }, 100);
  };

  // === HANDLERS DE FORMULÁRIO ===
  // Permite envio de formulário com Enter
  const handleKeyDown = (e, action) => {
    if (e.key === 'Enter' && !isLoading) {
      e.preventDefault();
      action();
    }
  };

  // Gerencia troca entre abas de login e registro
  const handleTabSwitch = (mode) => {
    setLoginMode(mode);
    // Limpa formulários ao trocar de aba para evitar confusão
    if (mode === 'traditional') {
      setRegisterData({ name: '', email: '', password: '', confirmPassword: '' });
    } else {
      setCredentials({ email: '', password: '' });
    }
  };

  // === RENDER ===
  return (
    <div className="login-container">
      {/* Animação de fundo decorativa */}
      <div className="background-animation">
        <div className="floating-element element-1"></div>
        <div className="floating-element element-2"></div>
        <div className="floating-element element-3"></div>
        <div className="floating-element element-4"></div>
      </div>

      <div className="login-card">
        {/* Cabeçalho com branding do sistema */}
        <div className="login-header">
          <div className="logo-container">
            <MessageSquare size={40} />
          </div>
          <h1 className="system-title">ConsultAI</h1>
          <p className="system-subtitle">Consultas Inteligentes de Documentos</p>
          <p className="document-version">Resoluções da UFMA</p>
        </div>

        {/* Conteúdo principal com formulários */}
        <div className="login-content">
          {/* Botões de navegação entre Login e Cadastro */}
          <div className="tab-buttons">
            <button
              onClick={() => handleTabSwitch('traditional')}
              className={`tab-button ${loginMode === 'traditional' ? 'active-tab-blue' : 'inactive-tab'}`}
              disabled={isLoading}
            >
              Login
            </button>
            <button
              onClick={() => handleTabSwitch('register')}
              className={`tab-button ${loginMode === 'register' ? 'active-tab-green' : 'inactive-tab'}`}
              disabled={isLoading}
            >
              Cadastro
            </button>
          </div>

          {/* Formulário de Login */}
          {loginMode === 'traditional' ? (
            <div className="form-container">
              <div className="form-group">
                <label className="form-label">Email</label>
                <input
                  type="email"
                  className="form-input"
                  placeholder="Digite seu email"
                  value={credentials.email}
                  onChange={(e) => setCredentials({...credentials, email: e.target.value})}
                  onKeyDown={(e) => handleKeyDown(e, handleLogin)}
                  disabled={isLoading}
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Senha</label>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Digite sua senha"
                  value={credentials.password}
                  onChange={(e) => setCredentials({...credentials, password: e.target.value})}
                  onKeyDown={(e) => handleKeyDown(e, handleLogin)}
                  disabled={isLoading}
                />
              </div>
              
              <div className="button-group">
                {/* Botão principal de login */}
                <button
                  onClick={handleLogin}
                  disabled={isLoading}
                  className="primary-button"
                >
                  {isLoading ? (
                    <div className="loading-content">
                      <div className="spinner"></div>
                      Entrando...
                    </div>
                  ) : (
                    <>
                      <User className="button-icon" />
                      Entrar
                    </>
                  )}
                </button>
                
                {/* Componente de login com Google */}
                <GoogleLoginButton
                  clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}
                  onSuccess={handleGoogleSuccess}
                  onError={handleGoogleError}
                  disabled={isLoading}
                  buttonText="Entrar com Google"
                />
                  {/* DEBUG EXPANDIDO */}

                
                {/* Seção de teste rápido com credenciais pré-definidas */}
                <div>
                  <p className="info-text">
              
                    
                  </p>
                  <div className="admin-credentials">
                    <div>
                      <span 
                        onClick={() => !isLoading && handleQuickLogin('admin@ufma.br', 'admin123')}
                        style={{ 
                          cursor: isLoading ? 'not-allowed' : 'pointer',
                          textDecoration: 'underline',
                          opacity: isLoading ? 0.5 : 1
                        }}
                      >
              
                      </span>
                    </div>
                    <div>
                      <span 
                        onClick={() => !isLoading && handleQuickLogin('usuario@gmai.com', 'user123')}
                        style={{ 
                          cursor: isLoading ? 'not-allowed' : 'pointer',
                          textDecoration: 'underline',
                          opacity: isLoading ? 0.5 : 1
                        }}
                      >
                        
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Formulário de Registro */
            <div className="form-container">
              <div className="form-group">
                <label className="form-label">Nome Completo</label>
                <input
                  type="text"
                  className="form-input-green"
                  placeholder="Digite seu nome completo"
                  value={registerData.name}
                  onChange={(e) => setRegisterData({...registerData, name: e.target.value})}
                  disabled={isLoading}
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Email</label>
                <input
                  type="email"
                  className="form-input-green"
                  placeholder="Digite seu email"
                  value={registerData.email}
                  onChange={(e) => setRegisterData({...registerData, email: e.target.value})}
                  disabled={isLoading}
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Senha</label>
                <input
                  type="password"
                  className="form-input-green"
                  placeholder="Digite sua senha (mín. 6 caracteres)"
                  value={registerData.password}
                  onChange={(e) => setRegisterData({...registerData, password: e.target.value})}
                  disabled={isLoading}
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Confirmar Senha</label>
                <input
                  type="password"
                  className="form-input-green"
                  placeholder="Confirme sua senha"
                  value={registerData.confirmPassword}
                  onChange={(e) => setRegisterData({...registerData, confirmPassword: e.target.value})}
                  onKeyDown={(e) => handleKeyDown(e, handleRegister)}
                  disabled={isLoading}
                />
              </div>
              
              {/* Botão de criação de conta */}
              <button
                onClick={handleRegister}
                disabled={isLoading}
                className="register-button"
              >
                {isLoading ? (
                  <div className="loading-content">
                    <div className="spinner"></div>
                    Criando conta...
                  </div>
                ) : (
                  <>
                    <User className="button-icon" />
                    Criar Conta
                  </>
                )}
              </button>
              
            </div>
          )}
        </div>

        {/* Rodapé com termos de uso */}
        <div className="terms-text">
          <p>Ao fazer login, você concorda com nossos termos de uso</p>
        </div>
      </div>
    </div>
  );
};

export default LoginView;