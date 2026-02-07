# 🚀 Preparação para GitHub

Este documento descreve os passos para preparar o repositório GeoBot para upload no GitHub.

---

## ✅ Checklist de Preparação

### 1. Arquivos Removidos
- ✅ `test_sidebar.py` (arquivo de teste)
- ✅ `test_geobot.py` (arquivo de teste)
- ✅ `geobot.log` (logs temporários)
- ✅ `__pycache__/` (cache Python)

### 2. Documentação Criada/Atualizada
- ✅ `README.md` - Documentação principal moderna com badges
- ✅ `DEVELOPER_GUIDE.md` - Guia lúdico para desenvolvedores
- ✅ `USER_GUIDE.md` - Manual simplificado para usuários
- ✅ `CONTRIBUTING.md` - Guia de contribuição
- ✅ `.gitignore` - Atualizado e organizado

### 3. Código Otimizado
- ✅ Suporte GPU adicionado (NVIDIA CUDA e Apple Silicon)
- ✅ Detecção automática de dispositivo
- ✅ Fallback para CPU quando GPU não disponível

### 4. Arquivos Mantidos
- ✅ `geobot.py` - Aplicação principal
- ✅ `requirements.txt` - Dependências
- ✅ `INSTALAR.bat` - Instalador Windows
- ✅ `INICIAR_GEOBOT.bat` - Launcher Windows
- ✅ `LICENSE` - Licença MIT
- ✅ `DOCUMENTACAO.md` - Documentação técnica original
- ✅ `example_data/` - Dados de exemplo
- ✅ `assets/` - Logos e imagens
- ✅ `.streamlit/config.toml` - Configurações Streamlit

---

## 📤 Comandos para GitHub

### Opção 1: Criar Novo Repositório

```bash
# 1. Inicialize Git (se ainda não foi inicializado)
cd c:\Users\AllanRamalho\Desktop\GeoBot\GeoBot_Mestrado
git init

# 2. Adicione todos os arquivos
git add .

# 3. Faça o primeiro commit
git commit -m "feat: versão inicial do GeoBot com suporte GPU e documentação completa"

# 4. Crie repositório no GitHub
# Acesse: https://github.com/new
# Nome sugerido: GeoBot
# Descrição: "🌍 Agente de IA conversacional para processamento de dados geofísicos"
# Público ou Privado: conforme preferência
# NÃO inicialize com README (já temos um)

# 5. Adicione remote do GitHub (substitua SEU-USUARIO)
git remote add origin https://github.com/SEU-USUARIO/GeoBot.git

# 6. Push para GitHub
git branch -M main
git push -u origin main
```

### Opção 2: Substituir Repositório Existente

```bash
cd c:\Users\AllanRamalho\Desktop\GeoBot\GeoBot_Mestrado

# 1. Verifique remote atual
git remote -v

# 2. Se já existe remote 'origin', remova
git remote remove origin

# 3. Adicione novo remote (substitua URL pelo seu repositório)
git remote add origin https://github.com/SEU-USUARIO/GeoBot.git

# 4. Adicione mudanças
git add .

# 5. Commit
git commit -m "refactor: reestruturação completa com GPU, documentação e limpeza"

# 6. Force push (CUIDADO: sobrescreve repositório remoto!)
git push -f origin main
```

### Opção 3: Push Incremental

```bash
cd c:\Users\AllanRamalho\Desktop\GeoBot\GeoBot_Mestrado

# 1. Adicione mudanças
git add .

# 2. Commit
git commit -m "docs: adiciona README, DEVELOPER_GUIDE, USER_GUIDE e CONTRIBUTING

- ✨ README modernizado com badges e exemplos
- 🛠️ DEVELOPER_GUIDE com templates práticos
- 📘 USER_GUIDE simplificado para iniciantes
- 🤝 CONTRIBUTING com padrões de código
- 🚀 Suporte GPU (NVIDIA CUDA e Apple Silicon)
- 🧹 Limpeza de arquivos temporários
- 📝 .gitignore atualizado e organizado"

# 3. Push
git push origin main
```

---

## 🔧 Configurações Recomendadas do GitHub

### Sobre o Repositório

**Nome:** `GeoBot`

**Descrição:**
```
🌍 Agente de IA conversacional para processamento de dados geofísicos (gravimetria e magnetometria) com suporte GPU
```

**Website:** `https://ppgdot-uff.com.br/`

**Topics (Tags):**
```
geophysics
artificial-intelligence
streamlit
python
llm
rag
groq-api
gpu
gravity
magnetometry
pytorch
scientific-computing
```

### README Badges

Os badges já estão incluídos no README.md:
- ![Python](https://img.shields.io/badge/Python-3.11.9-blue)
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B)
- ![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Ready-EE4C2C)
- ![License](https://img.shields.io/badge/License-MIT-green.svg)

### Configurações do Repositório

1. **Settings → General:**
   - ✅ Issues habilitados
   - ✅ Discussions habilitados (recomendado)
   - ✅ Wiki desabilitado (usamos docs no repo)

2. **Settings → Branches:**
   - Branch padrão: `main`
   - Proteção de branch (opcional para projeto pessoal)

3. **Settings → GitHub Pages (opcional):**
   - Source: Deploy from branch
   - Branch: `main` / docs
   - Pode hospedar documentação estática

---

## 📋 Checklist Final Antes do Push

- [ ] `.gitignore` está correto e completo
- [ ] Não há arquivos sensíveis (API keys, .env)
- [ ] Todos os arquivos temporários foram removidos
- [ ] README.md está completo e sem erros
- [ ] Links no README apontam para URLs corretos
- [ ] LICENSE está presente (MIT)
- [ ] Código está funcionando localmente
- [ ] Documentação está atualizada

---

## 🎨 Melhorias Pós-Upload

Após fazer upload para o GitHub, considere:

### 1. Adicionar GitHub Actions (CI/CD)

Crie `.github/workflows/python-app.yml`:

```yaml
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.11
      uses: actions/setup-python@v3
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### 2. Adicionar Templates de Issues

Crie `.github/ISSUE_TEMPLATE/bug_report.md` e `feature_request.md`

### 3. Criar Releases

Após estabilizar, crie releases versionadas:
- `v1.0.0` - Versão inicial
- `v1.1.0` - Novas funcionalidades
- `v1.0.1` - Correções de bugs

### 4. Adicionar GIFs/Screenshots

Capture screenshots da interface:
- Página inicial
- Upload de dados
- Chat funcionando
- Mapas gerados

Adicione na pasta `docs/screenshots/`

---

## 📞 Suporte

Se tiver dúvidas sobre o processo:

- 📧 Email: allansoares@id.uff.br
- 📖 Docs GitHub: [docs.github.com](https://docs.github.com)

---

<div align="center">

**Pronto para compartilhar com o mundo! 🚀**

</div>
