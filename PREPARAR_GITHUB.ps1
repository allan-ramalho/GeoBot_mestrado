# ============================================================================
# GeoBot - Preparação para GitHub
# ============================================================================
# Este script prepara o repositório para push ao GitHub
# Execute ANTES de fazer push para garantir segurança
# ============================================================================

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "   PREPARANDO REPOSITÓRIO GEOBOT PARA GITHUB" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"
$repoPath = $PSScriptRoot

# ============================================================================
# 1. VERIFICAÇÃO DE ARQUIVOS SENSÍVEIS
# ============================================================================
Write-Host "1️⃣  Verificando arquivos sensíveis..." -ForegroundColor Yellow

$sensitiveFiles = @(
    ".env",
    "geobot.log",
    "*.pyc",
    "__pycache__",
    "venv",
    "rag_database/chromadb"
)

$foundSensitive = $false
foreach ($pattern in $sensitiveFiles) {
    $files = Get-ChildItem -Path $repoPath -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    if ($files) {
        Write-Host "   ⚠️  Encontrado: $pattern" -ForegroundColor Yellow
        $foundSensitive = $true
    }
}

if (-not $foundSensitive) {
    Write-Host "   ✅ Nenhum arquivo sensível encontrado no Git" -ForegroundColor Green
}

# ============================================================================
# 2. VERIFICA SE .ENV ESTÁ NO .GITIGNORE
# ============================================================================
Write-Host "`n2️⃣  Verificando .gitignore..." -ForegroundColor Yellow

$gitignorePath = Join-Path $repoPath ".gitignore"
if (Test-Path $gitignorePath) {
    $gitignoreContent = Get-Content $gitignorePath -Raw
    if ($gitignoreContent -match "\.env") {
        Write-Host "   ✅ .env está no .gitignore" -ForegroundColor Green
    } else {
        Write-Host "   ❌ AVISO: .env NÃO está no .gitignore!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "   ❌ ERRO: .gitignore não encontrado!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 3. VERIFICA SE .ENV.EXAMPLE EXISTE
# ============================================================================
Write-Host "`n3️⃣  Verificando .env.example..." -ForegroundColor Yellow

$envExamplePath = Join-Path $repoPath ".env.example"
if (Test-Path $envExamplePath) {
    Write-Host "   ✅ .env.example encontrado" -ForegroundColor Green
    
    # Verifica se não tem chaves reais
    $envExampleContent = Get-Content $envExamplePath -Raw
    if ($envExampleContent -match "gsk_|sk-proj-|your_.*_here") {
        if ($envExampleContent -match "your_.*_here") {
            Write-Host "   ✅ .env.example é um template seguro" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  AVISO: .env.example pode conter chaves reais!" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "   ❌ ERRO: .env.example não encontrado!" -ForegroundColor Red
    Write-Host "   Execute: copy .env .env.example" -ForegroundColor Yellow
    Write-Host "   E substitua as chaves reais por placeholders" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# 4. LIMPA ARQUIVOS DESNECESSÁRIOS
# ============================================================================
Write-Host "`n4️⃣  Limpando arquivos desnecessários..." -ForegroundColor Yellow

# Remove __pycache__
Get-ChildItem -Path $repoPath -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
Write-Host "   ✅ Removidos diretórios __pycache__" -ForegroundColor Green

# Remove .pyc
Get-ChildItem -Path $repoPath -Filter "*.pyc" -Recurse | Remove-Item -Force
Write-Host "   ✅ Removidos arquivos .pyc" -ForegroundColor Green

# Remove logs
if (Test-Path (Join-Path $repoPath "geobot.log")) {
    Remove-Item (Join-Path $repoPath "geobot.log") -Force
    Write-Host "   ✅ Removido geobot.log" -ForegroundColor Green
}

# ============================================================================
# 5. VERIFICA STATUS DO GIT
# ============================================================================
Write-Host "`n5️⃣  Verificando repositório Git..." -ForegroundColor Yellow

Set-Location $repoPath

if (-not (Test-Path ".git")) {
    Write-Host "   ⚠️  Repositório Git não inicializado!" -ForegroundColor Yellow
    Write-Host "   Inicializando..." -ForegroundColor Cyan
    git init
    Write-Host "   ✅ Git inicializado" -ForegroundColor Green
}

# Verifica se há remote
$remotes = git remote
if (-not $remotes) {
    Write-Host "   ⚠️  Nenhum remote configurado" -ForegroundColor Yellow
    Write-Host "   Configure com:" -ForegroundColor Cyan
    Write-Host "   git remote add origin https://github.com/allan-ramalho/GeoBot_mestrado.git" -ForegroundColor White
} else {
    Write-Host "   ✅ Remote configurado: $remotes" -ForegroundColor Green
}

# ============================================================================
# 6. MOSTRA STATUS
# ============================================================================
Write-Host "`n6️⃣  Status do repositório:" -ForegroundColor Yellow
git status --short

# ============================================================================
# RESUMO FINAL
# ============================================================================
Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "   ✅ REPOSITÓRIO PREPARADO PARA PUSH!" -ForegroundColor Green
Write-Host "============================================================================`n" -ForegroundColor Cyan

Write-Host "📋 Próximos passos para fazer push:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Configure o remote (se ainda não configurou):" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/allan-ramalho/GeoBot_mestrado.git" -ForegroundColor White
Write-Host ""
Write-Host "2. Faça o commit de todas as mudanças:" -ForegroundColor Yellow
Write-Host "   git add ." -ForegroundColor White
Write-Host '   git commit -m "feat: Implementação completa com aceleração GPU"' -ForegroundColor White
Write-Host ""
Write-Host "3. Faça push FORÇADO (substitui todo o repositório remoto):" -ForegroundColor Yellow
Write-Host "   git push -f origin main" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  IMPORTANTE: O push com -f substitui TODO o histórico remoto!" -ForegroundColor Red
Write-Host "   Use apenas se tiver certeza!" -ForegroundColor Red
Write-Host ""
Write-Host "✨ Após o push, seu repositório estará em:" -ForegroundColor Cyan
Write-Host "   https://github.com/allan-ramalho/GeoBot_mestrado" -ForegroundColor White
Write-Host ""
