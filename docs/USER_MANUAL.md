# 📖 Manual do Usuário - GeoBot

## Bem-vindo ao GeoBot

GeoBot é um assistente inteligente para processamento e interpretação de dados geofísicos, combinando algoritmos científicos com inteligência artificial.

---

## 📑 Índice

1. [Instalação](#instalação)
2. [Primeiros Passos](#primeiros-passos)
3. [Interface do Chat](#interface-do-chat)
4. [Processamento de Dados](#processamento-de-dados)
5. [Gerenciamento de Projetos](#gerenciamento-de-projetos)
6. [Visualização de Mapas](#visualização-de-mapas)
7. [Workflows Automáticos](#workflows-automáticos)
8. [Configurações](#configurações)
9. [Solução de Problemas](#solução-de-problemas)

---

## Instalação

### Windows

1. **Baixar o Instalador**
   - Acesse a página de releases
   - Baixe `GeoBot-x.x.x-win-x64.exe`

2. **Instalar**
   - Execute o instalador
   - Escolha o diretório de instalação
   - Crie atalhos (recomendado)
   - Clique em "Instalar"

3. **Executar**
   - Use o atalho da área de trabalho ou
   - Menu Iniciar → GeoBot

### Linux

#### AppImage (Recomendado)

```bash
# Tornar executável
chmod +x GeoBot-x.x.x-linux-x86_64.AppImage

# Executar
./GeoBot-x.x.x-linux-x86_64.AppImage
```

#### Debian/Ubuntu (.deb)

```bash
sudo apt install ./GeoBot_x.x.x_amd64.deb
```

#### Fedora/RHEL (.rpm)

```bash
sudo dnf install ./GeoBot-x.x.x.x86_64.rpm
```

---

## Primeiros Passos

### 1. Configurar API Keys

Na primeira execução, você precisa configurar as chaves de API para os serviços de IA:

1. Abra **Configurações** (ícone de engrenagem)
2. Vá para a aba **API Keys**
3. Insira pelo menos uma chave:
   - **OpenAI**: Para GPT-4 e GPT-3.5
   - **Anthropic**: Para Claude (opcional)
   - **Google**: Para Gemini (opcional)
   - **Groq**: Para Llama 3 (opcional, gratuito)

4. Clique em **Salvar**

#### Como Obter API Keys

**OpenAI**:
- Acesse: https://platform.openai.com/api-keys
- Crie uma nova chave
- Copie e cole no GeoBot

**Anthropic**:
- Acesse: https://console.anthropic.com/
- Crie uma API key
- Copie e cole no GeoBot

**Google (Gemini)**:
- Acesse: https://makersuite.google.com/app/apikey
- Crie uma API key
- Copie e cole no GeoBot

### 2. Configurar Supabase (Opcional - para RAG)

Se você quiser usar o sistema RAG (Retrieval-Augmented Generation) com papers científicos:

1. Crie uma conta no Supabase: https://supabase.com
2. Crie um novo projeto
3. Copie a URL e a chave de API
4. Cole nas configurações do GeoBot

---

## Interface do Chat

### Enviando Mensagens

1. Digite sua pergunta no campo de texto
2. Pressione **Enter** ou clique em **Enviar**
3. Aguarde a resposta do assistente

### Usando RAG (Citações Científicas)

1. Ative a opção **"Use RAG"** acima do campo de mensagem
2. Faça perguntas sobre conceitos geofísicos
3. O GeoBot incluirá citações de papers científicos nas respostas

**Exemplo**:
```
Usuário: "O que é redução ao polo em magnetometria?"
GeoBot: "Redução ao Polo (RTP) é uma técnica que transforma..."
        [1] Blakely (1996): Potential Theory in Gravity...
```

### Histórico de Conversas

- Todas as conversas são salvas automaticamente
- Clique em **"Nova Conversa"** para iniciar um novo chat
- Acesse conversas anteriores na barra lateral

### Comandos Especiais

- `/clear`: Limpa a conversa atual
- `/export`: Exporta conversa como texto
- `/rag on|off`: Liga/desliga RAG

---

## Processamento de Dados

### Funções Disponíveis

GeoBot oferece **24 funções geofísicas** organizadas em 4 categorias:

#### 🌍 Gravimetria (5 funções)

1. **Bouguer Correction**
   - Remove efeito gravitacional da topografia
   - Parâmetros: densidade (g/cm³)
   - Fórmula: BC = 0.04193 × ρ × h

2. **Free-Air Correction**
   - Corrige variação de gravidade com altitude
   - Parâmetros: nenhum
   - Fórmula: FAC = -0.3086 × h

3. **Terrain Correction**
   - Corrige irregularidades topográficas
   - Parâmetros: densidade, raio de busca

4. **Isostatic Correction**
   - Corrige compensação isostática
   - Parâmetros: espessura crustal, densidades

5. **Regional-Residual Separation**
   - Separa anomalias regionais e residuais
   - Métodos: polinomial, upward continuation

#### 🧲 Magnetometria (8 funções)

1. **Reduction to Pole (RTP)**
   - Transforma para magnetização vertical
   - Parâmetros: inclinação, declinação

2. **Upward Continuation**
   - Continua campo para altitudes maiores
   - Parâmetros: altitude (m)

3. **Analytic Signal**
   - Calcula amplitude do sinal analítico
   - Detecta bordas independente de magnetização

4. **Total Horizontal Derivative (THD)**
   - Derivada horizontal total
   - Realça contatos e lineamentos

5. **Vertical Derivative**
   - Derivada vertical de ordem n
   - Parâmetros: ordem (1, 2, 3...)

6. **Tilt Derivative**
   - Ângulo de inclinação do campo
   - Range: -90° a +90°

7. **Pseudogravity**
   - Transforma magnético em gravimétrico
   - Relação de Poisson

8. **Matched Filter**
   - Filtro para profundidade específica
   - Parâmetros: profundidade alvo, SI

#### 🔧 Filtros (5 funções)

1. **Butterworth Filter**
   - Filtro passa-baixa/alta/banda
   - Parâmetros: comprimento de onda, ordem

2. **Gaussian Filter**
   - Suavização gaussiana
   - Parâmetros: sigma

3. **Median Filter**
   - Remove ruídos (spikes)
   - Parâmetros: tamanho da janela

4. **Directional Filter**
   - Realça lineamentos direcionais
   - Parâmetros: azimute (°)

5. **Wiener Filter**
   - Redução ótima de ruído
   - Parâmetros: ruído estimado

#### 🎯 Métodos Avançados (4 funções)

1. **Euler Deconvolution**
   - Estimativa automática de profundidade
   - Parâmetros: SI, tamanho da janela

2. **Source Parameter Imaging (SPI)**
   - Imageamento de parâmetros de fonte
   - Parâmetros: janela de análise

3. **Werner Deconvolution**
   - Profundidade de contatos/diques
   - Perfis 2D

4. **Tilt-Depth Method**
   - Profundidade via zero-crossing do tilt
   - Rápido e robusto

### Como Processar Dados

1. **Abra a página de Processamento**
2. **Selecione uma Função**
   - Navegue pelas categorias
   - Ou use a busca
3. **Configure Parâmetros**
   - Preencha os valores necessários
   - Valores padrão são sugeridos
4. **Execute**
   - Clique em "Executar"
   - Acompanhe o progresso na fila

### Fila de Processamento

- Visualize jobs em execução
- Veja progresso em tempo real
- Exclua jobs pendentes
- Reexecute jobs falhados

### Comparação Antes/Depois

- Ative "Comparar com Original"
- Visualize lado a lado
- Avalie o resultado

---

## Gerenciamento de Projetos

### Criar Novo Projeto

1. Clique em **"+ Novo Projeto"**
2. Preencha:
   - Nome do projeto
   - Descrição
   - Tags (opcional)
3. Clique em **"Criar"**

### Organizar Arquivos

- **Arrastar e soltar** para adicionar arquivos
- Crie **pastas** para organizar
- Use **tags** para categorizar
- **Pesquise** por nome ou tag

### Metadados

Cada arquivo pode ter:
- Tags personalizadas
- Data de criação/modificação
- Informações geofísicas (rows, cols, unidade)
- Parâmetros de processamento

### Exportar/Importar

**Exportar Projeto**:
- Formato: `.geobot` (ZIP)
- Inclui todos os arquivos e metadados

**Importar Projeto**:
- Arraste `.geobot` para a área de importação

---

## Visualização de Mapas

### Tipos de Plot

1. **Contour** (Isolinhas)
   - Linhas de contorno
   - Ideal para interpretação

2. **Filled Contour** (Contorno Preenchido)
   - Cores entre linhas
   - Visual intuitivo

3. **Heatmap** (Mapa de Calor)
   - Grid com cores
   - Valores discretos

4. **3D Surface** (Superfície 3D)
   - Visualização tridimensional
   - Rotação e zoom

### Colormaps

12 paletas disponíveis:
- **Viridis**: Perceptualmente uniforme
- **Plasma**: Alto contraste
- **Jet**: Clássico (evite para daltonismo)
- **Rainbow**: Colorido
- **RdBu**: Divergente (vermelho-azul)
- **RdYlGn**: Divergente (vermelho-verde)
- E outros...

**Inverter Colormap**: Use a opção "Reverso"

### Ajustes

- **Range Z**: Defina min/max manualmente
- **Níveis de Contorno**: 5 a 50
- **Colorbar**: Mostre/oculha legenda
- **Grid**: Linhas de grade
- **Aspecto**: Proporção igual ou auto

### Perfis

1. Ative **Modo Perfil** (ícone régua)
2. Clique em dois pontos no mapa
3. Visualize perfil cross-section

### Exportar Imagem

- **PNG**: Para apresentações
- **SVG**: Vetorial (editável)
- **JSON**: Dados brutos

---

## Workflows Automáticos

### Workflows Pré-Configurados

#### 1. Magnetic Enhancement
```
Etapas:
1. Reduction to Pole
2. Upward Continuation (500m)
3. Total Horizontal Derivative
4. Tilt Derivative
```

#### 2. Gravity Reduction
```
Etapas:
1. Free-Air Correction
2. Bouguer Correction
3. Terrain Correction
4. Regional-Residual
```

#### 3. Depth Estimation
```
Etapas:
1. Analytic Signal
2. Euler Deconvolution
3. Tilt-Depth Method
4. SPI
```

#### 4. Data Filtering
```
Etapas:
1. Median Filter (remove spikes)
2. Gaussian Smoothing
3. Directional Filter (realce)
```

### Criar Workflow Customizado

1. Clique em **"Novo Workflow"**
2. **Arraste funções** da paleta
3. **Conecte** as etapas
4. **Configure parâmetros** de cada função
5. **Salve** com nome descritivo
6. **Execute** no dataset

### Executar Workflow

1. Selecione dados de entrada
2. Escolha workflow
3. Clique em **"Executar"**
4. Acompanhe progresso
5. Visualize resultados intermediários

---

## Configurações

### API Keys
- Gerencie chaves de API
- Teste conexões
- Revogue chaves antigas

### Preferências
- **Idioma**: Português/English
- **Tema**: Claro/Escuro
- **Unidades**: SI/Imperial

### Cache
- Limpe cache de processamento
- Libere espaço em disco

### Avançado
- **Threads**: Número de CPUs para processamento
- **Memória**: Limite de RAM
- **Logs**: Nível de detalhe

---

## Solução de Problemas

### Erro: "API Key Inválida"

**Solução**:
1. Verifique se a chave foi copiada corretamente
2. Sem espaços no início/fim
3. Chave não expirada
4. Saldo disponível (OpenAI/Anthropic)

### Erro: "Servidor Não Conecta"

**Solução**:
1. Verifique se porta 8000 está livre
2. Firewall não está bloqueando
3. Reinicie o GeoBot
4. Windows: Execute como administrador

### Processamento Muito Lento

**Solução**:
1. Reduza tamanho do grid
2. Use menos threads (Configurações → Avançado)
3. Feche outros programas
4. Aumente memória disponível

### RAG Não Retorna Citações

**Solução**:
1. Configure Supabase corretamente
2. Rode script de ingestão de PDFs
3. Verifique conexão com Supabase
4. Logs: `backend/geobot.log`

### Mapas Não Carregam

**Solução**:
1. Verifique formato dos dados
2. Dados devem ter x, y, z
3. Grid regular (nx × ny)
4. Console do navegador (F12) para erros

---

## Atalhos de Teclado

### Global
- `Ctrl + N`: Nova conversa
- `Ctrl + S`: Salvar projeto
- `Ctrl + E`: Exportar
- `Ctrl + ,`: Configurações
- `F11`: Tela cheia

### Chat
- `Enter`: Enviar mensagem
- `Shift + Enter`: Nova linha
- `Ctrl + K`: Limpar conversa
- `Ctrl + /`: Comandos

### Processamento
- `Ctrl + F`: Buscar função
- `Ctrl + Enter`: Executar
- `Esc`: Cancelar

---

## Dicas e Melhores Práticas

### Organização

✅ **Faça**:
- Use nomes descritivos para projetos
- Organize em pastas (Raw, Processed, Results)
- Adicione tags relevantes
- Documente parâmetros nos metadados

❌ **Evite**:
- Nomes genéricos ("data1", "test")
- Arquivos soltos sem organização
- Processar sem salvar projeto

### Processamento

✅ **Faça**:
- Teste em subset pequeno primeiro
- Use workflows para consistência
- Salve resultados intermediários
- Compare antes/depois

❌ **Evite**:
- Processar grid inteiro sem testar
- Aplicar funções em ordem errada
- Ignorar range válido de parâmetros

### Visualização

✅ **Faça**:
- Use colormaps adequados
- Ajuste range Z para destaque
- Exporte em alta resolução
- Use contornos para interpretação

❌ **Evite**:
- Jet colormap (ruim para daltonismo)
- Range Z automático sem revisar
- PNG de baixa resolução

---

## Formatos de Arquivo Suportados

### Importação

- **XYZ**: Texto com colunas X Y Z
- **CSV**: Comma-separated values
- **GRD**: Grid format (Surfer/GMT)
- **NetCDF**: Formato científico
- **JSON**: Dados estruturados

### Exportação

- **XYZ**: Texto simples
- **CSV**: Excel-compatível
- **JSON**: Metadados completos
- **PNG/SVG**: Imagens
- **PDF**: Relatórios

---

## Recursos Adicionais

- **Documentação Técnica**: `/docs/DEVELOPER.md`
- **API Reference**: `http://localhost:8000/docs`
- **Issues**: GitHub Issues
- **Forum**: Comunidade GeoBot

---

## Glossário

**RTP**: Reduction to Pole - Redução ao Polo
**THD**: Total Horizontal Derivative
**SPI**: Source Parameter Imaging
**RAG**: Retrieval-Augmented Generation
**SI**: Structural Index (índice estrutural)
**FAC**: Free-Air Correction
**BC**: Bouguer Correction

---

**Versão**: 1.0.0  
**Última atualização**: Janeiro 2026  
**Suporte**: support@geobot.com
