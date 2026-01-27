# ❓ FAQ - Perguntas Frequentes

## Instalação e Configuração

### P: Preciso instalar Python ou Node.js?
**R**: Não. GeoBot já vem com todas as dependências incluídas no instalador. É uma aplicação standalone.

### P: Preciso de API keys para usar o GeoBot?
**R**: Sim, pelo menos uma chave de API de IA é necessária para usar o chat assistant. Para processamento geofísico, não são necessárias chaves.

### P: Qual provider de IA devo escolher?
**R**:
- **OpenAI (GPT-4)**: Melhor qualidade, mais caro
- **Anthropic (Claude)**: Excelente para textos longos
- **Google (Gemini)**: Boa relação custo-benefício
- **Groq (Llama 3)**: Gratuito, mais rápido, qualidade inferior

### P: Como obter chave de API gratuita?
**R**: **Groq** oferece API gratuita para Llama 3. Cadastre-se em https://console.groq.com

### P: O Supabase é obrigatório?
**R**: Não. Supabase é apenas para o sistema RAG (citações científicas). Todas as outras features funcionam sem ele.

---

## Uso Geral

### P: Como importar meus dados?
**R**: 
1. Vá para **Projetos**
2. Clique em **Novo Projeto**
3. Arraste arquivos `.xyz`, `.csv`, ou `.grd` para a área de importação

### P: Quais formatos de arquivo são suportados?
**R**:
- **XYZ**: Texto com colunas X Y Z (mais comum)
- **CSV**: Valores separados por vírgula
- **GRD**: Surfer/GMT grid format
- **JSON**: Dados estruturados

### P: Posso processar dados em lote?
**R**: Sim! Use a fila de processamento ou workflows para processar múltiplos datasets automaticamente.

### P: Como salvar meus resultados?
**R**: 
- **Processar → Executar**: Resultado vai para fila
- **Exportar**: Botão de download
- **Projeto**: Salvamento automático

---

## Processamento de Dados

### P: Qual função devo usar para realçar anomalias magnéticas?
**R**: Use o workflow **"Magnetic Enhancement"**:
1. Reduction to Pole
2. Upward Continuation
3. Total Horizontal Derivative
4. Tilt Derivative

### P: Como separar anomalias regionais e residuais?
**R**: Use **Regional-Residual Separation**:
- Método **polynomial**: Para tendências suaves
- Método **upward**: Para fontes profundas

### P: Qual filtro usar para remover ruído?
**R**:
- **Median**: Remove spikes pontuais
- **Gaussian**: Suavização geral
- **Wiener**: Redução ótima de ruído

### P: Como estimar profundidade de fontes?
**R**: Use o workflow **"Depth Estimation"**:
1. Analytic Signal
2. Euler Deconvolution
3. Tilt-Depth Method
4. Source Parameter Imaging

### P: O que é Structural Index (SI)?
**R**: Parâmetro que define o tipo de fonte geológica:
- **SI = 0**: Contato, sill
- **SI = 1**: Dique vertical
- **SI = 2**: Cilindro horizontal
- **SI = 3**: Esfera

### P: Meus resultados estão estranhos. O que fazer?
**R**:
1. Verifique **parâmetros** de entrada
2. Compare **antes/depois**
3. Teste em **subset pequeno**
4. Consulte a **documentação técnica**

---

## Visualização

### P: Como escolher o colormap ideal?
**R**:
- **Viridis/Plasma**: Perceptualmente uniformes, melhores para publicação
- **RdBu**: Divergente, bom para anomalias positivas/negativas
- **Jet**: Evite (ruim para daltonismo)

### P: Como fazer perfis cross-section?
**R**:
1. Ative **Modo Perfil** (ícone régua)
2. Clique em dois pontos no mapa
3. Visualize perfil

### P: Posso exportar mapas em alta resolução?
**R**: Sim:
- **PNG**: 300 DPI para impressão
- **SVG**: Vetorial (editável no Illustrator/Inkscape)
- **PDF**: Para relatórios

### P: Como sobrepor múltiplas camadas?
**R**: Ainda não implementado na v1.0. Planejado para v1.1.

---

## Chat e IA

### P: O que é RAG?
**R**: **Retrieval-Augmented Generation** - Sistema que busca informações em papers científicos para fundamentar respostas.

### P: Como ativar citações científicas?
**R**: 
1. Configure **Supabase** em Configurações
2. Rode script de ingestão: `python scripts/ingest_pdfs.py`
3. Ative **"Use RAG"** no chat

### P: Por que as respostas às vezes demoram?
**R**:
- Modelos grandes (GPT-4) são mais lentos
- RAG adiciona ~2s para busca
- Use **Groq** para respostas mais rápidas

### P: Posso usar o chat offline?
**R**: Não. O chat requer conexão com API de IA. Mas o processamento geofísico funciona offline.

### P: O GeoBot armazena minhas conversas?
**R**: Sim, localmente. Nenhum dado é enviado para servidores (exceto APIs de IA para processamento de mensagens).

---

## Projetos e Dados

### P: Onde ficam salvos meus projetos?
**R**: 
- **Windows**: `C:\Users\[user]\AppData\Local\GeoBot\projects`
- **Linux**: `~/.local/share/GeoBot/projects`

### P: Posso compartilhar projetos com colegas?
**R**: Sim! Use **Exportar Projeto** para criar arquivo `.geobot`. Envie para o colega importar.

### P: Qual o tamanho máximo de dataset?
**R**: 
- **Recomendado**: 1000 × 1000 pontos
- **Máximo**: 10.000 × 10.000 (pode ser lento)

### P: Posso processar dados 3D?
**R**: Não diretamente. GeoBot trabalha com grids 2D (X, Y, Z). Para 3D, fatie em níveis.

---

## Performance

### P: Por que o processamento está lento?
**R**:
1. Dataset muito grande → Reduza resolução
2. Poucos cores → Aumente threads (Configurações)
3. Pouca RAM → Feche outros programas
4. Disco lento → Use SSD

### P: Como acelerar processamento em lote?
**R**:
1. Use **threads máximos** (igual ao número de cores)
2. Ative **cache** de resultados
3. Use **workflows** (evita recalcular etapas)

### P: Meu PC tem 32GB RAM mas só usa 8GB
**R**: Configure em **Configurações → Avançado → Limite de Memória**. Padrão é 8GB.

---

## Erros Comuns

### P: "Failed to fetch" no chat
**R**:
1. Backend não está rodando → Reinicie GeoBot
2. Firewall bloqueando → Adicione exceção
3. Porta 8000 ocupada → Feche outros serviços

### P: "Invalid API Key"
**R**:
1. Copie chave novamente (sem espaços)
2. Verifique validade no console do provider
3. Teste conexão em Configurações

### P: "Out of Memory"
**R**:
1. Reduza tamanho do grid
2. Aumente RAM disponível (Configurações)
3. Processe em tiles menores

### P: Workflow falhou na etapa 3
**R**:
1. Verifique **parâmetros** da etapa
2. Veja **logs** detalhados
3. Execute etapas **individualment**e para debug

---

## Avançado

### P: Posso adicionar minhas próprias funções?
**R**: Sim! (Requer programação Python)
1. Crie arquivo em `backend/app/services/geophysics/functions/custom.py`
2. Siga template das funções existentes
3. Registre no `processing_engine.py`

### P: Como fazer backup dos meus dados?
**R**:
1. **Projetos**: Exporte como `.geobot`
2. **Configurações**: Copie `.env`
3. **Histórico**: Copie pasta `AppData/GeoBot`

### P: Posso rodar GeoBot em servidor?
**R**: Sim! Backend é FastAPI:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### P: Como integrar com scripts Python externos?
**R**: Use a **API REST**:
```python
import requests

response = requests.post('http://localhost:8000/api/processing/execute', json={
    'function_id': 'reduction_to_pole',
    'data': {'x': [...], 'y': [...], 'z': [...]},
    'params': {'inclination': -30, 'declination': 0}
})

result = response.json()
```

### P: Suporta GPU?
**R**: Não na v1.0. Planejado para v2.0 (CUDA para operações FFT).

---

## Licença e Suporte

### P: GeoBot é gratuito?
**R**: Sim, 100% gratuito e open-source. Mas você precisa de chaves de API pagas (OpenAI, etc.).

### P: Posso usar comercialmente?
**R**: Sim, licença MIT permite uso comercial.

### P: Como reportar bugs?
**R**: 
1. **GitHub Issues**: https://github.com/yourusername/geobot/issues
2. **Email**: support@geobot.com
3. Inclua: logs, screenshots, steps to reproduce

### P: Como contribuir?
**R**:
1. Fork o repositório
2. Crie branch (`feature/nova-feature`)
3. Commit changes
4. Pull request

---

## Roadmap

### P: Quais features vêm na v1.1?
**R**:
- Overlay de múltiplas camadas
- Exportar relatórios PDF
- Integração com QGIS
- Plugin system
- GPU acceleration

### P: Quando sai suporte para Mac?
**R**: Planejado para v1.2 (Q2 2026)

### P: Terá versão mobile?
**R**: Versão mobile read-only planejada para v2.0 (viewer de mapas apenas)

---

**Não encontrou sua resposta?**  
📧 Email: support@geobot.com  
💬 Discord: discord.gg/geobot  
📖 Docs: docs.geobot.com
