# RAG Database - Base de Conhecimento Científica

Este diretório contém a base de conhecimento para o sistema RAG (Retrieval-Augmented Generation) do GeoBot.

## Estrutura Recomendada

```
rag_database/
├── papers/
│   ├── gravimetry/
│   └── magnetometry/
├── books/
│   ├── blakely_potential_theory.pdf
│   ├── telford_applied_geophysics.pdf
│   └── li_oldenburg_methods.pdf
├── tutorials/
│   ├── harmonica_tutorials/
│   └── fatiando_examples/
└── embeddings/
    └── (gerado automaticamente pelo ChromaDB/FAISS)
```

## Como Adicionar Conteúdo

1. Coloque PDFs de papers científicos em `papers/`
2. Organize por categoria (gravimetria, magnetometria)
3. O sistema automaticamente:
   - Extrairá texto dos PDFs
   - Criará embeddings
   - Indexará para busca semântica

## Fontes Recomendadas

### Livros Fundamentais
- Blakely, R. J. (1995). Potential Theory in Gravity and Magnetic Applications
- Telford, W. M. et al. (1990). Applied Geophysics
- Li, Y. & Oldenburg, D. W. Papers em geofísica de métodos potenciais

### Papers Essenciais
- Redução ao polo (Baranov, Silva)
- Continuação de campos (Jacobsen, Hansen)
- Derivadas direcionais (Verduzco, Cooper)
- Matched filtering (Syberg, Nabighian)

### Repositórios Online
- Fatiando a Terra: https://www.fatiando.org/
- Harmonica: https://www.fatiando.org/harmonica/
- SEG Library: https://library.seg.org/
