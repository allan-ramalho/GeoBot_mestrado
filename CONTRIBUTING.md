# 🤝 Guia de Contribuição - GeoBot

Obrigado por considerar contribuir para o GeoBot! 🎉

Este documento descreve como você pode ajudar a melhorar o projeto.

---

## 📋 Índice

1. [Como Posso Contribuir?](#-como-posso-contribuir)
2. [Reportando Bugs](#-reportando-bugs)
3. [Sugerindo Funcionalidades](#-sugerindo-funcionalidades)
4. [Contribuindo com Código](#-contribuindo-com-código)
5. [Padrões de Código](#-padrões-de-código)
6. [Processo de Pull Request](#-processo-de-pull-request)
7. [Licença](#-licença)

---

## 💡 Como Posso Contribuir?

Existem várias formas de contribuir, mesmo se você não é programador:

### 🐛 Reportar Bugs
Encontrou um erro? Ajude-nos a corrigi-lo!

### 💡 Sugerir Funcionalidades
Tem uma ideia para melhorar o GeoBot? Compartilhe!

### 📚 Melhorar Documentação
Correções, exemplos, tutoriais - toda ajuda é bem-vinda!

### 🔬 Adicionar Processamentos
Implemente novos métodos geofísicos

### 🧪 Testar e Validar
Use o GeoBot com seus dados e reporte experiência

### 🌍 Traduzir
Ajude a traduzir para outros idiomas

### ⭐ Dar Estrela
Se o projeto te ajudou, deixe uma estrela no GitHub!

---

## 🐛 Reportando Bugs

Antes de reportar um bug, verifique se ele já não foi reportado em [Issues](https://github.com/seu-usuario/GeoBot/issues).

### Como Reportar um Bom Bug Report

Use este template:

```markdown
**Descrição do Bug**
Descrição clara do problema

**Para Reproduzir**
Passos para reproduzir:
1. Carregue arquivo X
2. Execute comando Y
3. Veja erro Z

**Comportamento Esperado**
O que deveria acontecer

**Comportamento Atual**
O que realmente aconteceu

**Screenshots**
Se aplicável, adicione screenshots

**Ambiente:**
- OS: [Windows 11, Ubuntu 22.04, macOS 14]
- Python: [3.11.9]
- GeoBot: [versão]
- GPU: [NVIDIA RTX 3080 / Não / Apple M2]

**Dados de Exemplo**
Se possível, anexe um arquivo CSV pequeno que reproduza o erro

**Logs**
Copie o conteúdo de `geobot.log` se relevante
```

---

## 💡 Sugerindo Funcionalidades

Tem uma ideia? Abra uma [Issue](https://github.com/seu-usuario/GeoBot/issues) com a tag `enhancement`.

### Template de Sugestão

```markdown
**Funcionalidade Desejada**
Descrição clara da funcionalidade

**Por que é Útil?**
Explique o caso de uso e benefícios

**Solução Proposta**
Como você imagina que funcione

**Alternativas Consideradas**
Outras formas de resolver o problema

**Contexto Adicional**
Screenshots, papers, referências
```

### Exemplos de Boas Sugestões

✅ **Específica:** "Adicionar suporte para formato SEG-Y"  
✅ **Justificada:** "Muito usado em sísmica e magnetometria marinha"  
✅ **Realista:** "Pode usar biblioteca `segyio`"

❌ **Vaga:** "Melhorar interface"  
❌ **Sem contexto:** "Adicionar feature X"

---

## 👨‍💻 Contribuindo com Código

### Setup do Ambiente de Desenvolvimento

1. **Fork o repositório** no GitHub

2. **Clone seu fork**
   ```bash
   git clone https://github.com/SEU-USUARIO/GeoBot.git
   cd GeoBot
   ```

3. **Crie ambiente virtual**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

4. **Instale dependências**
   ```bash
   pip install -r requirements.txt
   ```

5. **Instale PyTorch (opcional, para GPU)**
   ```bash
   # NVIDIA CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # Apple Silicon
   pip install torch torchvision
   ```

6. **Crie uma branch**
   ```bash
   git checkout -b feature/minha-funcionalidade
   # ou
   git checkout -b fix/correcao-bug
   ```

7. **Faça suas alterações**

8. **Teste localmente**
   ```bash
   streamlit run geobot.py
   ```

9. **Commit suas mudanças**
   ```bash
   git add .
   git commit -m "feat: adiciona processamento X"
   ```

10. **Push para seu fork**
    ```bash
    git push origin feature/minha-funcionalidade
    ```

11. **Abra um Pull Request** no GitHub

---

## 📝 Padrões de Código

### Estilo Python

Seguimos **PEP 8** com algumas adaptações:

```python
# ✅ BOM
def minha_funcao(parametro: float) -> ProcessingResult:
    """
    Descrição da função.
    
    Parameters:
    -----------
    parametro : float
        Descrição do parâmetro
    
    Returns:
    --------
    ProcessingResult
        Descrição do retorno
    """
    resultado = processar(parametro)
    return resultado

# ❌ EVITE
def MinhaFuncao(p):
    r = processar(p)
    return r
```

### Convenções de Nomenclatura

| Tipo | Convenção | Exemplo |
|------|-----------|---------|
| Variáveis | `snake_case` | `densidade_crosta` |
| Funções | `snake_case` | `calcular_bouguer()` |
| Classes | `PascalCase` | `GeophysicalData` |
| Constantes | `UPPER_SNAKE_CASE` | `GRAVITY_UNITS` |
| Arquivos | `snake_case.py` | `geobot.py` |

### Documentação

Todas as funções públicas devem ter **docstrings** no formato NumPy:

```python
def minha_funcao(param1: float, param2: str = "default") -> dict:
    """
    Breve descrição (uma linha).
    
    Descrição mais detalhada do que a função faz,
    como funciona e quando usar.
    
    Parameters:
    -----------
    param1 : float
        Descrição do primeiro parâmetro
    param2 : str, optional
        Descrição do segundo parâmetro (padrão: "default")
    
    Returns:
    --------
    dict
        Descrição do que é retornado
    
    Raises:
    -------
    ValueError
        Quando param1 é negativo
    
    Examples:
    ---------
    >>> resultado = minha_funcao(10.5, "teste")
    >>> print(resultado)
    {'sucesso': True}
    
    Notes:
    ------
    Notas adicionais sobre implementação ou limitações
    
    References:
    -----------
    AUTOR, A. **Título do Paper**. Journal, v. XX, p. YY, 2020.
    DOI: 10.xxxx/xxxxx
    """
    if param1 < 0:
        raise ValueError("param1 deve ser positivo")
    
    return {"sucesso": True, "param1": param1}
```

### Type Hints

Use type hints para melhor legibilidade:

```python
from typing import List, Dict, Optional, Tuple

def processar_dados(
    data: GeophysicalData,
    params: Dict[str, float],
    verbose: bool = False
) -> Tuple[ProcessingResult, List[go.Figure]]:
    ...
```

### Logging

Use o sistema de logging ao invés de `print()`:

```python
# ✅ BOM
logger.info("Iniciando processamento...")
logger.debug(f"Valores: {values}")
logger.warning("Outliers detectados")
logger.error(f"Erro: {e}")

# ❌ EVITE
print("Iniciando processamento...")
```

### Tratamento de Erros

Sempre use try-except e levante exceções específicas:

```python
# ✅ BOM
try:
    resultado = processar_dados(data)
except InvalidDataError as e:
    logger.error(f"Dados inválidos: {e}")
    raise
except Exception as e:
    logger.error(f"Erro inesperado: {e}")
    raise ProcessingError(f"Falha: {e}")

# ❌ EVITE
try:
    resultado = processar_dados(data)
except:
    pass
```

---

## 🔄 Processo de Pull Request

### Checklist Antes de Enviar

- [ ] Código segue os padrões do projeto
- [ ] Todas as funções têm docstrings
- [ ] Type hints adicionados
- [ ] Testado localmente com sucesso
- [ ] Logs adequados implementados
- [ ] Documentação atualizada (se necessário)
- [ ] Commit messages seguem convenção
- [ ] Sem arquivos temporários ou logs commitados

### Convenção de Commit Messages

Usamos **Conventional Commits**:

```
<tipo>: <descrição curta>

<corpo opcional>

<footer opcional>
```

**Tipos:**
- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Documentação
- `style:` Formatação (sem mudança de lógica)
- `refactor:` Refatoração
- `perf:` Melhoria de performance
- `test:` Testes
- `chore:` Tarefas gerais

**Exemplos:**

```
feat: adiciona suporte para formato SEG-Y

Implementa parser para arquivos SEG-Y usando biblioteca segyio.
Adiciona testes com dados sintéticos.

Closes #123
```

```
fix: corrige cálculo de Bouguer para alta elevação

O fator de correção estava usando raio médio incorreto.
Agora usa raio local baseado em latitude.

Fixes #456
```

### Revisão de Código

Após enviar o PR:

1. ✅ CI/CD rodará automaticamente (quando configurado)
2. 👀 Mantenedores revisarão o código
3. 💬 Podem solicitar mudanças
4. ✅ Após aprovação, será feito merge

**Seja paciente e receptivo ao feedback!** 🙏

---

## 🧪 Testando

### Testes Manuais

Antes de enviar PR, teste:

1. **Carregamento de dados**
   - CSV com vírgula
   - CSV com ponto-e-vírgula
   - Excel
   - Dados de exemplo incluídos

2. **Processamentos**
   - Execute sua nova função
   - Teste com parâmetros diferentes
   - Verifique visualizações geradas

3. **Chat**
   - Comando natural: "Aplique X"
   - Parâmetros extraídos corretamente
   - Resposta adequada do LLM

### Dados de Teste

Use os datasets em `example_data/`:

```python
# No final do arquivo, adicione:
if __name__ == "__main__":
    # Teste rápido
    from pathlib import Path
    
    test_file = Path("example_data/gravity_basin_example.csv")
    data = parse_uploaded_file(test_file.open('rb'), test_file.name)
    
    result = minha_nova_funcao(data, param=10.0)
    
    print(f"✅ Teste OK! Tempo: {result.execution_time:.2f}s")
```

---

## 📦 Adicionando Dependências

Se sua contribuição precisa de novas bibliotecas:

1. Adicione ao `requirements.txt`
2. Justifique no PR
3. Verifique compatibilidade com Python 3.11+

```txt
# requirements.txt
numpy>=1.24.0
scipy>=1.10.0
sua-nova-lib>=1.0.0  # Justifique aqui
```

---

## 🌍 Contribuições Não-Código

### Documentação

- Corrija erros de digitação
- Melhore explicações
- Adicione exemplos
- Traduza para outros idiomas

### Datasets de Exemplo

- Contribua dados sintéticos
- Adicione exemplos reais (com permissão)
- Documente origem e características

### Tutoriais

- Crie vídeos tutoriais
- Escreva blog posts
- Compartilhe em redes sociais

---

## 📄 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a **MIT License** do projeto.

---

## 🙏 Reconhecimento

Todos os contribuidores serão reconhecidos no README e CONTRIBUTORS.md!

---

## ❓ Dúvidas?

- 📧 Email: allansoares@id.uff.br
- 💬 Discussões: [GitHub Discussions](https://github.com/seu-usuario/GeoBot/discussions)
- 📖 Documentação: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

<div align="center">

**Obrigado por contribuir! 🎉**

Made with ❤️ by the GeoBot community

</div>
