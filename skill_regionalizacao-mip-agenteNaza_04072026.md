---
name: regionalizacao-mip
description: >
  Use esta skill quando precisar regionalizar a Matriz Insumo-Produto (MIP)
  nacional para um estado ou região brasileira. Ativa quando o usuário pedir:
  construção de MIP regional, cálculo de quocientes locacionais (SLQ, FLQ, CILQ),
  balanceamento RAS ou entropia cruzada, coleta de dados RAIS/IBGE, simulação
  de choques de demanda, decomposição estrutural (SDA), ou validação de matriz
  regional. Entrega uma matriz de coeficientes técnicos regionalizada
  (A_reg), balanceada, validada e documentada.
---
# Skill: Regionalização de MIP — AgenteNazaré

## Propósito
Regionalizar a Matriz Insumo-Produto (MIP) nacional para uma unidade federativa brasileira, executando todo o pipeline: coleta de microdados (RAIS, IBGE), cálculo de quocientes locacionais (SLQ, FLQ, CILQ), aplicação dos quocientes à matriz nacional, balanceamento via RAS (ou entropia cruzada) e validação numérica rigorosa da matriz resultante.

## O que esta skill NÃO faz
- Não cria a MIP nacional do zero — parte da MIP nacional oficial do IBGE
- Não faz modelos CGE completos (apenas simulações de choque via Leontief)
- Não coleta dados que exijam autenticação especial (ex: PNAD microdados restritos)
- Não substitui julgamento econométrico — executa o pipeline com parâmetros documentados

## Fontes
- Referência: `MIP, Regionalização e Coleta de dados.pdf` (AgenteNazaré — Davi Lucena da Silva)
- Internet: documentação oficial IBGE SCN (sistema de contas nacionais), IBGE MIP 2015, RAIS/MTE, PyPI (numpy, pandas, scipy, openpyxl)

---

## 1. Instalação de Dependências

```bash
pip install numpy pandas scipy openpyxl requests matplotlib
```

---

## 2. Pré-requisitos

Antes de executar o pipeline, é necessário:

| Item | Fonte | Ano recomendado |
|---|---|---|
| MIP nacional (matriz A) | IBGE — SCN | 2015 (mais recente disponível) |
| RAIS microdados (vínculos) | MTE/PDET | Mesmo ano da MIP |
| Tabela de correspondência CNAE 2.0 → Setores MIP | IBGE | — |
| Contas Regionais (alvos RAS) | IBGE | Mesmo ano da MIP |

---

## 3. Fluxo Passo a Passo

### 3.1 Coleta e ingestão dos dados

```python
import os, time, hashlib, json, datetime
import requests
import numpy as np
import pandas as pd

def download_seguro(url, destino, max_tentativas=3):
    """
    Baixa arquivo com retry e verificação de integridade.
    
    Parâmetros:
        url (str): URL do arquivo
        destino (str): Caminho local para salvar
        max_tentativas (int): Número máximo de tentativas
    
    Retorna:
        bool: True se bem-sucedido
    
    Levanta:
        Exception se todas as tentativas falharem
    """
    for tentativa in range(max_tentativas):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            tamanho_esperado = int(resp.headers.get('content-length', 0))
            
            with open(destino, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            tamanho_real = os.path.getsize(destino)
            if tamanho_esperado > 0 and tamanho_real != tamanho_esperado:
                raise ValueError(
                    f"Tamanho incompatível: {tamanho_real} != {tamanho_esperado}"
                )
            return True
            
        except Exception as e:
            print(f"Tentativa {tentativa + 1} falhou: {e}")
            if tentativa < max_tentativas - 1:
                time.sleep(2 ** tentativa)
            else:
                raise
```

### 3.2 Agregação RAIS por setor MIP

```python
def agregar_emprego_rais(caminho_rais, mapa_cnae_setor, colunas=None, chunksize=100000):
    """
    Agrega vínculos RAIS por setor MIP, processando em chunks.
    
    Parâmetros:
        caminho_rais (str): Caminho do arquivo RAIS (CSV)
        mapa_cnae_setor (dict): Dicionário CNAE -> setor MIP
        colunas (list): Colunas a carregar (None = todas)
        chunksize (int): Tamanho do chunk para processamento
    
    Retorna:
        pd.Series: Emprego total por setor MIP
    """
    if colunas is None:
        colunas = ['cnae', 'vinculos']
    
    chunks = pd.read_csv(
        caminho_rais,
        chunksize=chunksize,
        usecols=colunas,
        dtype={'cnae': str}
    )
    resultados = []
    for chunk in chunks:
        chunk['setor_mip'] = chunk['cnae'].str[:5].map(mapa_cnae_setor)
        chunk = chunk.dropna(subset=['setor_mip'])
        resultados.append(chunk.groupby('setor_mip')['vinculos'].sum())
    
    return pd.concat(resultados).groupby(level=0).sum()
```

### 3.3 Cálculo dos Quocientes Locacionais

#### 3.3.1 Simple Location Quotient (SLQ)

```python
def calcular_SLQ(E_ir, E_r, E_in, E_n):
    """
    Calcula o Simple Location Quotient.
    
    SLQ_i = (E_ir / E_r) / (E_in / E_n)
    
    Parâmetros:
        E_ir (array): Emprego por setor na região
        E_r (float): Emprego total na região
        E_in (array): Emprego por setor no país
        E_n (float): Emprego total no país
    
    Retorna:
        array: SLQ para cada setor
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        slq = (E_ir / E_r) / (E_in / E_n)
    return np.where(np.isfinite(slq), slq, 0.0)
```

#### 3.3.2 Flegg's Location Quotient (FLQ)

```python
def calcular_FLQ(E_ir, E_r, E_in, E_n, delta=0.3):
    """
    Calcula o Flegg's Location Quotient (FLQ).
    
    FLQ_i = SLQ_i × [log2(1 + E_r / E_n)]^δ
    
    Parâmetros:
        E_ir (array): Emprego por setor na região
        E_r (float): Emprego total na região
        E_in (array): Emprego por setor no país
        E_n (float): Emprego total no país
        delta (float): Parâmetro de Flegg (default 0.3, intervalo comum 0.2-0.4)
    
    Retorna:
        array: FLQ para cada setor
    """
    slq = calcular_SLQ(E_ir, E_r, E_in, E_n)
    fator = np.log2(1 + E_r / E_n) ** delta
    flq = slq * fator
    # FLQ deve ser sempre positivo e finito
    return np.where(np.isfinite(flq) & (flq >= 0), flq, 0.0)
```

#### 3.3.3 Cross-Industry Location Quotient (CILQ)

```python
def calcular_CILQ(SLQ_i, SLQ_j):
    """
    Calcula o Cross-Industry Location Quotient (CILQ).
    
    CILQ_ij = SLQ_i / SLQ_j
    
    Parâmetros:
        SLQ_i (array): SLQ do setor fornecedor i
        SLQ_j (array): SLQ do setor comprador j
    
    Retorna:
        array 2D: CILQ_ij para cada par (i, j)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        cilq = SLQ_i[:, np.newaxis] / SLQ_j[np.newaxis, :]
    return np.where(np.isfinite(cilq), cilq, 1.0)
```

### 3.4 Regionalização da Matriz (FLQ)

```python
def regionalizar_matriz(A_nac, FLQ):
    """
    Aplica os FLQs à matriz nacional para obter a matriz regional.
    
    Regra:
        - Se FLQ_i >= 1: setor i é autossuficiente → mantém coeficiente nacional
        - Se FLQ_i < 1: setor i é importador → A_reg[i,:] = A_nac[i,:] × FLQ_i
    
    Parâmetros:
        A_nac (ndarray): Matriz de coeficientes técnicos nacional (n x n)
        FLQ (array): FLQ para cada setor (n,)
    
    Retorna:
        ndarray: Matriz de coeficientes técnicos regionalizada
    """
    A_reg = A_nac.copy()
    for i in range(A_nac.shape[0]):
        if FLQ[i] < 1:
            A_reg[i, :] = A_nac[i, :] * FLQ[i]
        # Se FLQ[i] >= 1, mantém coeficiente nacional
    return A_reg
```

### 3.5 Balanceamento — Método RAS

```python
def ras(A_inicial, u_alvo, v_alvo, max_iter=1000, tol=1e-6):
    """
    Aplica o método biproporcional RAS para balancear a matriz A.
    
    O RAS ajusta a matriz inicial para satisfazer totais de linha e coluna
    conhecidos, preservando ao máximo a estrutura relativa original.
    
    Parâmetros:
        A_inicial (ndarray): Matriz inicial (n x n) — ex: A_reg pós-FLQ
        u_alvo (array): Vetor de totais de linha alvo (n,)
        v_alvo (array): Vetor de totais de coluna alvo (n,)
        max_iter (int): Número máximo de iterações
        tol (float): Tolerância para convergência
    
    Retorna:
        tuple: (A_balanceada, num_iteracoes, max_desvio)
    """
    A = A_inicial.copy().astype(float)
    
    for k in range(max_iter):
        # Fatores de linha
        row_sums = A.sum(axis=1)
        r = np.where(row_sums > 0, u_alvo / row_sums, 0)
        A = A * r[:, np.newaxis]
        
        # Fatores de coluna
        col_sums = A.sum(axis=0)
        s = np.where(col_sums > 0, v_alvo / col_sums, 0)
        A = A * s[np.newaxis, :]
        
        # Verificar convergência
        max_dev = max(np.max(np.abs(r - 1)), np.max(np.abs(s - 1)))
        if max_dev < tol:
            return A, k + 1, max_dev
    
    return A, max_iter, max_dev
```

#### 3.5.1 Construção dos alvos para o RAS

```python
def construir_alvos_regionais(A_nac, X_nac, E_r, E_n):
    """
    Constrói os alvos de linha (u) e coluna (v) para uma região.
    
    1. Calcula matriz de consumo intermediário: Z = A × diag(X)
    2. Calcula totais de linha e coluna nacionais
    3. Escala para a região proporcionalmente ao emprego
    
    Parâmetros:
        A_nac (ndarray): Matriz de coeficientes nacional
        X_nac (array): Vetor de produção nacional
        E_r (float): Emprego total na região
        E_n (float): Emprego total no país
    
    Retorna:
        tuple: (u_reg, v_reg) — alvos de linha e coluna para a região
    """
    # Matriz de consumo intermediário nacional
    Z_nac = A_nac * X_nac  # broadcasting: A_nac @ diag(X_nac)
    
    # Totais nacionais
    u_nac = Z_nac.sum(axis=1)  # totais de linha
    v_nac = Z_nac.sum(axis=0)  # totais de coluna
    
    # Escalar para a região
    proporcao = E_r / E_n
    u_reg = u_nac * proporcao
    v_reg = v_nac * proporcao
    
    return u_reg, v_reg
```

### 3.6 Validação Final

```python
def validar_matriz_regional(A_reg, nome="Matriz Regional"):
    """
    Executa todas as validações obrigatórias na matriz regional.
    
    Critérios:
        - Dimensão n x n
        - Todos os coeficientes >= 0
        - Soma de cada coluna < 1
        - det(I - A) > 1e-6 (não singular)
        - Multiplicadores de produção >= 1
    
    Parâmetros:
        A_reg (ndarray): Matriz de coeficientes técnicos regional
    
    Retorna:
        dict: Resultado de cada validação (passed: bool, value: float)
    """
    n = A_reg.shape[0]
    I = np.eye(n)
    resultados = {}
    
    # 1. Dimensão
    resultados['dimensao'] = {
        'passed': A_reg.shape[0] == A_reg.shape[1],
        'value': f"{A_reg.shape[0]}x{A_reg.shape[1]}"
    }
    
    # 2. Coeficientes não negativos
    resultados['nao_negativo'] = {
        'passed': np.all(A_reg >= 0),
        'value': f"min={A_reg.min():.6f}"
    }
    
    # 3. Soma das colunas < 1
    col_sums = A_reg.sum(axis=0)
    resultados['soma_colunas'] = {
        'passed': np.all(col_sums < 1),
        'value': f"max_col_sum={col_sums.max():.6f}"
    }
    
    # 4. Determinante de (I - A) != 0
    M = I - A_reg
    det = np.linalg.det(M)
    resultados['determinante'] = {
        'passed': abs(det) > 1e-6,
        'value': f"det={det:.6e}"
    }
    
    # 5. Número de condição
    cond = np.linalg.cond(M)
    resultados['condicionamento'] = {
        'passed': cond < 1e10,
        'value': f"cond={cond:.2f}"
    }
    
    # 6. Multiplicadores de produção >= 1
    try:
        L = np.linalg.solve(M, I)
        multiplicadores = L.sum(axis=0)
        resultados['multiplicadores'] = {
            'passed': bool(np.all(multiplicadores >= 1)),
            'value': f"min_mult={multiplicadores.min():.4f}"
        }
    except np.linalg.LinAlgError as e:
        resultados['multiplicadores'] = {
            'passed': False,
            'value': f"Erro: {e}"
        }
    
    return resultados
```

### 3.7 Simulação de Choque de Demanda

```python
def simular_choque(A_reg, Y0, delta_Y):
    """
    Simula o impacto de um choque na demanda final.
    
    ΔX = L × ΔY
    Onde L = (I - A)^{-1}
    
    Parâmetros:
        A_reg (ndarray): Matriz de coeficientes regional
        Y0 (array): Vetor de demanda final inicial
        delta_Y (array): Vetor de choque na demanda final
    
    Retorna:
        dict: Efeitos direto, indireto e total na produção
    """
    n = A_reg.shape[0]
    I = np.eye(n)
    L = np.linalg.solve(I - A_reg, I)
    
    Y1 = Y0 + delta_Y
    X0 = L @ Y0
    X1 = L @ Y1
    delta_X = X1 - X0
    
    return {
        'efeito_direto': delta_Y,
        'efeito_indireto': delta_X - delta_Y,
        'efeito_total': delta_X,
        'X0': X0,
        'X1': X1
    }
```

### 3.8 Log de Execução

```python
def gerar_log(parametros, downloads, validacoes):
    """
    Gera um registro de execução documentando parâmetros, downloads e validações.
    
    Parâmetros:
        parametros (dict): Parâmetros usados (delta, ras_tol, etc.)
        downloads (list): Lista de arquivos baixados com tamanho e hash
        validacoes (list): Lista de testes de validação com resultados
    
    Retorna:
        dict: Log completo da execução
    """
    log = {
        "execution_date": datetime.datetime.now().isoformat(),
        "parameters": parametros,
        "downloads": downloads,
        "validations": validacoes
    }
    return log
```

---

## 4. Tabelas de Referência

### 4.1 Métodos de Regionalização

| Método | Fórmula | Quando usar |
|---|---|---|
| SLQ | (E_ir/E_r) / (E_in/E_n) | Regiões grandes, visão preliminar |
| FLQ | SLQ_i × [log₂(1+E_r/E_n)]^δ | Regiões pequenas/médias (corrige viés do SLQ) |
| CILQ | SLQ_i / SLQ_j | Análise de pares setoriais (fornecedor-comprador) |
| FLQ + CILQ | min(CILQ_ij, FLQ_i) | Abordagem combinada recomendada |
| RAS | Iterativo (linha → coluna) | Balancear para totais conhecidos |
| CE | Min ∑ a_ij log(a_ij/ā_ij) | Quando RAS não converge ou há restrições adicionais |

### 4.2 Critérios de Validação Obrigatórios

| Componente | Critério | Tolerância | Ação se falhar |
|---|---|---|---|
| Download | Nº linhas vs. documentação | ±1% | Parar e investigar |
| Matriz A | 0 ≤ a_ij < 1 | — | Corrigir ou descartar |
| Soma colunas A | ∑_i a_ij < 1 | < 1.0 | Revisar dados/estimação |
| det(I-A) | \|det\| > 0 | > 1e-6 | Não inverter |
| Multiplicadores | ≥ 1 | ≥ 1.0 | Revisar matriz |
| RAS | Desvio alvos | < 0.5% | Revisar parâmetros |
| FLQ | FLQ_i > 0 | > 0 | Investigar divisão por zero |
| Consistência fontes | RAIS vs. CAGED | ±2% | Documentar divergência |

### 4.3 Parâmetro δ de Flegg

| Contexto | δ sugerido | Fonte |
|---|---|---|
| Padrão (região pequena/média) | 0.30 | Flegg & Webber (2000) |
| Região muito pequena (E_r/E_n < 0.02) | 0.35–0.40 | Literatura |
| Região grande (E_r/E_n > 0.20) | 0.20–0.25 | Literatura |
| **Sempre documentar a escolha** | — | — |

---

## 5. Regras

### O que SEMPRE fazer
- Verificar integridade do download (tamanho vs. servidor) antes de processar
- Converter CNAE para string de comprimento fixo antes do mapeamento
- Usar `np.errstate(divide='ignore', invalid='ignore')` ao calcular quocientes
- Verificar FLQ positivo e finito antes de aplicar à matriz
- Executar as 6 validações da seção 3.6 após o balanceamento
- Documentar a escolha do parâmetro δ do FLQ e a tolerância do RAS
- Salvar log de execução com data, parâmetros, validações

### O que NUNCA fazer
- Nunca modificar o dado bruto — sempre fazer uma cópia antes de processar
- Nunca criar resíduos artificiais para forçar o fechamento da matriz
- Nunca forçar a inversa de (I-A) se o determinante for próximo de zero
- Nunca escolher uma fonte de dados arbitrariamente sem documentar
- Nunca usar tolerância RAS acima de 0.5%
- Nunca executar validação "olhando para o número" — sempre medir e documentar

### Quando algo falhar
- **Download incompleto**: tentar novamente com `download_seguro`; se persistir, reportar ao administrador da fonte
- **RAS não converge**: (1) verificar ∑u_i == ∑v_j; (2) verificar alvos negativos; (3) aumentar tolerância (max 0.5%); (4) tentar entropia cruzada (CE); (5) reportar não convergência — não forçar com resíduos artificiais
- **Matriz (I-A) singular**: (1) verificar colunas que somam próximo de 1; (2) verificar linhas/colunas de zeros; (3) ajustar coeficientes suspeitos e rebalancear; (4) reportar a limitação
- **Inconsistência RAIS vs CAGED**: se < 2%, usar a fonte mais apropriada; se > 2%, documentar a divergência explicitamente e explicar a escolha
- **Dados muito grandes (RAIS completo)**: processar em chunks (chunksize=100000), filtrar por UF antes, carregar apenas as colunas necessárias (`usecols` no pandas)

---

## 6. Pipeline Completo (Árvore de Decisão)

```
1. COLETA
   ├── Arquivo já existe? → Sim: verificar integridade
   │                       └─ Não: baixar (download_seguro)
   └── Download completo? → Sim: verificar tamanho
                            └─ Não: tentar novamente

2. LEITURA
   ├── Formato legível? → Sim: carregar
   │                      └─ Não: instalar biblioteca adequada
   └── Nº linhas confere? → Sim: seguir
                            └─ Não: verificar documentação

3. PROCESSAMENTO
   ├── Valores negativos onde não deveriam? → Investigar
   └── Agregações fecham com totais conhecidos? → Verificar filtros

4. FLQ
   ├── Setor com emprego zero? → FLQ = 0
   └── FLQ é NaN ou infinito? → Investigar divisão por zero

5. RAS
   ├── ∑u_i == ∑v_j? → Sim: rodar RAS
   │                   └─ Não: ajustar alvos
   └── RAS convergiu? → Sim: validar
                        └─ Não: tentar entropia cruzada (CE)

6. VALIDAÇÃO FINAL
   └── Todos os critérios passaram? → Sim: exportar e documentar
                                      └─ Não: identificar e corrigir
```

---

## 7. Exemplo de Uso Completo

```python
# ============================================================
# Exemplo: Regionalização da MIP para Minas Gerais (MG)
# ============================================================

import numpy as np
import pandas as pd

# --- Dados simulados para demonstração ---
# Uma MIP real viria dos arquivos do IBGE e RAIS

n_setores = 12  # Exemplo com 12 setores

# Matriz nacional simulada (coeficientes técnicos)
np.random.seed(42)
A_nac = np.random.dirichlet(np.ones(n_setores), size=n_setores) * 0.3

# Produção nacional
X_nac = np.random.uniform(100, 1000, size=n_setores)

# Emprego
E_n = 100_000_000  # emprego total Brasil
E_r = 10_000_000   # emprego total MG (~10%)
E_in = np.random.uniform(1_000, 20_000_000, size=n_setores)  # emprego por setor BR
E_ir = np.random.uniform(100, 2_000_000, size=n_setores)     # emprego por setor MG

# Garantir proporcionalidade grosseira para o exemplo
E_in = E_in / E_in.sum() * E_n
E_ir = E_ir / E_ir.sum() * E_r

# --- 1. Calcular FLQ ---
flq = calcular_FLQ(E_ir, E_r, E_in, E_n, delta=0.3)
print("FLQ por setor:", np.round(flq, 4))

# --- 2. Regionalizar matriz ---
A_reg = regionalizar_matriz(A_nac, flq)

# --- 3. Construir alvos RAS ---
u_reg, v_reg = construir_alvos_regionais(A_nac, X_nac, E_r, E_n)

# --- 4. Balancear com RAS ---
A_balanced, num_iter, max_dev = ras(A_reg, u_reg, v_reg)

print(f"RAS convergiu em {num_iter} iterações, desvio máximo: {max_dev:.6e}")

# --- 5. Validar ---
validacoes = validar_matriz_regional(A_balanced)
for teste, resultado in validacoes.items():
    status = "✅" if resultado['passed'] else "❌"
    print(f"{status} {teste}: {resultado['value']}")

# --- 6. Log ---
log = gerar_log(
    parametros={"delta": 0.3, "ras_tol": 1e-6, "ras_max_iter": 1000},
    downloads=[{"file": "MIP_2015.xls", "size": 132608, "hash": "abc123..."}],
    validacoes=[
        {"teste": k, "value": v['value'], "passed": v['passed']}
        for k, v in validacoes.items()
    ]
)
with open("log_execucao.json", "w") as f:
    json.dump(log, f, indent=2)
```

---

## 8. Glossário

| Termo | Definição |
|---|---|
| **MIP** | Matriz Insumo-Produto |
| **TRU** | Tabela de Recursos e Usos |
| **SCN** | Sistema de Contas Nacionais |
| **A** | Matriz de coeficientes técnicos (n x n) |
| **L** | Inversa de Leontief (I - A)⁻¹ |
| **FLQ** | Flegg's Location Quotient |
| **SLQ** | Simple Location Quotient |
| **CILQ** | Cross-Industry Location Quotient |
| **RAS** | Método biproporcional de balanceamento |
| **CE** | Entropia Cruzada (Cross-Entropy) |
| **VA** | Valor Adicionado |
| **DF** | Demanda Final |
| **CNAE** | Classificação Nacional de Atividades Econômicas |
| **RAIS** | Relação Anual de Informações Sociais |
| **CAGED** | Cadastro Geral de Empregados e Desempregados |
