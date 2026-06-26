# Pipeline de Curadoria e Análise Estatística — Transtorno Bipolar do Humor (PROMAN)

Este repositório contém o pipeline completo de curadoria, auditoria e análise estatística de dados clínicos para o estudo quantitativo do sofrimento psíquico em pacientes com **Transtorno Bipolar do Humor (TB)**. A base de dados utilizada provém do **Programa de Transtorno Bipolar (PROMAN)** do Instituto de Psiquiatria do Hospital das Clínicas da Faculdade de Medicina da USP (IPq-HC-FMUSP).

O projeto aborda desafios típicos de bases de dados do mundo real (dados ausentes, violações de domínio e inconsistências clínicas) por meio de uma abordagem híbrida de **reconstrução longitudinal lógica baseada em regras psiquiátricas** e **Imputação Múltipla por Equações Encadeadas (MICE)**.

---

## 📂 Estrutura do Repositório

```text
pipeline-dados/
├── graficos/                        # Pasta com os gráficos gerados pelo código
├── dicionario_variaveis.csv         # Dicionário descritivo com tipos e domínios legais
├── tratar_dados.py                  # Script 1: Pipeline de limpeza, lógica lifetime e MICE
├── verificar_integridade.py         # Script 2: Auditoria inicial de inconsistências clínicas
├── analise_de_dados.py              # Script 3: EDA demográfica e testes estatísticos (Chi2, T-Test, Spearman)
├── analise_longitudinal.py          # Script 4: Evolução da linha do tempo e dispersão de episódios
└── print_resultados.py              # Script 5: Consolidador das 5 tabelas oficiais do artigo
```

---

## 🛠️ Instalação e Requisitos

Este pipeline foi desenvolvido em **Python 3**. As dependências de bibliotecas de ciência de dados podem ser instaladas em um ambiente virtual.

### 1. Criar e Ativar Ambiente Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # No macOS/Linux
# .venv\Scripts\activate   # No Windows
```

### 2. Instalar Dependências
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

---

## 🚀 Como Executar o Pipeline

Siga a ordem lógica para executar o pipeline, gerar os relatórios e reproduzir os gráficos:

### Passo 1: Auditoria Inicial de Qualidade
Identifica todas as inconsistências estruturais e lógicas do banco de dados bruto original. Gera os gráficos de auditoria no diretório.
```bash
python verificar_integridade.py
```
* **Gráficos gerados**: `grafico_inconsistencias_top_vars.png` e `grafico_inconsistencias_tipo.png`.

### Passo 2: Tratamento, Curadoria e Imputação (MICE)
Aplica as correções de inversão de variáveis (mania/depressão), resolve a lógica de diagnósticos vitalícios (*lifetime*), executa o MICE e realiza o pós-processamento de arredondamento e limitação de domínio (*clipping*). Salva a base tratada em `dados_projeto/db-longitudinal-tratado.xlsx`.
```bash
python tratar_dados.py
```
* **Gráfico gerado**: `top_10_faltantes.png` (redução de dados ausentes antes vs depois).

### Passo 3: Análise Exploratória e Inferenciais
Mapeia o perfil demográfico, distribuições físicas e executa os testes estatísticos formais (Qui-Quadrado, correlações de Spearman e comparações de média via Teste t de Welch).
```bash
python analise_de_dados.py
```
* **Gráficos gerados**: `dist_diagnostico.png`, `dist_sexo.png`, `dist_etnia.png`, `ocupacao.png`, `est_civil.png`, `dist_idade.png`, `anos_estudo.png`, `dist_imc.png`, `idade_por_diagnostico.png`, `top_qui_quadrado.png` e `top_correlacoes.png`.

### Passo 4: Trajetória Longitudinal
Desenha a linha do tempo média do curso clínico da doença (idade de início, idade da primeira internação e idade atual) e traça a taxa de acúmulo de episódios ao longo do tempo.
```bash
python analise_longitudinal.py
```
* **Gráficos gerados**: `evolucao_longitudinal.png` e `relacao_doenca_episodios.png`.

### Passo 5: Visualizador Final de Tabelas
Executa a consolidação matemática e exibe diretamente no terminal as **5 tabelas oficiais** de resultados inseridas no artigo em LaTeX.
```bash
python print_resultados.py
```
