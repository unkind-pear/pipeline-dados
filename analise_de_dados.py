# -*- coding: utf-8 -*-
"""
Análise Exploratória de Dados (EDA) e Testes Estatísticos Inferenciais
Transtorno Bipolar do Humor (PROMAN - IPq-HC-FMUSP)

Este script lê o banco de dados tratado pós-curadoria e executa as seguintes etapas:
1. Plotagem de perfil geral demográfico (sexo, etnia, ocupação, estado civil).
2. Plotagem e análise de variáveis contínuas cruciais (idade, escolaridade, IMC).
3. Testes estatísticos inferenciais:
   - Qui-Quadrado para associação categórica.
   - Correlações de Spearman para variáveis numéricas (evitando famílias lifetime).
   - Teste t de Welch para comparação de médias (subtipos e tentativa de suicídio).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, ttest_ind
import warnings

# Silencia avisos estatísticos repetitivos inofensivos
warnings.filterwarnings('ignore')

# Configuração global de diretórios e plotagem
PATH_DADOS = "./dados_projeto/db-longitudinal-tratado.xlsx"
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def save_plot(filename):
    """Auxiliar para salvar gráficos uniformemente mantendo layout ajustado."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pie(df, col, map_dict, title, filename, palette="pastel"):
    """Gera um gráfico de pizza para dados demográficos qualitativos nominais."""
    if col in df.columns:
        series = df[col].map(map_dict).fillna("Não Informado").value_counts()
        total = series.sum()
        
        print(f"\n[DEMOGRÁFICOS] Contagem para '{title}' ({col}):")
        for idx, val in series.items():
            print(f"  - {idx}: {val} pacientes ({(val / total) * 100:.1f}%)")
            
        plt.figure(figsize=(8, 8))
        autopct = lambda p: f'{p:.0f}%' if p >= 2 else ''
        
        plt.pie(
            series.values, 
            labels=series.index, 
            autopct=autopct, 
            colors=sns.color_palette(palette),
            textprops={'fontsize': 14},
            labeldistance=1.1,
            pctdistance=0.75
        )
        plt.title(title, fontsize=18, pad=20, fontweight='bold')
        save_plot(filename)

def plot_bar(df, col, map_dict, title, filename, palette="viridis"):
    """Gera um gráfico de barras horizontais para distribuições demográficas complexas."""
    if col in df.columns:
        series = df[col].map(map_dict).fillna("Não Informado").value_counts()
        total = series.sum()
        
        print(f"\n[DEMOGRÁFICOS] Contagem para '{title}' ({col}):")
        for idx, val in series.items():
            print(f"  - {idx}: {val} pacientes ({(val / total) * 100:.1f}%)")
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=series.values, y=series.index, hue=series.index, legend=False, palette=palette)
        plt.xlabel("Número absoluto de pacientes")
        plt.ylabel("")
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        save_plot(filename)

def plot_dist(df, col, title, filename, color="skyblue"):
    """Exibe histograma com densidade (KDE) e boxplot sobrepostos para análise contínua."""
    if col in df.columns:
        data = df[col].dropna()
        
        print(f"\n[DISTRIBUIÇÃO] Estatísticas Descritivas para '{title}' ({col}):")
        print(f"  - Casos Válidos: {len(data)} | Média: {data.mean():.2f} | Mediana: {data.median():.2f}")
        print(f"  - Intervalo: [{data.min():.2f} a {data.max():.2f}] | Desvio Padrão: {data.std():.2f}")
        
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 7))
        sns.boxplot(x=data, ax=ax_box, color=color)
        sns.histplot(x=data, ax=ax_hist, bins=20, kde=True, color=color)
        
        ax_box.set_xlabel('')
        ax_box.tick_params(axis='both', labelsize=12)
        ax_hist.set_xlabel(ax_hist.get_xlabel(), fontsize=14)
        ax_hist.set_ylabel(ax_hist.get_ylabel(), fontsize=14)
        ax_hist.tick_params(axis='both', labelsize=12)
        
        ax_box.set_title(title, fontsize=16, pad=15, fontweight='bold')
        save_plot(filename)

def plot_grouped_boxplot(df, col_num, col_cat, map_dict_cat, title, filename):
    """Cria boxplot comparativo agrupado por categoria e imprime estatísticas por grupo."""
    if col_num in df.columns and col_cat in df.columns:
        df_temp = df[[col_num, col_cat]].dropna()
        df_temp[col_cat] = df_temp[col_cat].map(map_dict_cat).fillna("Outro")
        
        print(f"\n[AGRUPADO] Estatísticas de '{col_num}' por '{col_cat}':")
        stats = df_temp.groupby(col_cat)[col_num].describe()
        for idx, row in stats.iterrows():
            print(f"  - Grupo {idx}: N={row['count']:.0f} | Média={row['mean']:.2f} | Mediana={row['50%']:.2f}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col_cat, y=col_num, data=df_temp, hue=col_cat, legend=False, palette="Set1")
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.xlabel(col_cat, fontsize=13)
        plt.ylabel(col_num, fontsize=13)
        save_plot(filename)

# -----------------------------------------------------------------------------
# TRATAMENTO DE VARIÁVEIS RELACIONADAS (EVITAR AUTO-CORRELAÇÃO)
# -----------------------------------------------------------------------------
FAMILIAS_LIFETIME = [
    {"teptlifetime", "tept_at", "tept_pas"},
    {"panicosemgorafobialifetime", "pan_saga", "pan_sagp"},
    {"anorexialifetime", "anorex_a", "anorexia_p"},
    {"panicagorafobialifetime", "pan_agoa", "pan_agop"},
    {"abusoalcoollifetime", "abus_a", "abus_p"},
    {"depalcoollifetime", "dep_a", "dependenciaalcoolpassado"},
    {"toclifetime", "toc_at", "toc_pas"},
    {"agorafobialifetime", "agoraf_a", "agoraf_p"},
    {"fobiasocialalifetime", "fob_soca", "fob_socp"},
    {"taglifetime", "tag_at", "tag_pas"},
    {"tanssoelifetime", "ta_soea", "ta_soep"},
    {"bulimialifetime", "bulim_at", "bulim_ps"},
    {"compulsalimlifetime", "compulsaoalimatual", "tcap_pas"},
    {"abusosubstllifetime", "abusn_a", "abusn_p"},
    {"depsubstllifetime", "depen_a", "depen_p"},
    {"fobiaespecificaifetime", "fob_espa", "fob_espp"}
]

def sao_da_mesma_familia(v1, v2):
    """Determina se duas variáveis tratam da mesma patologia clínica (atual vs lifetime)."""
    for familia in FAMILIAS_LIFETIME:
        if v1 in familia and v2 in familia:
            return True
    return False

# -----------------------------------------------------------------------------
# QUI-QUADRADO
# -----------------------------------------------------------------------------
def run_all_chi_square(df, p_limite=0.05):
    """Roda testes Qui-Quadrado para todas as combinações de variáveis qualitativas."""
    print("\n" + "." * 70)
    print(" MOTOR ESTATÍSTICO: QUI-QUADRADO DE INDEPENDÊNCIA ".center(70, "."))
    print("." * 70)

    dic = pd.read_csv("./dados_projeto/dicionario_variaveis.csv")
    cat_cols_dic = dic[dic['Tipo de Variável'].str.contains('qualitativa', na=False, case=False)]['Variável'].tolist()
    
    cols_cat = []
    for col in cat_cols_dic:
        if col in df.columns:
            # Ignoramos variáveis excluídas por redundância ou instruções
            if col in ['enxaq', 'enxaqueca12', 'hipo12', 'hipotir']:
                continue
            df.loc[:, col] = df[col].round()
            counts = df[col].value_counts(dropna=True)
            # Exige variabilidade mínima e frequência para validade de Qui-Quadrado
            if len(counts) > 1 and counts.min() > 5:
                cols_cat.append(col)

    resultados = []

    for i in range(len(cols_cat)):
        for j in range(i + 1, len(cols_cat)):
            c1, c2 = cols_cat[i], cols_cat[j]
            if sao_da_mesma_familia(c1, c2) or {c1, c2} == {'humor', 'est_hum_atual'}:
                continue

            ct = pd.crosstab(df[c1], df[c2])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue

            chi2, p, dof, esperadas = chi2_contingency(ct)

            # Restrições formais: 80% das células com frequência esperada >= 5 e nenhuma < 1
            if (esperadas >= 5).sum() / esperadas.size < 0.8 or (esperadas < 1).any():
                continue

            if p < p_limite:
                resultados.append((f"{c1} x {c2}", chi2, p))

    # Exibe e plota os 15 maiores Qui-Quadrados significativos
    resultados_sorted = sorted(resultados, key=lambda x: x[1], reverse=True)[:15]
    print(f"\n[QUI-QUADRADO] Associações mais significativas encontradas (p < {p_limite}):")
    for r in resultados_sorted:
        print(f"  - {r[0]}: chi2={r[1]:.2f}, p={r[2]:.2e}")

    if resultados_sorted:
        df_chi = pd.DataFrame(resultados_sorted, columns=['Relação', 'Chi2', 'P_valor'])
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='Chi2', y='Relação', data=df_chi, palette='magma', hue='Relação', legend=False)
        plt.title("Top 15 Associações Categóricas - Qui-Quadrado", fontsize=13, fontweight='bold')
        plt.xlabel("Estatística Qui-Quadrado (\u03C7\u00B2)")
        plt.ylabel("Pares de Variáveis")
        
        for idx, p in enumerate(ax.patches):
            val_p = df_chi.iloc[idx]['P_valor']
            ax.annotate(f"p={val_p:.1e} | \u03C7\u00B2={p.get_width():.1f}", 
                        (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=9, fontweight='bold')
            
        plt.xlim(0, max(df_chi['Chi2']) + 12)
        save_plot("./top_qui_quadrado.png")
        print("[OK] Gráfico 'top_qui_quadrado.png' exportado.")

# -----------------------------------------------------------------------------
# SPEARMAN CORRELATION
# -----------------------------------------------------------------------------
def plot_significant_correlations_bar(df, p_limite=0.05, corr_minima=0.3):
    """Calcula e plota as correlações monotônicas de Spearman entre variáveis numéricas."""
    print("\n" + "." * 70)
    print(" MOTOR ESTATÍSTICO: CORRELAÇÕES DE SPEARMAN ".center(70, "."))
    print("." * 70)
    
    dic = pd.read_csv("./dados_projeto/dicionario_variaveis.csv")
    quant_cols = dic[dic['Tipo de Variável'].str.contains('quantitativa', na=False, case=False)]['Variável'].tolist()
    valid_cols = [c for c in quant_cols if c in df.columns and df[c].nunique() > 1]
    
    df_num = df[valid_cols]
    cols = df_num.columns
    resultados = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if sao_da_mesma_familia(c1, c2):
                continue
            
            mask = df_num[[c1, c2]].notnull().all(axis=1)
            x, y = df_num.loc[mask, c1], df_num.loc[mask, c2]
            if len(x) > 5:
                rho, p = spearmanr(x, y)
                if p < p_limite and abs(rho) >= corr_minima:
                    resultados.append((f"{c1} x {c2}", rho, p, abs(rho)))

    if not resultados:
        print("\n[INFO] Nenhuma correlação estatisticamente significante encontrada.")
        return

    top_res = sorted(resultados, key=lambda x: x[3], reverse=True)[:15]
    print(f"\n[SPEARMAN] Correlações mais fortes e significativas (|rho| >= {corr_minima}):")
    for r in top_res:
        print(f"  - {r[0]}: rho={r[1]:.3f}, p={r[2]:.2e}")

    df_rho = pd.DataFrame(top_res, columns=['Relação', 'Rho', 'P_valor', 'AbsRho'])
    plt.figure(figsize=(10, 8))
    # Cores personalizadas: azul para correlações positivas, vermelho para negativas
    colors_mapped = ['#e74c3c' if r < 0 else '#3498db' for r in df_rho['Rho']]
    
    ax = sns.barplot(x='Rho', y='Relação', data=df_rho, palette=colors_mapped, hue='Relação', legend=False)
    plt.title("Top 15 Correlações de Spearman (Dados Tratados)", fontsize=13, fontweight='bold')
    plt.xlabel("Coeficiente de Correlação (\u03C1)")
    plt.ylabel("Pares de Variáveis")
    plt.axvline(0, color='black', linewidth=1)
    
    for idx, p in enumerate(ax.patches):
        rho_v = df_rho.iloc[idx]['Rho']
        p_v = df_rho.iloc[idx]['P_valor']
        if rho_v > 0:
            ax.annotate(f" \u03C1={rho_v:.2f} (p={p_v:.1e})", 
                        (p.get_width() + 0.02, p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax.annotate(f" \u03C1={rho_v:.2f} (p={p_v:.1e})", 
                        (p.get_width() - 0.02, p.get_y() + p.get_height() / 2),
                        ha='right', va='center', fontsize=9, fontweight='bold')
            
    plt.xlim(-1.1, 1.1)
    save_plot("./top_correlacoes.png")
    print("[OK] Gráfico 'top_correlacoes.png' exportado.")

# -----------------------------------------------------------------------------
# TESTE T DE WELCH
# -----------------------------------------------------------------------------
def run_t_tests(df, group_col='diagnostico', p_limite=0.05):
    """Compara médias de variáveis contínuas entre dois subgrupos via T-Test de Welch."""
    print(f"\n[!] Executando comparações de médias agrupando por '{group_col}'...")
    
    dic = pd.read_csv("./dados_projeto/dicionario_variaveis.csv")
    quant_cols = dic[dic['Tipo de Variável'].str.contains('quantitativa', na=False, case=False)]['Variável'].tolist()
    quant_cols = [c for c in quant_cols if c in df.columns and df[c].nunique() > 2]
    
    if group_col not in df.columns:
        return
        
    df_temp = df.copy()
    df_temp[group_col] = df_temp[group_col].dropna().round()
    
    if df_temp[group_col].nunique() != 2:
        return
        
    groups = df_temp[group_col].dropna().unique()
    g1, g2 = groups[0], groups[1]
    resultados = []

    for col in quant_cols:
        data1 = df_temp[df_temp[group_col] == g1][col].dropna()
        data2 = df_temp[df_temp[group_col] == g2][col].dropna()
        if len(data1) > 5 and len(data2) > 5:
            # Roda Welch (equal_var=False) pois tamanhos de grupos e variâncias divergem frequentemente
            t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
            if p_val < p_limite:
                resultados.append((col, t_stat, p_val, data1.mean(), data2.mean(), g1, g2))
                
    if resultados:
        print(f"  [RESULTADOS SIGNIFICATIVOS] (p < {p_limite}):")
        for res in sorted(resultados, key=lambda x: x[2]):
            col, t, p, m1, m2, grp1, grp2 = res
            print(f"    - {col:<25} | Média G{grp1}: {m1:<8.2f} | Média G{grp2}: {m2:<8.2f} | p = {p:.1e}")
    else:
        print("  - Nenhuma média de variável numérica apresentou variação significante entre os grupos.")

# -----------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(PATH_DADOS):
        print(f"[ERRO] Arquivo de dados tratados não encontrado em: {PATH_DADOS}")
        print("Por favor, execute o script 'tratar_dados.py' primeiro para gerar o banco tratado.")
        return

    df = pd.read_excel(PATH_DADOS)
    print("=" * 70)
    print(" ETAPA 2: ANÁLISE EXPLORATÓRIA E ESTATÍSTICA INFERENCIAL ".center(70, "="))
    print("=" * 70)

    # 1. Gráficos Demográficos
    print("\n[+] Renderizando perfis demográficos da amostra...")
    diag_dict = {1: "Tipo I", 2: "Tipo II"}
    plot_pie(df, "diagnostico", diag_dict, "Distribuição Diagnóstica (TB-I vs TB-II)", "dist_diagnostico.png")
    plot_pie(df, "sexo", {1: "Masculino", 2: "Feminino"}, "Distribuição por Sexo", "dist_sexo.png", "Set2")
    plot_pie(df, "caucasiano", {1: "Branco", 2: "Negro", 3: "Pardo", 4: "Não Branco"}, "Distribuição Étnica Auto-declarada", "dist_etnia.png", "YlOrBr")
    
    ocup_dict = {1: "Estudante", 2: "Remunerada Ativa", 3: "Sem Ocupação", 4: "Dona de Casa", 
                 5: "Auxílio Doença", 6: "Aposentado Invalidez", 7: "Aposentado Tempo Serviço"}
    plot_bar(df, "ocup", ocup_dict, "Ocupação Laboral dos Pacientes", "ocupacao.png", "viridis")
    
    est_civil_dict = {1: "Solteiro", 2: "Casado", 3: "Separado", 4: "Viúvo"}
    plot_bar(df, "est_civil", est_civil_dict, "Estado Civil da Amostra", "est_civil.png", "magma")

    # 2. Distribuições Físicas e Idade
    print("\n[+] Renderizando distribuições de idades e IMC...")
    plot_dist(df, "idade", "Distribuição de Idade (Anos)", "dist_idade.png", "skyblue")
    plot_dist(df, "an_compl", "Anos Completos de Estudo (Escolaridade)", "anos_estudo.png", "lightgreen")
    plot_dist(df, "imc", "Distribuição de Índice de Massa Corporal (IMC)", "dist_imc.png", "lightpink")

    # 3. Cruzamento Gráfico: Idade por Tipo Diagnóstico
    plot_grouped_boxplot(df, "idade", "diagnostico", diag_dict, "Distribuição de Idade por Subtipo Diagnóstico", "idade_por_diagnostico.png")

    # 4. Processamento dos Motores Estatísticos
    run_all_chi_square(df)
    plot_significant_correlations_bar(df)
    
    # 5. Comparações Comparativas de Grupo (Média)
    print("\n" + "." * 70)
    print(" MOTOR ESTATÍSTICO: COMPARAÇÕES DE MÉDIAS (WELCH T-TEST) ".center(70, "."))
    print("." * 70)
    run_t_tests(df, group_col='diagnostico')  # TB-I vs TB-II
    run_t_tests(df, group_col='hospit')       # Internou vs Não Internou
    run_t_tests(df, group_col='tent_sui')     # Histórico de suicídio vs Sem Histórico

    print("\n" + "=" * 70)
    print(" ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO! ".center(70, "="))
    print(" Os gráficos estatísticos foram salvos no diretório raiz.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
