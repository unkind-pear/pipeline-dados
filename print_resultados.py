# -*- coding: utf-8 -*-
"""
Visualizador de Resultados e Consolidação das Tabelas do Artigo
Transtorno Bipolar do Humor (PROMAN - IPq-HC-FMUSP)

Este script lê a base de dados tratada e o dicionário de variáveis e exibe
de forma limpa e formatada no terminal todas as 5 tabelas contidas no artigo
e no relatório de pesquisa. Serve como validador final de concordância estatística.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_ind
import warnings

warnings.filterwarnings('ignore')

def format_p(p):
    """Auxiliar para formatação do p-valor no estilo científico ou decimal curto."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def main():
    print("=" * 80)
    print(" CONSOLIDAÇÃO DE TODOS OS RESULTADOS ESTATÍSTICOS DO PROJETO ".center(80, "="))
    print("=" * 80)

    # 1. Carregamento dos dados
    path_raw = "./dados_projeto/db-longitudinal.xlsx"
    path_trat = "./dados_projeto/db-longitudinal-tratado.xlsx"

    try:
        df_raw = pd.read_excel(path_raw)
        df_trat = pd.read_excel(path_trat)
    except FileNotFoundError:
        print("[ERRO] Falha ao localizar os arquivos de dados na pasta './dados_projeto/'.")
        print("Certifique-se de executar o tratamento 'tratar_dados.py' primeiro.")
        return

    # Criar versão imputada pela média para a tabela de sensibilidade
    df_mean = df_raw.copy()
    cols_num = df_mean.select_dtypes(include=[np.number]).columns
    for col in cols_num:
        df_mean[col] = df_mean[col].fillna(df_mean[col].mean())

    # =========================================================================
    # TABELA 1: CARACTERÍSTICAS GERAIS DA AMOSTRA (N=139)
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TABELA 1: Características Gerais da Amostra (N=139) ".center(80, "-"))
    print("-" * 80)
    
    tb1_count = (df_trat["diagnostico"] == 1).sum()
    fem_count = (df_trat["sexo"] == 2).sum()
    white_count = (df_trat["caucasiano"] == 1).sum()
    casado_count = (df_trat["est_civil"] == 2).sum()
    solteiro_count = (df_trat["est_civil"] == 1).sum()
    
    # Ocupação informada (excluindo ausentes na base bruta original)
    df_ocup_val = df_raw[df_raw["ocup"].notnull()]
    ocup_ativa_count = (df_ocup_val["ocup"] == 2).sum() 
    ocup_total = len(df_ocup_val)

    print(f"  * Diagnóstico TB-I:              {tb1_count} pacientes ({tb1_count/139*100:.1f}%)")
    print(f"  * Sexo Feminino:                 {fem_count} pacientes ({fem_count/139*100:.1f}%)")
    print(f"  * Etnia Branca:                  {white_count} pacientes ({white_count/139*100:.1f}%)")
    print(f"  * Estado Civil Casado:           {casado_count} pacientes ({casado_count/139*100:.1f}%)")
    print(f"  * Estado Civil Solteiro:         {solteiro_count} pacientes ({solteiro_count/139*100:.1f}%)")
    print(f"  * Ocupação Remunerada Ativa:     {ocup_ativa_count} de {ocup_total} informados ({ocup_ativa_count/ocup_total*100:.1f}%)")
    print("  " + "." * 76)
    print(f"  * Idade Atual (Anos):            {df_trat['idade'].mean():.2f} ± {df_trat['idade'].std():.2f} (Min: {df_trat['idade'].min():.0f}, Max: {df_trat['idade'].max():.0f})")
    print(f"  * IMC (kg/m²):                   {df_trat['imc'].mean():.2f} ± {df_trat['imc'].std():.2f} (Min: {df_trat['imc'].min():.2f}, Max: {df_trat['imc'].max():.2f})")
    print(f"  * Escolaridade (Anos):           {df_trat['an_compl'].mean():.2f} ± {df_trat['an_compl'].std():.2f} (Min: {df_trat['an_compl'].min():.0f}, Max: {df_trat['an_compl'].max():.0f})")
    print(f"  * Idade Início TB (Anos):        {df_trat['id_inic_doenca'].mean():.2f} ± {df_trat['id_inic_doenca'].std():.2f} (Min: {df_trat['id_inic_doenca'].min():.0f}, Max: {df_trat['id_inic_doenca'].max():.0f})")

    # =========================================================================
    # TABELA 2: COMPARAÇÃO ENTRE SUBTIPOS DIAGNÓSTICOS (TB-I vs TB-II)
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TABELA 2: Comparação de Médias por Subtipo de Transtorno Bipolar ".center(80, "-"))
    print("-" * 80)
    print(f"  {'Marcador Clínico':<40} | {'TB-I (N=127)':<15} | {'TB-II (N=12)':<15} | {'Valor-p (Welch)':<10}")
    print("  " + "." * 88)

    g1_diag = df_trat[df_trat["diagnostico"] == 1]
    g2_diag = df_trat[df_trat["diagnostico"] == 2]

    for label, col in [("Quantidade de hospitalizações", "quantas"), 
                       ("Episódios de mania", "n_mania"), 
                       ("Episódios de hipomania isolada", "n_hipoma")]:
        m1 = g1_diag[col].mean()
        m2 = g2_diag[col].mean()
        _, p = ttest_ind(g1_diag[col], g2_diag[col], equal_var=False)
        print(f"  {label:<40} | {m1:<15.2f} | {m2:<15.2f} | {format_p(p):<10}")

    # =========================================================================
    # TABELA 3: CORRELAÇÕES DE SPEARMAN SIGNIFICATIVAS
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TABELA 3: Correlações de Spearman na Base Tratada (N=139) ".center(80, "-"))
    print("-" * 80)
    print(f"  {'Variável 1':<30} | {'Variável 2':<30} | {'Coeficiente (rho)':<18} | {'Valor-p':<10}")
    print("  " + "." * 94)

    corrs = [
        ("idade", "quantas", "Idade atual", "Número de hospitalizações"),
        ("n_mania", "n_depres", "Episódios de Mania", "Episódios Depressivos"),
        ("id_inic_doenca", "quantas", "Idade início TB", "Número de hospitalizações")
    ]

    for c1, c2, l1, l2 in corrs:
        rho, p = spearmanr(df_trat[c1], df_trat[c2])
        print(f"  {l1:<30} | {l2:<30} | {rho:<18.3f} | {format_p(p):<10}")

    # =========================================================================
    # TABELA 4: COMPARAÇÃO POR HISTÓRICO DE TENTATIVA DE SUICÍDIO
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TABELA 4: Comparação por Histórico de Tentativa de Suicídio ".center(80, "-"))
    print("-" * 80)
    print(f"  {'Marcador Clínico':<45} | {'Tentaram (N=54)':<17} | {'Nunca (N=85)':<15} | {'Valor-p (Welch)':<10}")
    print("  " + "." * 95)

    g1_tent = df_trat[df_trat["tent_sui"] == 1]
    g2_tent = df_trat[df_trat["tent_sui"] == 2]

    tests_tent = [
        ("Idade de início da doença (anos)", "id_inic_doenca"),
        ("Idade início primeira medicação (anos)", "id_med_p"),
        ("Idade primeira hospitalização (anos)", "id_1hos"),
        ("Idade corrente (anos)", "idade"),
        ("Idade de estabilização clínica (anos)", "id_estab"),
        ("Contagem de comorbidades ao longo da vida", "comorbidadelifetime1")
    ]

    for label, col in tests_tent:
        m1 = g1_tent[col].mean()
        m2 = g2_tent[col].mean()
        _, p = ttest_ind(g1_tent[col], g2_tent[col], equal_var=False)
        print(f"  {label:<45} | {m1:<17.2f} | {m2:<15.2f} | {format_p(p):<10}")

    # =========================================================================
    # TABELA 5: ANÁLISE DE SENSIBILIDADE (COMPARAÇÃO DE IMPUTAÇÕES)
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TABELA 5: Análise de Sensibilidade e Comparação de Imputações ".center(80, "-"))
    print("-" * 80)
    print(f"  {'Par de Variáveis':<25} | {'Dados Brutos (Pairwise)':<25} | {'Imputação Média':<18} | {'Pipeline MICE (Ours)':<22}")
    print("  " + "." * 96)

    sens_pairs = [
        ("idade", "quantas", "Idade x Hospit."),
        ("n_mania", "n_depres", "Mania x Depres."),
        ("id_inic_doenca", "quantas", "Início TB x Hospit.")
    ]

    for c1, c2, label in sens_pairs:
        # Dados Brutos Pareados (Pairwise Deletion)
        mask_raw = df_raw[[c1, c2]].notnull().all(axis=1)
        n_raw = mask_raw.sum()
        rho_raw, p_raw = spearmanr(df_raw.loc[mask_raw, c1], df_raw.loc[mask_raw, c2])
        raw_str = f"{rho_raw:.3f} (p={format_p(p_raw)}, N={n_raw})"
        
        # Imputação Simples pela Média
        rho_mean, p_mean = spearmanr(df_mean[c1], df_mean[c2])
        mean_str = f"{rho_mean:.3f} (p={format_p(p_mean)})"
        
        # Imputação Múltipla MICE Proposta
        rho_mice, p_mice = spearmanr(df_trat[c1], df_trat[c2])
        mice_str = f"{rho_mice:.3f} (p={format_p(p_mice)})"
        
        print(f"  {label:<25} | {raw_str:<25} | {mean_str:<18} | {mice_str:<22}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
