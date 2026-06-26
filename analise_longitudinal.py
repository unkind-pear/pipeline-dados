# -*- coding: utf-8 -*-
"""
Análise Longitudinal e de Trajetória Clínica do Adoecimento
Transtorno Bipolar do Humor (PROMAN - IPq-HC-FMUSP)

Este script reconstrói a linha do tempo média do adoecimento dos pacientes
mapeando três grandes marcos temporais de interesse psiquiátrico:
  1. T0: Idade de início dos primeiros sintomas do Transtorno Bipolar.
  2. T1: Idade da primeira hospitalização psiquiátrica (para os hospitalizados).
  3. T2: Idade atual dos participantes.

Também avalia a associação linear entre a Duração da Doença e o total de episódios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    path_dados = "./dados_projeto/db-longitudinal-tratado.xlsx"

    print("=" * 70)
    print(" ETAPA 3: ANÁLISE LONGITUDINAL DA TRAJETÓRIA CLÍNICA ".center(70, "="))
    print("=" * 70)

    try:
        df = pd.read_excel(path_dados)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de dados tratados não encontrado em: {path_dados}")
        print("Por favor, execute o script 'tratar_dados.py' primeiro.")
        return

    # Configuração de temas visuais
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    print("\n[+] Calculando as idades médias dos marcos temporais da doença...")

    # -------------------------------------------------------------------------
    # 1. T0: Idade de Início dos Sintomas
    # Filtra e remove outliers ou idades impossíveis
    # -------------------------------------------------------------------------
    df_clean_inic = df[(df['id_inic_doenca'] > 0) & (df['id_inic_doenca'] <= df['idade'])].dropna(subset=['id_inic_doenca'])
    media_inic = df_clean_inic['id_inic_doenca'].mean()

    # -------------------------------------------------------------------------
    # 2. T1: Idade na Primeira Hospitalização
    # Aplica-se apenas para os pacientes que de fato passaram por internações
    # -------------------------------------------------------------------------
    df_clean_hosp = df[(df['hospit'] == 1) & (df['id_1hos'] > 0) & (df['id_1hos'] >= df['id_inic_doenca'])].dropna(subset=['id_1hos'])
    media_1hos = df_clean_hosp['id_1hos'].mean()

    # -------------------------------------------------------------------------
    # 3. T2: Idade Atual da Coorte
    # -------------------------------------------------------------------------
    df_clean_idade = df[(df['idade'] > 0)].dropna(subset=['idade'])
    media_idade = df_clean_idade['idade'].mean()

    marcos = ['Início da Doença (T0)', '1ª Hospitalização (T1)', 'Idade Atual (T2)']
    valores = [media_inic, media_1hos, media_idade]

    print(f"  * T0 (Início Médio dos Sintomas):      {media_inic:.2f} anos")
    print(f"  * T1 (1ª Hospitalização Psiquiátrica): {media_1hos:.2f} anos")
    print(f"  * T2 (Idade Média Corrente):           {media_idade:.2f} anos")

    # -------------------------------------------------------------------------
    # RENDERIZAÇÃO: Gráfico da Trajetória Longitudinal
    # -------------------------------------------------------------------------
    print("\n[+] Renderizando gráfico de evolução temporal (linha do tempo)...")
    
    # Cria o lineplot ligando os marcos
    ax = sns.lineplot(x=marcos, y=valores, marker="o", color="crimson", linewidth=3, markersize=10, legend=False)

    # Anotação com rótulo e idade em cima de cada marcador
    for idx, val in enumerate(valores):
        ax.annotate(f"{val:.1f} anos", 
                    (idx, valores[idx] + 0.6), 
                    ha='center', 
                    fontsize=11, 
                    fontweight='bold', 
                    color='darkred')

    plt.title("Evolução Longitudinal Média do Transtorno Bipolar na Amostra", fontsize=13, fontweight='bold', pad=20)
    plt.ylabel("Idade do Paciente (anos)", fontsize=11)
    plt.xlabel("Estágio de Progresso Clínico Vitalício", fontsize=11)
    plt.ylim(0, max(valores) + 8)  # Margem superior para rótulos

    plt.tight_layout()
    plt.savefig('./evolucao_longitudinal.png', dpi=300)
    plt.close()
    print("[OK] Gráfico 'evolucao_longitudinal.png' exportado.")

    # -------------------------------------------------------------------------
    # RENDERIZAÇÃO: Relação Duração da Doença vs Total de Episódios
    # -------------------------------------------------------------------------
    print("\n[+] Calculando dispersão e regressão: Duração da Doença vs Frequência de Episódios...")
    if 'dur_doenca' in df.columns and 'nepisodios' in df.columns:
        df_scatter = df[['dur_doenca', 'nepisodios']].dropna()
        df_scatter = df_scatter[(df_scatter['dur_doenca'] > 0)]
        
        plt.figure(figsize=(10, 6))
        # Adiciona a reta de regressão linear para mostrar tendência de acúmulo
        sns.regplot(x='dur_doenca', y='nepisodios', data=df_scatter, 
                    scatter_kws={'alpha': 0.6, 'color': '#2c3e50'}, 
                    line_kws={'color': '#e74c3c', 'linewidth': 2}, 
                    fit_reg=True)
        
        plt.title("Dispersão e Relação: Duração do Diagnóstico vs Total de Episódios Acumulados", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Duração da Doença Ativa (anos)", fontsize=11)
        plt.ylabel("Número Total de Episódios Clínicos (nepisodios)", fontsize=11)
        
        plt.tight_layout()
        plt.savefig('./relacao_doenca_episodios.png', dpi=300)
        plt.close()
        print("[OK] Gráfico 'relacao_doenca_episodios.png' exportado.")
    else:
        print("[AVISO] Colunas 'dur_doenca' ou 'nepisodios' ausentes. Gráfico de dispersão abortado.")

    print("=" * 70)
    print(" ANÁLISE LONGITUDINAL CONCLUÍDA! ".center(70, "="))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
