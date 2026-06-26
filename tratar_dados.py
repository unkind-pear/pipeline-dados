# -*- coding: utf-8 -*-
"""
Pipeline de Curadoria e Tratamento de Dados Clínicos
Transtorno Bipolar do Humor (PROMAN - IPq-HC-FMUSP)

Este script realiza a limpeza, correção de inconsistências clínicas e 
imputação de dados ausentes utilizando MICE (Multiple Imputation by Chained Equations).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.experimental import enable_iterative_imputer  # Necessário para habilitar o IterativeImputer
from sklearn.impute import IterativeImputer

def main():
    print("=" * 70)
    print(" ETAPA 1: TRATAMENTO E CURADORIA DE DADOS CHAVE ".center(70, "="))
    print("=" * 70)

    # 1. Leitura do Banco de Dados Bruto
    path_dados = "./dados_projeto/db-longitudinal.xlsx"
    path_dicionario = "./dados_projeto/dicionario_variaveis.csv"
    
    try:
        df = pd.read_excel(path_dados)
        print(f"[OK] Banco de dados bruto carregado: N={len(df)} pacientes, D={len(df.columns)} colunas.")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo de dados '{path_dados}': {e}")
        return

    # Registra a porcentagem de dados faltantes antes de qualquer alteração
    faltantes_antes = (df.isnull().sum() / len(df) * 100)

    # -------------------------------------------------------------------------
    # CORREÇÃO CLÍNICA: Colunas Invertidas no Banco de Dados Original
    # No banco bruto, 'n_mania' e 'n_depres' estavam invertidos, o que gerava 
    # a contradição de pacientes Tipo II (que por definição não têm episódios
    # de mania) apresentarem mania e não depressão. Desfazemos a troca aqui.
    # -------------------------------------------------------------------------
    print("\n[+] Corrigindo inconsistência clínica: Desfazendo inversão de n_mania e n_depres...")
    df.rename(columns={'n_mania': 'n_depres_temp', 'n_depres': 'n_mania_temp'}, inplace=True)
    df.rename(columns={'n_depres_temp': 'n_depres', 'n_mania_temp': 'n_mania'}, inplace=True)

    # 2. Remoção de Variáveis Irrecuperáveis (100% nulas)
    # Variáveis sem qualquer registro útil são descartadas para não poluir o modelo
    cols_nulas = ["fer_cab", "asma", "diabetes"]
    print(f"[+] Removendo colunas irrecuperáveis (100% nulas): {cols_nulas}")
    df.drop(columns=cols_nulas, inplace=True, errors='ignore')

    # 3. Preenchimentos Clínicos Conservadores
    # Se o número de episódios é nulo, clinicamente assume-se 0 (ausência de registro = sem episódio)
    cols_zerar = ["nepisodios", "n_mania", "n_depres", "n_hipoma", "n_mistos", "tansiedadelifetime"]
    print(f"[+] Aplicando preenchimento conservador (nulos -> 0) para contagens de episódios: {cols_zerar}")
    df[cols_zerar] = df[cols_zerar].fillna(0)

    # Imputação simples do IMC pela média (dado antropométrico estável)
    imc_media = round(df['imc'].mean(), 2)
    print(f"[+] Imputando valores ausentes de IMC pela média da amostra: {imc_media} kg/m²")
    df['imc'] = df['imc'].fillna(imc_media)

    # 4. Lógica de Reconstrução Longitudinal Lifetime
    # Pacientes com comorbidades diagnosticadas no atual OU passado devem apresentar
    # a presença vitalícia (lifetime) ativa. Reconstruímos isso logicamente.
    # L(X) = X_at OR X_pas = max(X_at, X_pas)
    lifetime_map = {
        "teptlifetime": ["tept_at", "tept_pas"], 
        "panicosemgorafobialifetime": ["pan_saga", "pan_sagp"],
        "anorexialifetime": ["anorex_a", "anorexia_p"], 
        "panicagorafobialifetime": ["pan_agoa", "pan_agop"],
        "abusoalcoollifetime": ["abus_a", "abus_p"], 
        "depalcoollifetime": ["dep_a", "dependenciaalcoolpassado"],
        "toclifetime": ["toc_at", "toc_pas"], 
        "agorafobialifetime": ["agoraf_a", "agoraf_p"],
        "fobiasocialalifetime": ["fob_soca", "fob_socp"], 
        "taglifetime": ["tag_at", "tag_pas"],
        "tanssoelifetime": ["ta_soea", "ta_soep"], 
        "bulimialifetime": ["bulim_at", "bulim_ps"],
        "compulsalimlifetime": ["compulsaoalimatual", "tcap_pas"], 
        "abusosubstllifetime": ["abusn_a", "abusn_p"],
        "depsubstllifetime": ["depen_a", "depen_p"], 
        "fobiaespecificaifetime": ["fob_espa", "fob_espp"]
    }

    print("[+] Reconstruindo variáveis 'lifetime' com base na disjunção lógica (atual OR passado)...")
    for alvo, (f1, f2) in lifetime_map.items():
        if alvo in df.columns and f1 in df.columns and f2 in df.columns:
            nulos = df[alvo].isnull()
            df.loc[nulos, alvo] = ((df.loc[nulos, f1] == 1) | (df.loc[nulos, f2] == 1)).astype(int)
            # Garantir que se o lifetime foi resolvido, preenchemos os subcampos como 0 se fossem nulos
            df[[f1, f2]] = df[[f1, f2]].fillna(0)

    # 5. Correção de Inconsistências Condicionais
    # - Pacientes sem histórico de internação (hospit=2) não podem ter contagens de hospitalizações
    if 'hospit' in df.columns and 'quantas' in df.columns:
        print("[+] Ajustando logicamente: Pacientes não internados (hospit=2) definidos com 0 hospitalizações.")
        df.loc[df["hospit"] == 2, "quantas"] = 0
    # - Pacientes sem histórico de tentativa de suicídio (tent_sui=2) têm 0 tentativas de suicídio
    if 'tent_sui' in df.columns and 'n_tent' in df.columns:
        print("[+] Ajustando logicamente: Pacientes sem tentativa de suicídio (tent_sui=2) definidos com 0 tentativas.")
        df.loc[df["tent_sui"] == 2, "n_tent"] = 0

    # Identificar colunas que ainda precisam de imputação estatística (MICE)
    colunas_com_nulos = df.columns[df.isnull().any()]
    print(f"\n[INFO] Colunas restantes com dados nulos a serem imputadas via MICE: {list(colunas_com_nulos)}")

    # 6. Imputação Estatística via MICE (IterativeImputer)
    print("\n[+] Executando Imputação Múltipla por Equações Encadeadas (MICE)...")
    
    # Selecionamos apenas colunas numéricas para o MICE, ignorando 'tip_subs' (categórica nominal de texto)
    cols_num = df.select_dtypes(include=[np.number]).columns.drop('tip_subs', errors='ignore')
    
    # Inicializa o MICE com 10 iterações e seed aleatória fixa para reprodutibilidade
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[cols_num] = imputer.fit_transform(df[cols_num])

    # 7. Pós-processamento de Imputações
    # Como o MICE opera no domínio contínuo, ele pode imputar floats (ex: 1.45) ou valores
    # negativos (ex: -1.2) para contagens inteiras ou variáveis binárias.
    # Usamos o dicionário de variáveis para arredondar e limitar os valores aos seus domínios legais.
    print("[+] Executando arredondamento e limitação (clipping) pós-MICE usando dicionário de variáveis...")
    try:
        dic = pd.read_csv(path_dicionario)
        for col in cols_num:
            if col in dic['Variável'].values:
                row_dic = dic[dic['Variável'] == col].iloc[0]
                tipo = str(row_dic['Tipo de Variável']).lower()
                valores_str = str(row_dic['Valores'])
                
                # Se for discreta ou qualitativa, arredondamos para o inteiro mais próximo
                if 'discreta' in tipo or 'qualitativa' in tipo:
                    df[col] = df[col].round()
                    
                    # Restrição por Domínio Existente (Valores válidos descritos no dicionário como {1, 2, ...})
                    if '{' in valores_str and '}' in valores_str:
                        valores = [float(x) if '.' in x else int(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', valores_str.split('{')[1].split('}')[0])]
                        if valores:
                            df[col] = df[col].clip(lower=min(valores), upper=max(valores))
                    elif 'discreta' in tipo:
                        # Contagens (ex: número de episódios) não podem ser negativas
                        df[col] = df[col].clip(lower=0)
    except Exception as e:
        print(f"[AVISO] Falha ao processar dicionário para pós-processamento: {e}")

    # 8. Engenharia de Variável Final: talcoolsubstlifetime
    # Define presença de comorbidade combinada de Álcool E Substâncias ao longo da vida
    if all(c in df.columns for c in ["tnaoalcoollifetime", "talcoollifetime", "talcoolsubstlifetime"]):
        df["talcoolsubstlifetime"] = 0
        df.loc[(df["tnaoalcoollifetime"] == 1) & (df["talcoollifetime"] == 1), "talcoolsubstlifetime"] = 1

    # 9. Geração de Relatórios Faltantes (Antes vs Depois)
    faltantes_depois = (df.isnull().sum() / len(df) * 100)
    df_plot = pd.DataFrame({'Antes (%)': faltantes_antes, 'Depois (%)': faltantes_depois})
    df_plot = df_plot[df_plot['Antes (%)'] > 0].sort_values(by='Antes (%)', ascending=False)

    print("\n" + "-" * 60)
    print(" RESUMO DA REDUÇÃO DE DADOS AUSENTES (Antes vs Depois) ".center(60, "-"))
    print("-" * 60)
    print(df_plot.to_string(float_format="%.2f")) 
    print("-" * 60 + "\n")

    # Exportar gráfico das 10 variáveis mais faltantes originais
    if not df_plot.empty:
        sns.set_theme(style="whitegrid")
        df_top10 = df_plot.head(10).copy()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Antes (%)', y=df_top10.index, data=df_top10, palette='Reds_r', hue=df_top10.index, legend=False)
        plt.title("Top 10 Variáveis com Mais Dados Faltantes (Original vs Pós-Tratamento)", fontsize=14, fontweight='bold')
        plt.xlabel("Porcentagem Faltante Original (%)")
        plt.ylabel("Variáveis")
        
        # Adicionar os valores exatos de texto na ponta das barras
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2. + 0.15, '{:1.1f}%'.format(width), ha="left", va="center", fontweight='bold')

        plt.xlim(0, 110)
        plt.tight_layout()
        plt.savefig("./top_10_faltantes.png", dpi=300)
        plt.close()
        print("[OK] Gráfico comparativo de dados nulos salvo como 'top_10_faltantes.png'")

    # 10. Exportação dos Dados Tratados
    try:
        df.to_excel("./dados_projeto/db-longitudinal-tratado.xlsx", index=False)
        df.to_csv("./dados_projeto/db-longitudinal-tratado.csv", index=False, encoding='utf-8')
        print("[OK] Arquivos tratados salvos com sucesso em './dados_projeto/' (Formatos XLSX e CSV).")
    except Exception as e:
        print(f"[ERRO] Falha ao exportar os dados tratados: {e}")

    print("=" * 70)
    print(" TRATAMENTO E CURADORIA FINALIZADOS COM SUCESSO! ".center(70, "="))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
