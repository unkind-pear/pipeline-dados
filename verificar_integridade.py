# -*- coding: utf-8 -*-
"""
Auditoria de Inconsistências e Integridade dos Dados
Transtorno Bipolar do Humor (PROMAN - IPq-HC-FMUSP)

Este script escaneia o banco de dados original em busca de contradições 
lógicas, clínicas, matemáticas e violações de intervalo de valores (domínio), 
mapeando a qualidade original da base de dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def main():
    path_dados = "./dados_projeto/db-longitudinal.xlsx"
    path_guia = "./dados_projeto/dicionario_variaveis.csv"

    print("=" * 70)
    print(" SCRIPT DE AUDITORIA DE INTEGRIDADE E QUALIDADE ".center(70, "="))
    print("=" * 70)

    try:
        df = pd.read_excel(path_dados)
        guia = pd.read_csv(path_guia)
        print("[OK] Arquivos originais carregados com sucesso.")
    except FileNotFoundError:
        print(f"[ERRO] Arquivos não encontrados nos caminhos informados:\n - '{path_dados}'\n - '{path_guia}'")
        return

    inconsistencias = []
    auditadas = set()

    def auditar(mask, var_alvo, func_msg, tipo_erro, cols_envolvidas):
        """
        Registra uma inconsistência detectada na base filtrada pela máscara lógica (mask).
        Esta função é otimizada para aplicar a lógica de registro de erros rapidamente.
        """
        cols_presentes = [c for c in cols_envolvidas if c in df.columns]
        auditadas.update(cols_presentes)
        
        if mask is not None and mask.sum() > 0:
            df_err = df[mask]
            inconsistencias.append(pd.DataFrame({
                'Linha': df_err.index,
                'ID_Paciente': df_err.get('id', 'N/A'),
                'Variavel_Foco': var_alvo,
                'Erro': df_err.apply(func_msg, axis=1),
                'Tipo': tipo_erro
            }))

    # -------------------------------------------------------------------------
    # [1] INCONSISTÊNCIAS DE RELAÇÃO CONDICIONAL
    # -------------------------------------------------------------------------
    tipo = "Relação Condicional"

    # Tentativa de suicídio (se tent_sui=2 [Não], o número de tentativas n_tent deve ser 0/vazio)
    if all(c in df.columns for c in ['tent_sui', 'n_tent']):
        auditar((df['tent_sui'] == 2) & (df['n_tent'] > 0), 
                'tent_sui', 
                lambda r: f"tent_sui=Não(2), mas n_tent={r['n_tent']}", 
                tipo, ['tent_sui', 'n_tent'])

    # Uso de substância (se subst_1=2 [Não], o tipo de substância tip_subs deve ser nulo)
    if all(c in df.columns for c in ['subst_1', 'tip_subs']):
        auditar((df['subst_1'] == 2) & df['tip_subs'].notna(), 
                'subst_1', 
                lambda r: f"subst_1=Não(2), mas tip_subs={r['tip_subs']}", 
                tipo, ['subst_1', 'tip_subs'])

    # Cronologia de tratamento (paciente não pode iniciar medicação após estabilizar)
    if all(c in df.columns for c in ['id_med_p', 'id_estab']):
        auditar(df['id_med_p'] > df['id_estab'], 
                'id_med_p', 
                lambda r: f"Idade medicação ({r['id_med_p']}) > Idade estabilização ({r['id_estab']})", 
                tipo, ['id_med_p', 'id_estab'])

    # Inconsistência de idade geral (marcos clínicos excedem a idade atual do paciente)
    cols_idade = ['dur_doenca', 'an_compl', 'idade', 'id_med_p', 'id_estab']
    if all(c in df.columns for c in cols_idade):
        mask = (df['dur_doenca'] > df['idade']) | (df['an_compl'] > df['idade']) | (df['id_med_p'] > df['idade']) | (df['id_estab'] > df['idade'])
        auditar(mask, 'idade', 
                lambda r: f"Idade atual ({r['idade']}) é inferior a marcos históricos clínicos", 
                tipo, cols_idade)

    # Coerência de Hospitalizações (se hospit=2 [Não], contagem e idade da 1a hosp. devem ser nulas)
    if 'hospit' in df.columns:
        if 'quantas' in df.columns: 
            auditar((df['hospit'] == 2) & (df['quantas'] > 0), 
                    'hospit', 
                    lambda r: f"hospit=Não(2), mas quantas={r['quantas']}", 
                    tipo, ['hospit', 'quantas'])
        if 'id_1hos' in df.columns: 
            auditar((df['hospit'] == 2) & (df['id_1hos'] > 0), 
                    'hospit', 
                    lambda r: f"hospit=Não(2), mas id_1hos={r['id_1hos']}", 
                    tipo, ['hospit', 'id_1hos'])

    # Coerência de Patologias Clínicas (se diagnóstico primário ausente, secundário não pode estar presente)
    for v1, v2 in [('conv_atq', 'convulsao12'), ('enxaq', 'enxaqueca12'), ('hipotir', 'hipo12'), ('hipertir', 'hiper12')]:
        if v1 in df.columns and v2 in df.columns:
            auditar((df[v1] == 2) & (df[v2] == 1), 
                    v1, 
                    lambda r, var=v1, var2=v2: f"{var}=Não(2), mas {var2}=Presente(1) no detalhado", 
                    tipo, [v1, v2])

    # -------------------------------------------------------------------------
    # [2] INCONSISTÊNCIAS DE LIFETIME VS ATUAL/PASSADO
    # -------------------------------------------------------------------------
    tipo = "Lifetime vs Atual/Passado"
    # O indicador vitalício (lifetime) deve estar presente se o sintoma atual ou o passado for 1
    vidas = [
        ("panicosemgorafobialifetime", "pan_saga", "pan_sagp"), 
        ("anorexialifetime", "anorex_a", "anorexia_p"), 
        ("panicagorafobialifetime", "pan_agoa", "pan_agop"), 
        ("abusoalcoollifetime", "abus_a", "abus_p"), 
        ("depalcoollifetime", "dep_a", "dependenciaalcoolpassado"), 
        ("toclifetime", "toc_at", "toc_pas"), 
        ("agorafobialifetime", "agoraf_a", "agoraf_p"), 
        ("fobiasocialalifetime", "fob_soca", "fob_socp"), 
        ("taglifetime", "tag_at", "tag_pas"), 
        ("tanssoelifetime", "ta_soea", "ta_soep"), 
        ("bulimialifetime", "bulim_at", "bulim_ps"), 
        ("compulsalimlifetime", "compulsaoalimatual", "tcap_pas"), 
        ("abusosubstllifetime", "abusn_a", "abusn_p"), 
        ("depsubstllifetime", "depen_a", "depen_p"), 
        ("teptlifetime", "tept_at", "tept_pas"), 
        ("fobiaespecificaifetime", "fob_espa", "fob_espp")
    ]
    for l, a, p in vidas:
        if all(c in df.columns for c in [l, a, p]):
            auditar((df[l] == 0) & ((df[a] == 1) | (df[p] == 1)), 
                    l, 
                    lambda r, rl=l: f"{rl}=Ausente(0), mas atual ou passado estão marcados como Presente(1)", 
                    tipo, [l, a, p])

    # -------------------------------------------------------------------------
    # [3] INCONSISTÊNCIAS MATEMÁTICAS (SOMAS E ESTRUTURAS)
    # -------------------------------------------------------------------------
    tipo = "Matemática"
    
    # Soma de classes de medicamentos prescritos deve bater com o total informado
    med_cols = [c for c in ['litio', 'anticonvulsionante', 'antipsicoticoatipico', 'antipsicoticoclassico', 'antidepressivo'] if c in df.columns]
    if 'nclassesmedicacao' in df.columns and med_cols:
        soma_meds = df[med_cols].fillna(0).sum(axis=1)
        auditar(df['nclassesmedicacao'].notna() & (df['nclassesmedicacao'] != soma_meds) & (df['nclassesmedicacao'] != 0), 
                'nclassesmedicacao', 
                lambda r: f"Soma dos medicamentos ({soma_meds[r.name]}) difere de nclassesmedicacao={r['nclassesmedicacao']}", 
                tipo, ['nclassesmedicacao'] + med_cols)

    # Soma de subtipos de episódios de humor deve ser igual ao número total de episódios
    epi_cols = [c for c in ['n_mania', 'n_depres', 'n_hipoma', 'n_mistos'] if c in df.columns]
    if 'nepisodios' in df.columns and epi_cols:
        soma_epi = df[epi_cols].fillna(0).sum(axis=1)
        auditar(df['nepisodios'].notna() & (abs(df['nepisodios'] - soma_epi) > 0) & (df['nepisodios'] != 0) & (soma_epi != 0), 
                'nepisodios', 
                lambda r: f"Soma dos episódios ({soma_epi[r.name]}) difere de nepisodios={r['nepisodios']}", 
                tipo, ['nepisodios'] + epi_cols)

    # Coerência da comorbidade de álcool e substâncias
    if all(c in df.columns for c in ['talcoolsubstlifetime', 'talcoollifetime', 'tnaoalcoollifetime']):
        auditar((df['talcoolsubstlifetime'] == 1) & (df['talcoollifetime'] == 0) & (df['tnaoalcoollifetime'] == 0), 
                'talcoolsubstlifetime', 
                lambda r: "Comorbidade Álcool/Substância marcada (1), mas subcampos indicam ausência (0)", 
                tipo, ['talcoolsubstlifetime', 'talcoollifetime', 'tnaoalcoollifetime'])
        
    auditadas.update(['comorbidadelifetime1', 'comorbidadepassado1', 'comorbidadeatual1'])

    # -------------------------------------------------------------------------
    # [4] INCONSISTÊNCIAS DE VALORES FORA DO DOMÍNIO (RANGE)
    # -------------------------------------------------------------------------
    tipo = "Range/Domínio"
    for _, row in guia.dropna(subset=['Variável', 'Valores']).iterrows():
        var, vals = row['Variável'], str(row['Valores'])
        if var in df.columns and '{' in vals and '}' in vals:
            # Extrair valores válidos aceitos entre as chaves usando Regex
            permitidos = [float(x) if '.' in x else int(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', vals.split('{')[1].split('}')[0])]
            if permitidos:
                auditar(df[var].notna() & ~df[var].isin(permitidos), 
                        var, 
                        lambda r, v=var, p=permitidos: f"Valor {r[v]} fora do domínio de opções permitidas {p}", 
                        tipo, [var])

    # -------------------------------------------------------------------------
    # EXIBIÇÃO DE RESULTADOS E EXPORTAÇÃO GRÁFICA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" RELATÓRIO FINAL DA AUDITORIA DE INTEGRIDADE ".center(70, "="))
    print("=" * 70)

    if not inconsistencias:
        print("[SUCESSO] Nenhuma inconsistência lógica ou de domínio detectada na base!")
    else:
        df_erros = pd.concat(inconsistencias, ignore_index=True)
        print(f"Total de Inconsistências Mapeadas na Base Bruta: {len(df_erros)}\n")
        
        # Resumo por Variável e Tipo
        resumo = df_erros.groupby(['Tipo', 'Variavel_Foco']).size().reset_index(name='Quantidade').sort_values('Quantidade', ascending=False)
        resumo['Porcentagem Pacientes (%)'] = (resumo['Quantidade'] / len(df) * 100).round(2).astype(str) + '%'
        print(resumo.to_string(index=False))
        
        # Configurar estilos de gráficos
        sns.set_theme(style="whitegrid")
        
        # Gráfico 1: Top Variáveis Afetadas
        plt.figure(figsize=(10, 6))
        top_vars = df_erros['Variavel_Foco'].value_counts().head(12)
        sns.barplot(x=top_vars.values, y=top_vars.index, hue=top_vars.index, palette="flare", legend=False)
        plt.title('Top 12 Variáveis com Mais Inconsistências de Registro', fontsize=13, fontweight='bold')
        plt.xlabel('Contagem absoluta de erros')
        plt.ylabel('Nome da Variável')
        plt.tight_layout()
        plt.savefig('./grafico_inconsistencias_top_vars.png', dpi=300)
        plt.close()
        
        # Gráfico 2: Divisão por Tipo de Regra Violada
        plt.figure(figsize=(8, 5))
        tipos = df_erros['Tipo'].value_counts()
        ax = sns.barplot(x=tipos.index, y=tipos.values, hue=tipos.index, palette="viridis", legend=False)
        plt.title('Inconsistências por Tipo de Regra Violada', fontsize=13, fontweight='bold')
        plt.ylabel('Quantidade de Registros Incoerentes')
        plt.xlabel('Categoria de Regra')
        ax.bar_label(ax.containers[0], fontweight='bold')
        plt.tight_layout()
        plt.savefig('./grafico_inconsistencias_tipo.png', dpi=300)
        plt.close('all')
        
        print("\n[OK] Gráficos de auditoria salvos em:")
        print(" -> './grafico_inconsistencias_top_vars.png'")
        print(" -> './grafico_inconsistencias_tipo.png'")
        
        print("\n" + "-" * 70)
        print(" Amostra dos 5 Primeiros Casos de Inconsistência ".center(70, "-"))
        print("-" * 70)
        print(df_erros[['ID_Paciente', 'Variavel_Foco', 'Erro', 'Tipo']].head().to_string(index=False))
        print("-" * 70)

    # Sumarizar número de pacientes afetados
    total_pacientes = len(df)
    pacientes_com_erro = df_erros['ID_Paciente'].nunique() if inconsistencias else 0
    porcentagem = (pacientes_com_erro / total_pacientes) * 100

    print(f"\n[RESUMO] Pacientes Afetados por Inconsistências: {pacientes_com_erro} de {total_pacientes} ({porcentagem:.2f}%).")
    
    puladas = set(df.columns) - auditadas
    print(f"[INFO] Variáveis puladas na auditoria (sem regras associadas): {len(puladas)}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
