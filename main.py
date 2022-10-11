import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import streamlit as st
import altair as alt

bra = pd.read_feather('bra2022.feather')
bra = bra[bra['mplacar'].notnull()]
times = bra['mandante'].unique()
tabela = pd.DataFrame()
for time in times:
    vit = emp = der = pro = con = jog = 0
    mandante = bra[bra['mandante'] == time]
    for indice, partida in mandante.iterrows():
        if partida['mplacar'] > partida['vplacar']:
            vit += 1 
        elif partida['mplacar'] == partida['vplacar']:
            emp += 1
        else:
            der += 1
        pro += partida['mplacar']
        con += partida['vplacar']
        jog += 1
    visitante = bra[bra['visitante'] == time]
    for indice, partida in visitante.iterrows():
        if partida['vplacar'] > partida['mplacar']:
            vit += 1
        elif partida['vplacar'] == partida['mplacar']:
            emp += 1
        else:
            der += 1
        pro += partida['vplacar']
        con += partida['mplacar']
        jog += 1
    new_row = {
        '1-Time': time,
        '3-J': jog,
        '4-V': vit,
        '5-E' : emp,
        '6-D': der,
        '7-GP': pro,
        '8-GC': con
    }    
    tabela = tabela.append(
        new_row,
        ignore_index = True
    )
tabela['2-P'] = (tabela['4-V'] * 3) + tabela['5-E']
tabela['9-SG'] = tabela['7-GP'] - tabela['8-GC']   
tabela_sort = tabela.sort_values(by=['2-P','4-V','9-SG'], ascending=False)
st.table(tabela_sort)

df_data = tabela[[
    '2-P',              
    '4-V',
    '5-E',
    '6-D',
    '9-SG'
]]
kmeans = KMeans(n_clusters=5, random_state=0).fit(df_data)
tabela['cluster'] = kmeans.labels_

df_cluster = pd.DataFrame()
for cluster, colunas in enumerate(kmeans.cluster_centers_):
    new_row = {
        'cluster': cluster, 
        '2-P': colunas[0],              
        '4-V': colunas[1],
        '5-E': colunas[2],
        '6-D': colunas[3],
        '9-SG': colunas[4],
    }
    df_cluster = df_cluster.append(
        new_row,
        ignore_index=True
    )
df_cluster_sort = df_cluster.sort_values(by='2-P', ascending=False)
df_cluster_sort['grupo'] = ['Titulo','Libertadores','Sul-Americana','limbo','Rebaixamento']
df_cluster_grupo = df_cluster_sort[['cluster','grupo']]
tabela = tabela.merge(df_cluster_grupo, on='cluster', how='left')
tabela_sort = tabela.sort_values(by=['2-P','4-V','9-SG'], ascending=False)
st.table(tabela_sort)


df_data = tabela[[
    'cluster',              
    '2-P',              
    '4-V',
    '5-E',
    '6-D',
    '9-SG'
]]

X_Train = df_data.drop(columns=['cluster'], axis=1)
X_Test = df_data.drop(columns=['cluster'], axis=1)
y_Train = df_data['cluster']
y_Test = df_data['cluster']

sc_x = StandardScaler()
X_Train = sc_x.fit_transform(X_Train)
X_Test = sc_x.fit_transform(X_Test)

logreg = LogisticRegression(solver="lbfgs", max_iter=500)
logreg.fit(X_Train, y_Train)
pred_logreg = logreg.predict(X_Test)
pred_proba = logreg.predict_proba(X_Test)

tabela["cluster_pred"] = pred_logreg

lista_proba = pred_proba.tolist()
df_prob_xy = pd.DataFrame()
index = 0
for proba in lista_proba:
    for i in range(0, len(proba)):
        new_row = {"index": index, "prob": i, "valor": round(proba[i], 4)}
        df_prob_xy = df_prob_xy.append(new_row, ignore_index=True)
    index += 1

df_prob = df_prob_xy.pivot_table(
    index="index", columns="prob", values="valor", aggfunc="sum"
)
df_prob = df_prob.reset_index()
df_prob = df_prob.set_index("index")

df_prob = df_prob.rename(columns={
    0.0: 'cl_o',
    1.0: 'cl_1',
    2.0: 'cl_2',
    3.0: 'cl_3',
    4.0: 'cl_4'
})

tabela = pd.merge(tabela, df_prob, left_index=True, right_index=True)
st.table(tabela)

rodadas = bra['rodada'].unique()
times = bra['mandante'].unique()
pontuacao = pd.DataFrame()
for rodada in rodadas:
    for time in times:
        resultado = bra[(bra['rodada'] == rodada) & ((bra['mandante'] == time) | (bra['visitante'] == time))]
        if len(resultado) > 0:
            resultado = resultado.reset_index()
            if resultado['mandante'][0] == time:
                if resultado['mplacar'][0] > resultado['vplacar'][0]:
                    pontos = 3
                elif resultado['mplacar'][0] == resultado['vplacar'][0]:
                    pontos = 1
                else:
                    pontos = 0
            else:                  
                if resultado['vplacar'][0] > resultado['mplacar'][0]:
                    pontos = 3
                elif resultado['vplacar'][0] == resultado['mplacar'][0]:
                    pontos = 1
                else:
                    pontos = 0
            new_row = {
                'time': time,
                'rodada': rodada,
                'pontos': pontos
            }        
            pontuacao = pontuacao.append(
                new_row,
                ignore_index = True
            )
pt_pontuacao = pontuacao.pivot_table(index='rodada', columns='time', values='pontos', aggfunc='sum')
pt_pontuacao_cum = pt_pontuacao.cumsum()
st.table(pt_pontuacao_cum)

pt_pontuacao_cum = pt_pontuacao_cum.reset_index()

colunas = pt_pontuacao_cum.columns
df_regressao = pd.DataFrame()
for coluna in colunas:
    if coluna != 'rodada':
        X = pt_pontuacao_cum['rodada'].values.reshape(-1, 1)
        y = pt_pontuacao_cum[coluna].values.reshape(-1, 1)
        regressor = LinearRegression()
        regressor.fit(X, y)
        A = regressor.intercept_[0]
        B = regressor.coef_[0][0]
        x = A + (B * 38)
        new_row = {
            'time': coluna,
            'intercept': round(A,2),
            'slope': round(B,2),
            'pontuacao_final': round(x,2)
        }
        df_regressao = df_regressao.append(
            new_row,
            ignore_index=True
        )
st.table(df_regressao)        
