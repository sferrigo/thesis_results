#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_incendio_plotly_v3.py

Versão interativa usando Plotly do script de detecção de fogo
(baseado em Dempster–Shafer + checagem de umidade), com legendas
para as linhas verticais pontilhadas:
- Temperatura em vermelho
- Linha pontilhada cinza = Detecção refinada
- Linha pontilhada preta = Alarme benchmark (14:26:50)
- Linha pontilhada azul = Início real do fogo (14:22:10)
"""

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# === 1. Leitura e pré-processamento ===
df = pd.read_csv(
    './teste_incendio_bg.csv',
    usecols=['variable', 'value', 'time'],
    parse_dates=['time']
)
df = df[df['variable'].isin(['temperature', 'humidity'])]
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df.dropna(subset=['value'])
df_wide = df.pivot_table(
    index='time', columns='variable', values='value', aggfunc='mean'
).dropna()
df_wide.index = df_wide.index - pd.Timedelta(hours=3)

times = df_wide.index.to_numpy()
temps = df_wide['temperature'].to_numpy()
hums  = df_wide['humidity'].to_numpy()

# === 2. Parâmetros Dempster–Shafer ===
WT           = 35
tr_threshold = 1.01
m_threshold  = 0.6

pts_temp = np.array([43,44,45,46,47,48,49,50,51,52,53,54], dtype=float)
pts_hum  = np.array([23,22,20,18,16,15,14,14,14,14,14,14], dtype=float)

def mass_lagrange(temp, hum):
    n = pts_temp.size
    m = 0.0
    for j in range(n):
        Lj = 1.0
        for k in range(n):
            if k != j:
                Lj *= (temp - pts_temp[k])/(pts_temp[j] - pts_temp[k])
        m += pts_hum[j] * Lj
    hum_exp = m
    return np.clip((hum_exp - hum)/hum_exp, 0, 1)

def mass_newton(temp, hum):
    n  = pts_temp.size
    dd = np.zeros((n,n))
    dd[:,0] = pts_hum
    for j in range(1,n):
        for i in range(n-j):
            dd[i,j] = (dd[i+1,j-1] - dd[i,j-1])/(pts_temp[i+j] - pts_temp[i])
    result = dd[0,0]
    prod   = 1.0
    for j in range(1,n):
        prod   *= (temp - pts_temp[j-1])
        result += dd[0,j]*prod
    hum_exp = result
    return np.clip((hum_exp - hum)/hum_exp, 0, 1)

def combine_ds(m1, m2):
    k = m1*(1-m2) + m2*(1-m1)
    if k == 1:
        return 0.0
    num = m1*m2 + m1*(1-m2) + (1-m1)*m2
    return num/(1-k)

# === 3. Detecta fogo refinado ===
detection_time = None
for i in range(WT, len(times)):
    if hums[i] >= hums[i-1]:
        continue
    if temps[i]/temps[i-WT:i].mean() <= tr_threshold:
        continue
    m1 = mass_lagrange(temps[i], hums[i])
    m2 = mass_newton(temps[i], hums[i])
    if (m1 >= m_threshold and m2 >= m_threshold) or \
       ((m1 >= m_threshold) ^ (m2 >= m_threshold) and combine_ds(m1,m2) >= m_threshold):
        detection_time = times[i]
        break

# === 4. Horários fixos ===
base_date       = df_wide.index[0].date().isoformat()
benchmark_time  = pd.to_datetime(f"{base_date} 14:26:50")
start_fire_time = pd.to_datetime(f"{base_date} 14:22:10")

# determinar limites de temperatura para vertical lines
y_min, y_max = temps.min(), temps.max()

# === 5. Plotly ===
fig = make_subplots(specs=[[{"secondary_y": True}]],
                    x_title="Timestamp (UTC-3)",
                    subplot_titles=["Simulação de detecção de Incêndio"])

# Temperatura (vermelho)
fig.add_trace(
    go.Scatter(x=times, y=temps, name="Temperatura (°C)",
               line=dict(color="red")),
    secondary_y=False
)
# Umidade
fig.add_trace(
    go.Scatter(x=times, y=hums, name="Umidade (%)"),
    secondary_y=True
)

# Linhas verticais pontilhadas com legendas
if detection_time is not None:
    fig.add_trace(
        go.Scatter(
            x=[detection_time, detection_time],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Alarme via temperatura e umidade"
        ),
        secondary_y=False
    )

fig.add_trace(
    go.Scatter(
        x=[benchmark_time, benchmark_time],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Alarme benchmark"
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=[start_fire_time, start_fire_time],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name="Início do incêndio"
    ),
    secondary_y=False
)

# Eixos e layout final
fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)
fig.update_yaxes(title_text="Umidade (%)", secondary_y=True)
fig.update_layout(
    legend=dict(x=1.03, y=1),
    margin=dict(l=50, r=150, t=80, b=50),
    height=500
)

fig.show()
