#!/usr/bin/env python3
import sys
import os
import ijson
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Aliases para as variáveis de interesse no JSON
HUMIDITY_ALIASES = ["humidity", "relative_humidity_3"]
RATIO_ALIASES   = ["pressao_dividida_pelo_orvalho"]

def parse_time(ts: str) -> datetime:
    """Converte timestamp ISO para datetime local."""
    return datetime.fromisoformat(ts)

def stream_values(filename: str, aliases: list):
    """
    Faz streaming do JSON e retorna duas listas (times, values)
    apenas para as aliases indicadas, ordenadas cronologicamente.
    """
    times, values = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, 'item'):
            if obj.get('variable') in aliases:
                try:
                    t = parse_time(obj['time'])
                    v = float(obj['value'])
                except Exception:
                    continue
                times.append(t)
                values.append(v)
    paired = sorted(zip(times, values), key=lambda x: x[0])
    if not paired:
        return [], []
    t_sorted, v_sorted = zip(*paired)
    return list(t_sorted), list(v_sorted)

def load_rain_csv(csv_file: str):
    """
    Carrega o CSV de chuva no novo formato CEMADEN e retorna listas ordenadas.
    Considera apenas estação 'G2-Santa Fé', converte 'datahora' UTC para horário local (UTC-3),
    e lê chuva em 'valorMedida'.
    """
    df = pd.read_csv(csv_file, sep=';', decimal=',', encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['nomeestacao'].str.strip().str.lower() == 'g2-santa fé']
    df['datetime'] = pd.to_datetime(df['datahora'], utc=True, errors='coerce') - pd.Timedelta(hours=3)
    df['rain'] = pd.to_numeric(df['valormedida'], errors='coerce')
    df = df[['datetime', 'rain']].dropna(subset=['datetime', 'rain'])
    df.sort_values('datetime', inplace=True)
    return list(df['datetime']), list(df['rain'])

def plot_with_rain(json_files: list, csv_file: str, output_dir: str):
    """
    Gera gráficos combinando humidity, ratio e chuva (barra).
    Mantém opções interativas para mostrar/ocultar linhas de cruzamento e linhas de grade.
    Limita o eixo principal entre 0 e 150.
    """
    os.makedirs(output_dir, exist_ok=True)
    rain_times, rain_vals = load_rain_csv(csv_file)

    for path in json_files:
        label = os.path.splitext(os.path.basename(path))[0]
        times_h, vals_h = stream_values(path, HUMIDITY_ALIASES)
        times_r, vals_r = stream_values(path, RATIO_ALIASES)
        if not times_h or not times_r:
            print(f"Atenção: dados insuficientes em {label}, pulando.")
            continue
        df_h = pd.DataFrame({"time": times_h, "humidity": vals_h})
        df_r = pd.DataFrame({"time": times_r, "ratio": vals_r})
        df = pd.merge(df_h, df_r, on="time", how="outer").sort_values("time")
        df.set_index("time", inplace=True)
        df = df.interpolate(method='time').dropna(subset=["humidity","ratio"])

        diff = df["humidity"] - df["ratio"]
        signs = np.sign(diff)
        crossings = []
        for i in range(len(signs) - 1):
            if signs[i] * signs[i+1] < 0:
                t0, t1 = diff.index[i], diff.index[i+1]
                y0, y1 = diff.iloc[i], diff.iloc[i+1]
                frac = -y0 / (y1 - y0)
                crossings.append(t0 + (t1 - t0) * frac)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["humidity"], mode='lines+markers',
            name='Umidade (%)', marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ratio"], mode='lines+markers',
            name='Pressão/Orvalho (mbar/ºC)', marker=dict(size=4)
        ))
        fig.add_trace(go.Bar(
            x=rain_times, y=rain_vals, name='Chuva (mm)',
            yaxis='y2', opacity=0.6, marker_color='black'
        ))

        # trace interativa para linhas de cruzamento
        if crossings:
            vertical_x, vertical_y = [], []
            for ct in crossings:
                vertical_x += [ct, ct, None]
                vertical_y += [0, 150, None]
            fig.add_trace(go.Scatter(
                x=vertical_x, y=vertical_y, mode='lines',
                name='Cruzamentos', line=dict(dash='dash', color='gray')
            ))

        # grades
        fig.update_yaxes(showgrid=True)
        fig.update_xaxes(showgrid=True)

        fig.update_layout(
            title=f"CHRUDA — {label}",
            xaxis=dict(title="Data", tickformat="%d/%m/%Y %H:%M"),
            yaxis=dict(title="Umidade/Pressão/Orvalho", range=[0,150]),
            yaxis2=dict(title="Chuva (mm)", overlaying='y', side='right', showgrid=False)
        )

        out_path = os.path.join(output_dir, f"{label}_combined.html")
        fig.write_html(out_path, include_plotlyjs='cdn', config={'locale':'pt-BR'})
        print(f"✔ Gráfico com chuva salvo em: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python script.py <arquivo1.json> [...] <chuva.csv> <pasta_saida>")
        sys.exit(1)
    args = sys.argv[1:]
    json_files = [p for p in args if p.lower().endswith('.json')]
    csv_files = [p for p in args if p.lower().endswith('.csv')]
    others = [p for p in args if not p.lower().endswith(('.json','.csv'))]
    out_dir = others[-1] if others else 'plots_combined'
    chuva_csv = csv_files[0]
    plot_with_rain(json_files, chuva_csv, out_dir)
