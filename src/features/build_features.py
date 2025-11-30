import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def criar_features_geo_fisicas(df_base: pd.DataFrame, regua_fisica: LinearRegression = None) -> pd.DataFrame:
    """
    Aplica a lógica PIML, Geo-Física e Temporal para criar o conjunto final de features.
    
    Args:
        df_base: DataFrame com dados de consumo e metadados.
        regua_fisica: Modelo LinearRegression treinado em navios limpos (baseline físico).
    """
    df = df_base.copy()
    df['LOF'] = pd.to_numeric(df['LOF'], errors='coerce')

    # A. FILTRO DE VELOCIDADE (Remove ruído de portos)
    df = df[df['speed'] >= 5.0].copy()

    # B. FEATURE GEOGRÁFICA: RISCO REGIONAL (Lógica Hard-coded)
    def get_region_risk(row):
        lat = row.get('decLatitude', 0)
        lon = row.get('decLongitude', 0)
        
        # Sua lógica de mapeamento de risco
        if (-18 <= lat <= 5) and (-50 <= lon <= -34): return 5.0 # BR NE (Hotspot)
        if (-25 <= lat < -18) and (-55 <= lon <= -39): return 4.0 # BR SE
        
        abs_lat = abs(lat)
        if abs_lat <= 23.5: return 5.0      # Trópicos
        elif 23.5 < abs_lat <= 40: return 3.0 # Subtropical
        else: return 1.0                      # Polar

    df['RISCO_REGIONAL'] = df.apply(get_region_risk, axis=1)

    # C. FEATURE QUÍMICA: FATOR DE ENERGIA (Conversão para MJ)
    mapa_energia = {'LSHFO': 40.5, 'MGO': 42.7, 'VLSFO': 41.2}
    def get_densidade(tipo):
        for k, v in mapa_energia.items():
            if str(k) in str(tipo): return v
        return 41.0 # Default seguro
    
    df['fator_energia'] = df['TIPO_COMBUSTIVEL_PRINCIPAL'].apply(get_densidade)
    df['ENERGIA_CALCULADA_MJ'] = df['MASSA_TOTAL_TON'] * 1000 * df['fator_energia']
    
    # D. Variáveis PIML para Predição
    df['velocidade_cubo'] = df['speed'] ** 3

    # E. FEATURE TEMPORAL: Dias desde a limpeza
    df['DiasDesdeUltimaLimpeza'] = (df['endGMTDate'] - df['Data_Ultima_Limpeza']).dt.days
    df['DiasDesdeUltimaLimpeza'] = df['DiasDesdeUltimaLimpeza'].fillna(365 * 2) # Imputação segura para nulos
    df.loc[df['DiasDesdeUltimaLimpeza'] < 0, 'DiasDesdeUltimaLimpeza'] = 0

    # F. O SINAL PURO PIML: PERFORMANCE_LOSS_MJ
    if regua_fisica is not None:
        # A régua usa (velocidade_cubo * duration) como input
        X_input = (df['velocidade_cubo'] * df['duration']).values.reshape(-1,1)
        df['ENERGIA_ESPERADA'] = regua_fisica.predict(X_input)
        df['PERFORMANCE_LOSS_MJ'] = df['ENERGIA_CALCULADA_MJ'] - df['ENERGIA_ESPERADA']
    else:
        # Cria a coluna, mas preenche com NaN se o modelo físico não for fornecido
        df['PERFORMANCE_LOSS_MJ'] = np.nan

    return df
