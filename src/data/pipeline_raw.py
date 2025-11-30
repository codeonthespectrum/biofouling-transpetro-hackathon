import pandas as pd

def gerar_dataset_avancado(df_eventos, df_metadata, df_consumo, df_relatorios, janela_dias=200):
    """
    Executa a Estratégia Sanduíche (Merge Temporal) para anexar a última limpeza 
    (Backward) e o próximo LOF (Forward) a cada registro de viagem.
    """
    # Limpeza básica e padronização de nomes (Case Insensitive)
    cols_navio = {'ev': 'shipName', 'meta': 'Nome do navio', 'rel': 'Embarcação'}
    df_eventos[cols_navio['ev']] = df_eventos[cols_navio['ev']].astype(str).str.lower().str.strip()
    df_metadata[cols_navio['meta']] = df_metadata[cols_navio['meta']].astype(str).str.lower().str.strip()
    df_relatorios[cols_navio['rel']] = df_relatorios[cols_navio['rel']].astype(str).str.lower().str.strip()

    # Conversão de Datas
    df_eventos['endGMTDate'] = pd.to_datetime(df_eventos['endGMTDate'], errors='coerce')
    df_relatorios['Data'] = pd.to_datetime(df_relatorios['Data'], errors='coerce')

    # Merges Iniciais (Metadados + Consumo)
    df_step1 = pd.merge(df_eventos, df_metadata[['Nome do navio', 'Classe', 'Area_Molhada']],
                        left_on='shipName', right_on='Nome do navio', how='left')
    
    cols_consumo = ['SESSION_ID', 'MASSA_TOTAL_TON', 'TIPO_COMBUSTIVEL_PRINCIPAL']
    df_step1['sessionId'] = df_step1['sessionId'].astype('int64')
    df_consumo['SESSION_ID'] = df_consumo['SESSION_ID'].astype('int64')
    
    df_step2 = pd.merge(df_step1, df_consumo[cols_consumo],
                        left_on='sessionId', right_on='SESSION_ID', how='left')

    df_step2.drop(columns=['Nome do navio', 'SESSION_ID'], inplace=True, errors='ignore')

    # MERGE BACKWARD (Data da última limpeza)
    df_past = pd.merge_asof(
        df_step2.sort_values('endGMTDate'),
        df_relatorios[['Embarcação', 'Data']].rename(columns={'Data': 'Data_Ultima_Limpeza'}),
        left_on='endGMTDate',
        right_on='Data_Ultima_Limpeza',
        left_by='shipName',
        right_by='Embarcação',
        direction='backward' 
    )

    # MERGE FORWARD (LOF no futuro - o Target)
    df_future = pd.merge_asof(
        df_step2.sort_values('endGMTDate'),
        df_relatorios[['Embarcação', 'Data', 'LOF_target']],
        left_on='endGMTDate',
        right_on='Data',
        left_by='shipName',
        right_by='Embarcação',
        direction='forward',
        tolerance=pd.Timedelta(days=janela_dias)
    )

    # Finalização
    df_final = df_past.copy()
    df_final['LOF'] = df_future['LOF_target']
    df_final = df_final.dropna(subset=['LOF']).copy()
    
    return df_final
