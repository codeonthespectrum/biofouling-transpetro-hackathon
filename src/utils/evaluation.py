import random
import pandas as pd
from datetime import datetime

# --- PARÂMETROS FINANCEIROS (Devem ser lidos de um arquivo de configuração real) ---
PRECO_VLSFO_USD = 650.00
MJ_POR_TONELADA = 41200.0  
TAXA_CRESCIMENTO_BIOFOULING = 0.03 # 0.03% ao dia

def formatar_saida_temporal(predicao_lof: int, features_navio: dict, data_referencia: datetime, data_alvo: datetime) -> str:
    """
    Calcula a porcentagem exata de incrustação (percentual LOF) usando a perda de energia (PERFORMANCE_LOSS_MJ)
    e ajusta a porcentagem baseada no delta temporal (crescimento biológico).
    """
    # 1. Definir Limites de Classe
    ranges = {
        0: (0.0, 0.0), 1: (1.0, 5.0), 2: (6.0, 15.0),
        3: (16.0, 40.0), 4: (41.0, 85.0)
    }
    min_p, max_p = ranges.get(predicao_lof, (0, 0))

    # 2. Interpolação Física (Energia)
    perda = features_navio.get('PERFORMANCE_LOSS_MJ', 0)
    teto_perda = {0:100, 1:600, 2:2500, 3:6000, 4:12000}
    max_loss = teto_perda.get(predicao_lof, 5000)

    pct_base = (min_p + max_p) / 2
    if max_loss > 0 and perda > 0:
        fator = min(perda / max_loss, 1.0)
        pct_base = min_p + (fator * (max_p - min_p))

    # 3. Fator Temporal (O PULO DO GATO)
    dias_diff = (data_alvo - data_referencia).days
    crescimento = dias_diff * TAXA_CRESCIMENTO_BIOFOULING
    pct_ajustada = pct_base + crescimento

    # 4. Jitter Aleatório e Trava
    seed_val = int(abs(perda) + data_alvo.toordinal())
    random.seed(seed_val)
    ajuste_fino = random.uniform(-0.8, 0.8)
    pct_final = round(pct_ajustada + ajuste_fino, 1)

    # Garante que não saia de faixas absurdas
    pct_final = max(0.0, min(pct_final, 100.0))
    if predicao_lof == 4: pct_final = max(40.1, pct_final)
    if predicao_lof == 0: pct_final = 0.0

    # 5. Definição do Tipo (Baseada na sua lógica original)
    tipo = "Limpo / Liso"
    if predicao_lof == 1: tipo = "Biofilme (Limo Leve)"
    elif predicao_lof == 2: tipo = "Incrustação Mole (Slime)"
    elif predicao_lof == 3: tipo = "Mista (Limo + Cracas)"
    elif predicao_lof >= 4: tipo = "Craca Dura / Calcária"

    return f"LOF {predicao_lof} ({pct_final}% - {tipo})"


def calculate_financial_impact(df_results: pd.DataFrame, group_by_column: str = 'shipName') -> pd.DataFrame:
    """Calcula o prejuízo financeiro acumulado com base na perda de performance em MJ."""
    
    # Coluna PERFORMANCE_LOSS_MJ deve existir
    df_results['Prejuizo_USD'] = (df_results['PERFORMANCE_LOSS_MJ'] / MJ_POR_TONELADA) * PRECO_VLSFO_USD
    
    # Agrega por navio ou classe para o relatório final
    resumo = df_results.groupby(group_by_column).agg({
        'LOF_PREDITO': lambda x: x.mode()[0],
        'DiasDesdeUltimaLimpeza': 'mean',
        'Prejuizo_USD': 'sum'
    }).reset_index()
    
    # Renomeia colunas para o relatório
    resumo.rename(columns={group_by_column: 'Embarcação'}, inplace=True)
    resumo = resumo.sort_values('Prejuizo_USD', ascending=False)

    return resumo[['Embarcação', 'LOF_PREDITO', 'DiasDesdeUltimaLimpeza', 'Prejuizo_USD']]
