import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Importa o pipeline de features criado por você
from src.features.build_features import criar_features_geo_fisicas 

# --- VARIÁVEIS DE TREINAMENTO ---
FEATURES = [
    'speed', 'duration', 'distance', 'beaufortScale',
    'Area_Molhada', 'PERFORMANCE_LOSS_MJ',
    'DiasDesdeUltimaLimpeza', 'velocidade_cubo', 'RISCO_REGIONAL'
]

# --- 1. TREINAR A RÉGUA FÍSICA (PIML BASELINE) ---
def train_regua_fisica(df_raw):
    """Treina o modelo de Regressão Linear em navios "limpos" (LOF <= 1.5)."""
    df_calib = df_raw.copy()
    
    # Recria as colunas base necessárias para a calibração
    df_calib['LOF'] = pd.to_numeric(df_calib['LOF'], errors='coerce')
    df_calib['velocidade_cubo'] = df_calib['speed'] ** 3
    
    # Lógica de ENERGIA_CALCULADA_MJ (simplificada, mas precisa)
    mapa_energia = {'LSHFO': 40.5, 'MGO': 42.7, 'VLSFO': 41.2}
    def get_densidade(tipo):
        for k, v in mapa_energia.items():
            if str(k) in str(tipo): return v
        return 41.0 
    df_calib['fator_energia'] = df_calib['TIPO_COMBUSTIVEL_PRINCIPAL'].apply(get_densidade)
    df_calib['ENERGIA_CALCULADA_MJ'] = df_calib['MASSA_TOTAL_TON'] * 1000 * df_calib['fator_energia']

    # Filtrar Navios Limpos (LOF baixo) para ensinar o "Ideal"
    df_limpo = df_calib[df_calib['LOF'] <= 1.5].dropna(subset=['velocidade_cubo', 'duration', 'ENERGIA_CALCULADA_MJ'])
    
    # Input da Regressão (A potência física)
    X_ref = (df_limpo['velocidade_cubo'] * df_limpo['duration']).values.reshape(-1,1)
    y_ref = df_limpo['ENERGIA_CALCULADA_MJ'].values
    
    regua_fisica_oficial = LinearRegression()
    regua_fisica_oficial.fit(X_ref, y_ref)
    
    print("✅ Régua Física (PIML Baseline) treinada.")
    return regua_fisica_oficial

# --- 2. TREINAR O ENSEMBLE FINAL ---
def train_ensemble(df_final: pd.DataFrame, label_encoder: LabelEncoder):
    """Treina e avalia o Voting Classifier (RF + XGB)."""
    
    # Codificação do Target (Y)
    df_final['LOF_encoded'] = label_encoder.transform(df_final['LOF'])
    
    # Pré-processamento e Divisão
    df_model = df_final.dropna(subset=FEATURES + ['LOF_encoded']).copy()
    X = df_model[FEATURES].fillna(0)
    Y = df_model['LOF_encoded']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )
    
    # Modelos Base
    clf_rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    clf_xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, 
                            objective='multi:softprob', random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    # Ensemble (Voting)
    voting_final_v3 = VotingClassifier(
        estimators=[('rf', clf_rf), ('xgb', clf_xgb)],
        voting='soft',
        weights=[1, 2] 
    )

    voting_final_v3.fit(X_train, Y_train)
    Y_pred = voting_final_v3.predict(X_test)

    # Relatório de Performance
    print("\n--- Relatório de Classificação no Test Set ---")
    print(classification_report(Y_test, Y_pred, target_names=label_encoder.classes_.astype(str)))
    print(f"\n✅ Acurácia Geral: {accuracy_score(Y_test, Y_pred):.2f}")
    
    return voting_final_v3

# --- EXECUÇÃO PRINCIPAL ---
def run_training_pipeline(df_raw):
    """Pipeline completo: cria régua, aplica features PIML, treina e avalia."""
    
    # Simulação da etapa que gera o LabelEncoder (necessário para a métrica)
    le = LabelEncoder()
    le.fit(df_raw['LOF'])

    # 1. TREINAR RÉGUA FÍSICA
    regua_fisica = train_regua_fisica(df_raw)
    
    # 2. APLICAR FEATURES PIML EM TODO O DATASET
    # A régua (regua_fisica) é usada aqui para criar a feature PERFORMANCE_LOSS_MJ
    df_piml_features = criar_features_geo_fisicas(df_raw, regua_fisica)
    
    # 3. TREINAR ENSEMBLE
    voting_model = train_ensemble(df_piml_features, le)
    
    print("\n-------------------------------------------")
    print("Treinamento concluído com sucesso!")
    print("O modelo voting_final_v3 está pronto para uso.")
    print("-------------------------------------------")

# --- Bloco de execução ---
if __name__ == '__main__':
    df_metadata1 = pd.read_csv("/content/Metadata.csv")
    df_eventos1 = pd.read_csv("/content/eventos_final.csv")
    df_relatorios1 = pd.read_csv("/content/Relatorios_IWS_Processados_LOF.csv")
    df_consumo1 = pd.read_csv("/content/CONSUMO_LIMPO_PARA_MERGE.csv")
    # run_training_pipeline(df_raw)
    pass
