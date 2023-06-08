
import warnings
warnings.filterwarnings('ignore')

import pathlib
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool

DATA_DIR = pathlib.Path(".")

# Lasso
MODEL_FILE_target01 = pathlib.Path(__file__).parent.joinpath("model_target0_for_target_1.pkl")
MODEL_FILE_target10 = pathlib.Path(__file__).parent.joinpath("model_target1_for_target_0.pkl")

# Ridge
MODEL_FILE_target00_ridge = pathlib.Path(__file__).parent.joinpath("model_target0_for_target_0_big.pkl")
MODEL_FILE_target10_ridge = pathlib.Path(__file__).parent.joinpath("model_target1_for_target_0_ss_ridge_001.pkl")

# SVR
MODEL_FILE_target01_svr = pathlib.Path(__file__).parent.joinpath("model_target0_for_target_1_svr_d4_C3.pkl")
MODEL_FILE_target11_svr = pathlib.Path(__file__).parent.joinpath("model_target1_for_target_1_svr_C_3.pkl")

# Square
MODEL_FILE_target11_square = pathlib.Path(__file__).parent.joinpath("model_target1_for_target_1_lasso_squares.pkl")


models_dict = {}
models_dict['target0'] = {}
models_dict['target1'] = {}

models_dict['target0'][0] = {}
models_dict['target0'][0]['ridge'] = MODEL_FILE_target00_ridge

models_dict['target0'][1] = {}
models_dict['target0'][1]['lasso'] = MODEL_FILE_target01
models_dict['target0'][1]['svr'] = MODEL_FILE_target01_svr

models_dict['target1'][0] = {}
models_dict['target1'][0]['lasso'] = MODEL_FILE_target10
models_dict['target1'][0]['ridge'] = MODEL_FILE_target10_ridge

models_dict['target1'][1] = {}
models_dict['target1'][1]['svr'] = MODEL_FILE_target11_svr
models_dict['target1'][1]['square'] = MODEL_FILE_target11_square

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    
    df['target_class'] = (df['feature4'] == 'gas1') * 1
    df.drop(labels=['feature4'], axis=1, inplace=True)
    key_cols = ['target0', 'target1', 'target_class']
    cols_for_features = ['feature' + str(i) for i in range(25) if i != 4]
    
    predictions = df[['target_class']]
    
    
    current_target = 'target0'
    current_target_class = 0
    current_test = df[df['target_class'] == current_target_class]
    model_ridge = pickle.load(open(models_dict[current_target][current_target_class]['ridge'], 'rb'))
    predictions.loc[current_test.index, current_target] = model_ridge.predict(current_test[cols_for_features])
    
    
    current_target = 'target0'
    current_target_class = 1
    current_test = df[df['target_class'] == current_target_class]
    model_lasso = pickle.load(open(models_dict[current_target][current_target_class]['lasso'], 'rb'))
    model_svr = pickle.load(open(models_dict[current_target][current_target_class]['svr'], 'rb'))
    predictions.loc[current_test.index, current_target] =\
                            (0.5 * model_lasso.predict(current_test[cols_for_features]) +\
                             0.5 * model_svr.predict(current_test[cols_for_features]))
    
    
    current_target = 'target1'
    current_target_class = 0
    current_test = df[df['target_class'] == current_target_class]
    model_lasso = pickle.load(open(models_dict[current_target][current_target_class]['lasso'], 'rb'))
    model_ridge = pickle.load(open(models_dict[current_target][current_target_class]['ridge'], 'rb'))
    predictions.loc[current_test.index, current_target] =\
                            (0.5 * model_lasso.predict(current_test[cols_for_features]) +\
                             0.5 * model_ridge.predict(current_test[cols_for_features]))
    
    
    current_target = 'target1'
    current_target_class = 1
    current_test = df.loc[df['target_class'] == current_target_class]
    model_svr = pickle.load(open(models_dict[current_target][current_target_class]['svr'], 'rb'))
    
    model_square = pickle.load(open(models_dict[current_target][current_target_class]['square'], 'rb'))
    test11 = df.loc[df['target_class'] == current_target_class, cols_for_features]
    for i in range(len(cols_for_features)):
        test11[cols_for_features[i] + '_2'] = test11[cols_for_features[i]] * test11[cols_for_features[i]]
        
    predictions.loc[current_test.index, current_target] =\
                            (0.5 * model_square.predict(test11) +\
                             0.5 * model_svr.predict(current_test[cols_for_features]))

    
    predictions = predictions[['target0', 'target1']]
    predictions['target0'] = np.clip(predictions['target0'], 0, 100)
    predictions['target1'] = np.clip(predictions['target1'], 0, 100)
    
    return predictions

def predict(df: pd.DataFrame) -> pd.DataFrame:
    pool = Pool(4)
    prediction_table = pool.map(predict_batch,
                    [df.iloc[i * 5_000: (i + 1) * 5_000] for i in range(df.shape[0] // 5_000 + 1)])
    return pd.concat(prediction_table).sort_index()
