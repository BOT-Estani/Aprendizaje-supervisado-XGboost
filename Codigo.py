import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import xgboost as xgb

# Definir la carpeta de origen y destino
carpeta_origen = "Codigo/Entrega/raw data" 
carpeta_destino = "Codigo/Entrega/filter"

# Definir las columnas a eliminar
columnas_a_borrar = ["device_id", "creative_categorical_10","auction_boolean_1", "auction_boolean_2", "auction_categorical_2", "auction_categorical_3",
                        "auction_categorical_10", "auction_categorical_8", "auction_categorical_9", "auction_categorical_7", "action_categorical_2", "auction_age", "timezone_offset",
                        "action_list_1","action_list_2", "auction_list_0"] 

# Función para extraer la hora desde epoch
def extraer_hora(epoch_time):
    return datetime.utcfromtimestamp(epoch_time).hour

# Obtener la lista de archivos CSV en la carpeta de origen
archivos_csv = [f for f in os.listdir(carpeta_origen) if f.endswith('.csv')]

# Iterar sobre cada archivo CSV
for archivo in archivos_csv:
    # Leer el archivo CSV
    ruta_archivo = os.path.join(carpeta_origen, archivo)
    datos = pd.read_csv(ruta_archivo)
    
    # Eliminar las columnas deseadas
    datos_modificado = datos.drop(columns=columnas_a_borrar, errors='ignore')
    datos_modificado["pixel_square"] = datos_modificado["creative_height"] * datos_modificado["creative_width"]
    datos_modificado["price_per_pixel"] = datos_modificado["pixel_square"] / datos_modificado["auction_bidfloor"].replace(0, np.nan)
    datos_modificado["creative_height_square"] = datos_modificado["creative_height"] * datos_modificado["creative_height"]

    # Si existe la columna "auction_time", convertirla a la hora
    if 'auction_time' in datos_modificado.columns:
        datos_modificado['auction_time'] = datos_modificado['auction_time'].apply(extraer_hora)
    
    # Generar el nombre del archivo en la carpeta de destino
    ruta_destino = os.path.join(carpeta_destino, archivo)
    
    # Guardar el archivo modificado en la carpeta de destino
    datos_modificado.to_csv(ruta_destino, index=False)

datos = pd.read_csv("Codigo/Entrega/test/test.csv")
ruta_mod = "Codigo/Entrega/test/test.csv"       
           
datos_modificado = datos.drop(columns=columnas_a_borrar, errors='ignore')
datos_modificado["pixel_square"] = datos_modificado["creative_height"] * datos_modificado["creative_width"]
datos_modificado["price_per_pixel"] = datos_modificado["pixel_square"] / datos_modificado["auction_bidfloor"].replace(0, np.nan)
datos_modificado["creative_height_square"] = datos_modificado["creative_height"] * datos_modificado["creative_height"]

# Si existe la columna "auction_time", convertirla a la hora
if 'auction_time' in datos_modificado.columns:
    datos_modificado['auction_time'] = datos_modificado['auction_time'].apply(lambda x: datetime.utcfromtimestamp(x).hour)

# Generar el nombre del archivo en la carpeta de destino
ruta_destino = os.path.join(ruta_mod, "filtrado one-hot.csv")
test = datos_modificado
# Guardar el archivo modificado en la carpeta de destino, comentado para ahorrar tiempo de ejecución
#datos_modificado.to_csv(ruta_destino, index=False)

print(f"Archivo modificado guardado en: {ruta_destino}")

#Definir la carpeta de origen y el nombre del archivo combinado
carpeta_origen = "Codigo/Entrega/filter"
archivo_destino = "Codigo/Entrega/filter/union filter.csv"

# Obtener la lista de archivos CSV en la carpeta de origen
archivos_csv = [f for f in os.listdir(carpeta_origen) if f.endswith('.csv')]

# Crear una lista para almacenar los DataFrames
dataframes = []

# Iterar sobre cada archivo CSV y agregarlo a la lista
for archivo in archivos_csv:
    ruta_archivo = os.path.join(carpeta_origen, archivo)
    datos = pd.read_csv(ruta_archivo)
    dataframes.append(datos)
    os.remove(ruta_archivo)

# Combinar todos los DataFrames en uno solo
union_df = pd.concat(dataframes, ignore_index=True)
df1 = union_df[union_df['Label'] == 1]
df2 = union_df[union_df['Label'] == 0]

# Graficar la proporción de apariciones
plt.figure(figsize=(10, 6))

plt.hist(df1['auction_bidfloor'], bins=50, alpha=0.6, label='Label 1', color='blue', density=True)
plt.hist(df2['auction_bidfloor'], bins=50, alpha=0.6, label='Label 0', color='orange', density=True)

plt.xlim(0, 11)
plt.title('Proporción de auction_bidfloor')
plt.xlabel('Proporción')
plt.ylabel('Frecuencia (normalizada)')
plt.legend()
plt.show()

columnas_listas = ['action_list_0']
    
for col in columnas_listas:
    if col in df1.columns and col in df2.columns:
        # Asumimos que los valores en estas columnas son strings o listas de strings
        all_items1 = []
        all_items2 = []
        
        # Recorrer cada fila de la columna y separar los elementos de las listas
        for row in df1[col].dropna():
            items = str(row).split(',')
            cleaned_items = [item.strip('[]" ') for item in items]
            all_items1.extend(cleaned_items)
        for row in df2[col].dropna():
            items = str(row).split(',')
            cleaned_items = [item.strip('[]" ') for item in items]
            all_items2.extend(cleaned_items)
        
        # Contar las frecuencias de cada elemento
        item_counts1 = Counter(all_items1)
        item_counts2 = Counter(all_items2)
        
        # Generar los gráficos de torta
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].pie(item_counts1.values(), labels=item_counts1.keys(), autopct='%1.1f%%', startangle=140)
        axes[0].set_title(f"Proporción de elementos en {col} (Clicked)")
        axes[1].pie(item_counts2.values(), labels=item_counts2.keys(), autopct='%1.1f%%', startangle=140)
        axes[1].set_title(f"Proporción de elementos en {col} (Not)")
        
        # Guardar el gráfico como una única imagen
        plt.close()


# Guardar el DataFrame combinado en un archivo CSV
union_df.to_csv(archivo_destino, index=False)

print(f"Archivos combinados en: {archivo_destino}")

# archivo = "Codigo/Entrega/filter/union filter.csv"
# datos = pd.read_csv(archivo)
print("leido el filter")

print("separo")
train, val = train_test_split(union_df, test_size=0.15, random_state=42)

# Comentamos este codigo para ahorrar tiempo de ejecución

# train.to_csv("Codigo/Entrega/trainers/Train.csv", index=False)
# val.to_csv("Codigo/Entrega/trainers/Val.csv", index=False)

print("Conjuntos creados y guardados: Train.csv, Val.csv")


def calculate_category_ratios(df, cat_column):
    label_counts = df.groupby([cat_column, 'Label']).size().unstack(fill_value=0)
    top_categories_1 = label_counts[1].nlargest(20).index
    top_categories_0 = label_counts[0].nlargest(20).index
    top_categories = top_categories_1.intersection(top_categories_0)
    label_counts = label_counts.loc[top_categories]
    total_1 = label_counts[1].sum()
    total_0 = label_counts[0].sum()
    prop_1 = (label_counts[1] / total_1) if total_1 != 0 else 0
    prop_0 = (label_counts[0] / total_0) if total_0 != 0 else 0
    if total_1 == 0:
        prop_1 = prop_0 / 2 if prop_0 > 0.01 else prop_0
    if total_0 == 0:
        prop_0 = prop_1 / 2 if prop_1 > 0.01 else prop_1
    label_counts['ratio'] =  prop_1 / prop_0
    label_counts['one_hot'] = np.where((label_counts['ratio'] < 0.95) | (label_counts['ratio'] > 1.05), True, False)
    return label_counts

def apply_one_hot_encoding(df, columnas_excluidas, val, test):
    i = 1
    for col in df.columns:
        if col not in columnas_excluidas and col != 'Label': 
            category_info = calculate_category_ratios(df, col)
            print(f"Processing column: {col}, original shape: {df.shape}")
            one_hot_categories = category_info[category_info['one_hot']].index.tolist()
            other = 'other_' + col
            df[col] = df[col].apply(lambda x: x if x in one_hot_categories else other)
            val[col] = val[col].apply(lambda x: x if x in one_hot_categories else other)
            test[col] = test[col].apply(lambda x: x if x in one_hot_categories else other)
            print(i)
            i += 1
            if col in columnas_excluidas and col != 'Label':
                med = df[col].median(skipna=True)
                df[col] = df[col].fillna(med) - med
                val[col] = val[col].fillna(med) - med
                test[col] = val[col].fillna(med) - med
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    categorical_columns = [col for col in df.columns if col not in columnas_excluidas and col != 'Label']
    for col in df.columns:
        if col in categorical_columns:
            df[col] = df[col].astype('string')
    print("fit")
    encoder.fit(df[categorical_columns])
    print("train")
    encoded_train = pd.DataFrame.sparse.from_spmatrix(encoder.transform(df[categorical_columns]), columns=encoder.get_feature_names_out())
    print("val")
    encoded_val = pd.DataFrame.sparse.from_spmatrix(encoder.transform(val[categorical_columns]), columns=encoder.get_feature_names_out())
    print("test")
    encoded_test = pd.DataFrame.sparse.from_spmatrix(encoder.transform(test[categorical_columns]), columns=encoder.get_feature_names_out())
    encoded_train = pd.concat([df.drop(columns=categorical_columns).reset_index(drop=True), 
                    encoded_train.reset_index(drop=True)], axis=1)
    print("val")
    encoded_val = pd.concat([val.drop(columns=categorical_columns).reset_index(drop=True), 
                            encoded_val.reset_index(drop=True)], axis=1)
    print("test")
    encoded_test = pd.concat([test.drop(columns=categorical_columns).reset_index(drop=True), 
                            encoded_test.reset_index(drop=True)], axis=1)


    return encoded_train, encoded_val, encoded_test

# Comentado para ahorrar  tiempo de ejecución
# print("leo val")
# val = pd.read_csv("C:/Users/estan/OneDrive/Escritorio/Cositas/facultad/TD VI/TP 2 - EL BOSQUE/DATA/Trainers/Basicos/Val.csv") 
# print("leo test")
# test= pd.read_csv("Codigo/Entrega/test/test.csv") 

print("Por leer train...")
archivo = "Codigo/Entrega/trainers/Train.csv" # El dataset sin as columnas que elegimos
train = pd.read_csv(archivo)
print("Leido!")


train0 = train[train['Label'] == 0].sample(frac=0.6, random_state=42)  # 60% de la clase 0
train1 = train[train['Label'] == 1]  # Todos los datos de la clase 1
train = pd.concat([train0, train1], axis=0) 

print("unido!")
columnas_excluidas = ["Label",'auction_bidfloor', "creative_height", "creative_width", 
                      "has_video", "creative_height_square", "price_per_pixel", "pixel_square"]  
print("por encodear")

df_encoded, val, test = apply_one_hot_encoding(train, columnas_excluidas, val, test)
print("listooooooo")

X_train = df_encoded.drop(columns=["Label", "creative_height"])
y_train = df_encoded["Label"]

X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

x_val = val.drop(columns=["Label", "creative_height"])
y_val = val["Label"]

x_val = x_val.fillna(0)
y_val = y_val.fillna(0)

x_test = test.drop(columns=["creative_height", "id"])
x_test = x_test.fillna(0)

X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
x_val.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in x_val.columns]
x_test.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in x_test.columns]

X_train_sparse = sparse.csr_matrix(X_train.values)
x_val_sparse = sparse.csr_matrix(x_val.values)
x_test_sparse = sparse.csr_matrix(x_test)


dtrain = xgb.DMatrix(X_train_sparse, label=y_train)
dval = xgb.DMatrix(x_val_sparse, label=y_val)
dtest = xgb.DMatrix(x_test_sparse)

print("train")
print(X_train.dtypes)
print("val")
print(x_val.dtypes)
print("test")
print(x_test.dtypes)



print("entrenamiento")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'colsample_bytree': 0.66666,
    'gamma': 0.6,
    'learning_rate': 0.08,
    'max_depth': 8,
    'min_child_weight': 1,
    'n_estimators': 400,
    'reg_lambda': 0.5,
    'subsample': 0.666666,
    'tree_method': 'hist'  
}

clf_xgb = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=400,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10
)



preds_test_xgb = clf_xgb.predict(dtest)

avg_pred_df = pd.DataFrame({
    "id": test["id"],  
    "pred": preds_test_xgb  
})


avg_pred_df.to_csv("Codigo/Entrega/predicciones/oh_freq_xgb.csv", index=False)

