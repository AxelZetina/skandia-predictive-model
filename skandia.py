import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ----Código hecho por Axel Andrei Zetina Cuevas
# ----para prueba de cientifico de datos Jr Skandia

#  -----------------------------Cargando las bases de datos a dataframes---------------------------------------------------------------

# Rutas a los archivos .parquet (bases de datos)
ruta_archivo_clientes = r'C:\Users\zetin\OneDrive\Escritorio\Pruebas skandia\2da prueba cientifico de datos\0clientes.parquet'  # sustituir ruta donde se encuentra el doc. 0clientes.parquet
ruta_archivo_saldos = r'C:\Users\zetin\OneDrive\Escritorio\Pruebas skandia\2da prueba cientifico de datos\0saldos.parquet'  # sustituir ruta donde se encuentra el doc. 0saldos.parquet
ruta_archivo_trans = r'C:\Users\zetin\OneDrive\Escritorio\Pruebas skandia\2da prueba cientifico de datos\0transferencias.parquet'  # sustituir ruta donde se encuentra el doc. 0transferencias.parquet

# Leer los archivos .parquet
tabla_parquet_clientes= pq.read_table(ruta_archivo_clientes)
tabla_parquet_saldos= pq.read_table(ruta_archivo_saldos)
tabla_parquet_trans= pq.read_table(ruta_archivo_trans)

# Conviertir las tablas(bases) a un DataFrame
dataframe_clientes = tabla_parquet_clientes.to_pandas()
dataframe_saldos = tabla_parquet_saldos.to_pandas()
dataframe_trans = tabla_parquet_trans.to_pandas()

# Datos de los DtaFrames de los clientes
print("Clientes :")
print(dataframe_clientes.columns)  # Mostrar el nombre de las columnas

# Datos de los DtaFrames de los saldos
print("saldos:")
print(dataframe_saldos.columns)  # Mostrar el nombre de las columnas


# Datos de los DtaFrames de las transferencias
print("transferencias:")
print(dataframe_trans.columns)  # Mostrar el nombre de las columnas

 # ---------------------------- Transformacion de datos -----------------------------------------------------------------

# Renombrar la columna 'TipoDocum' a 'TIPODOCUM' de la tabla saldos
dataframe_saldos.rename(columns={'TipoDocum': 'TIPODOCUM'}, inplace=True)

# Obtener conjuntos de columnas de cada DataFrame
columnas_clientes = set(dataframe_clientes.columns)
columnas_saldos = set(dataframe_saldos.columns)
columnas_trans = set(dataframe_trans.columns)

# Encontrar las columnas en comun clientes-saldos
columnas_en_comun_cs = columnas_clientes.intersection(columnas_saldos)
# Encontrar las columnas en comun saldos-transferencias
columnas_en_comun_st = columnas_saldos.intersection(columnas_trans)
# Encontrar las columnas en comun transferencias-clientes
columnas_en_comun_tc = columnas_trans.intersection(columnas_clientes)

print("Columnas en común entre clientes-saldos:", columnas_en_comun_cs)
print("Columnas en común entre saldos-transacciones:", columnas_en_comun_st)
print("Columnas en común entre transacciones-clientes:", columnas_en_comun_tc)


# -------------------------------Limpiando la data de las bases de datos--------------------------------------------
# Fusionar las bases de datos
merged_data = pd.merge(dataframe_clientes, dataframe_saldos, on=["NroDocum"])  # fusion de la base clientes con saldos
merged_data = pd.merge(merged_data, dataframe_trans, on=["Contrato", "PlanProducto"])  # fusion de la base resultante con transferencias

# Convertir columnas de fecha al formato adecuado de la base de datos fusionada
merged_data['FecNacim'] = pd.to_datetime(merged_data['FecNacim'], format='%d/%m/%Y')
merged_data['FechaEfectiva'] = pd.to_datetime(merged_data['FechaEfectiva'], format='%d/%m/%Y')
merged_data['FechaProceso'] = pd.to_datetime(merged_data['FechaProceso'], format='%d/%m/%Y')

# Cálculo de retiros netos
merged_data["Retiros_Netos"] = merged_data["ValorNeto"].where(merged_data["TipoOper"] == "Retiro", 0) - \
                                merged_data["ValorNeto"].where(merged_data["TipoOper"] == "Aportacion", 0)

# Agrupación de datos a nivel de cliente
grouped_data = merged_data.groupby(["NroDocum"]).agg({
    "Retiros_Netos": "sum",
    "Contrato": "nunique",  # Cantidad de contratos por cliente
    # Otras estadísticas resumidas que puedan ser relevantes
}).reset_index()

# Mostrar la cabeza y columnas de la base de datos
print("\nBase de datos fusionada:")
print(merged_data.head())
print(merged_data.columns)

# Fusionar las dos columnas TIPODOCUM_x y TIPODOCUM_y en una sola columna para tener una sola columa de este tipo
merged_data['TIPODOCUM'] = merged_data['TIPODOCUM_x'].fillna(merged_data['TIPODOCUM_y'])

# Eliminar las columnas TIPODOCUM_x y TIPODOCUM_y ya que ahora tenemos una sola columna
merged_data.drop(['TIPODOCUM_x', 'TIPODOCUM_y'], axis=1, inplace=True)

# Calcular la edad del cliente
merged_data['Edad'] = datetime.now().year - merged_data['FecNacim'].dt.year

# Obtener las columnas de saldo
saldo_columns = [col for col in merged_data.columns if col.startswith('SALDO_')]

# Calcular el saldo promedio por cliente
merged_data['Saldo_Promedio'] = merged_data[saldo_columns].mean(axis=1)

# Filtrar las transacciones de los últimos 3 meses
three_months_ago = datetime.now() - timedelta(days=90)
transactions_last_3_months = merged_data[merged_data['FechaProceso'] >= three_months_ago]

# Calcular el total retirado por cada cliente en los últimos 3 meses
retiros_last_3_months = transactions_last_3_months.groupby('NroDocum')['Retiros_Netos'].sum().reset_index()
retiros_last_3_months.rename(columns={'Retiros_Netos': 'Total_Retirado_3_meses'}, inplace=True)

# Fusionar el total retirado en los últimos 3 meses con merged_data
merged_data = pd.merge(merged_data, retiros_last_3_months, on="NroDocum", how="left")

# Llenar los valores NaN con ceros (en caso de que no haya habido retiros para algunos clientes)
merged_data['Total_Retirado_3_meses'].fillna(0, inplace=True)

# Definir la variable objetivo que es que el CLIENTE retire el 70% o más de su saldo en los siguientes 3 meses.
merged_data['Retiro_Importante'] = (merged_data['Total_Retirado_3_meses'] >= 0.7 * merged_data['Saldo_Promedio']).astype(int)

# Mostrar la cabeza y columnas de la base de datos final
print("\nBase de datos fusionada final:")
print(merged_data.head())
print(merged_data.columns)
# Separar variables independientes (x) de la dependiente o variable objetivo (y) del conjunto de datos.
X = merged_data[['Saldo_Promedio', 'Edad']]
y = merged_data['Retiro_Importante']

# Dividir los datos en ensayo y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------Modelo predictivo------------------------------------------------------------

# Inicializar el clasificador de bosques aleatorios
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# n_estimators especifica el número de árboles en el bosque. En este caso, se establece en 100.
# random_state se utiliza para garantizar que los resultados sean reproducibles. Aquí se establece en 42, pero puede ser cualquier valor entero.

# El modelo se entrena con los datos  y se utilizan las características  Saldo_Promedio y
# Edad para predecir si un cliente retirara el 70% o más de su saldo en los siguientes 3 meses.

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
# Sin perdida de generalidad, este modelo es de clasificación binaria por lo que se refiere a unpro blema en el que
# solo hay dos posibles resultados o clases. En este caso, las dos clases son "retiro importante" (1) o "no retiro importante" (0).
print("Accuracy:", accuracy)

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# La matriz de confusión es una herramienta que se utiliza en el análisis de clasificación para evaluar el rendimiento
# de un modelo predictivo. Esta matriz muestra la cantidad de verdaderos positivos (TP), falsos positivos (FP), verdaderos negativos (TN)
# y falsos negativos (FN) obtenidos por el modelo en un conjunto de datos de prueba.

print("Confusion Matrix:")
print(conf_matrix)