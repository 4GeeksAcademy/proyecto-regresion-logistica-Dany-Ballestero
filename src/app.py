#from utils import db_connect
#engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle # Librería clave para guardar el modelo final

# 1. CARGA DE DATOS
url = "https://storage.googleapis.com/breathecode/project-files/bank-marketing-campaign-data.csv"
try:
    df = pd.read_csv(url, sep=';')
    if len(df.columns) == 1:
        df = pd.read_csv(url, sep=',')
except:
    df = pd.read_csv(url, sep=',')

# 2. PREPROCESAMIENTO
# Convertimos la variable objetivo a números
df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Convertimos las variables categóricas
X = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
y = df['y']

# Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamos los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Nota: Aquí no necesitamos transformar X_test porque en producción 
# solo nos interesa dejar el modelo entrenado, pero es buena práctica mantener la lógica.

# 3. ENTRENAMIENTO DEL MODELO OPTIMIZADO
model_optimized = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
model_optimized.fit(X_train_scaled, y_train)

# 4. GUARDAR EL MODELO (SERIALIZACIÓN)
# Esto crea un archivo físico con tu "inteligencia artificial" ya entrenada
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model_optimized, file)

print("¡Modelo entrenado y guardado con éxito como 'logistic_regression_model.pkl'!")