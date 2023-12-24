from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, lit, row_number, udf, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql.types import BooleanType, IntegerType

# Crear una sesión de Spark con la reanudación de la transferencia desactivada y una carpeta temporal personalizada
spark = SparkSession.builder \
    .config("spark.shuffle.service.index.cache.size", "0") \
    .config("spark.local.dir", "D:\\PySpark\\temp") \
    .getOrCreate()

# Supongamos que 'paises' es una lista de los nombres de todos los países para los que tienes datos
paises = ["francia"]

# Definir una función para determinar si hay un gol en la primera mitad
def gol_primera_mitad(goles):
    for gol in goles:
        if gol == '-':
            return False
        elif int(gol) <= 45:
            return True
    return False

# Crear una UDF (función definida por el usuario)
gol_primera_mitad_udf = udf(gol_primera_mitad, BooleanType())

for pais in paises:
    # Leer el archivo CSV
    df = spark.read.csv(f"datos/{pais}.csv", header=True, inferSchema=True)

    # Reparticionar los datos
    df = df.repartition(6)

    # Convertir los minutos de gol en una lista de números enteros y crear una columna binaria
    df = df.withColumn("Local_gol", split(col("Local_gol"), "\."))
    df = df.withColumn("Visitante_gol", split(col("Visitante_gol"), "\."))
    df = df.withColumn("Local_gol_primera_mitad", gol_primera_mitad_udf(col("Local_gol")))
    df = df.withColumn("Visitante_gol_primera_mitad", gol_primera_mitad_udf(col("Visitante_gol")))
    df = df.withColumn("gol_primera_mitad", (col("Local_gol_primera_mitad") | col("Visitante_gol_primera_mitad")).cast(IntegerType()))

    # Codificar las columnas de tipo string
    indexer = StringIndexer(inputCol="Temporada", outputCol="TemporadaIndex")
    df = indexer.fit(df).transform(df)
    indexer = StringIndexer(inputCol="Local", outputCol="LocalIndex")
    df = indexer.fit(df).transform(df)
    indexer = StringIndexer(inputCol="Visitante", outputCol="VisitanteIndex")
    df = indexer.fit(df).transform(df)

    # Aplicar la codificación one-hot
    encoder = OneHotEncoder(inputCols=["TemporadaIndex", "LocalIndex", "VisitanteIndex"], outputCols=["TemporadaVec", "LocalVec", "VisitanteVec"])
    df = encoder.fit(df).transform(df)

    # Crear una columna de índice
    window = Window.partitionBy('Temporada').orderBy(df['Temporada'].desc(), df['Jornada'].desc())
    df = df.withColumn('indice', row_number().over(window))

    # Crear una columna de peso
    df = df.withColumn('peso', lit(1) / col('indice'))

    # Crear un VectorAssembler para combinar las columnas de características en una sola columna
    assembler = VectorAssembler(inputCols=["TemporadaVec", "Jornada", "LocalVec", "VisitanteVec", "peso"], outputCol="caracteristicas")

    # Transformar los datos
    df = assembler.transform(df)

    # Crear una instancia del modelo
    glm = GeneralizedLinearRegression(featuresCol='caracteristicas', labelCol='gol_primera_mitad', family="binomial", link="logit", maxIter=10, regParam=0.3)

    # Ajustar el modelo a los datos
    modelo = glm.fit(df)

    # Guardar el modelo
    # modelo.write().overwrite().save(f"modelos/{pais}_modelo")
    modelo.save(f"modelos/{pais}_modelo")

    # Imprimir los coeficientes y la intersección del modelo
    print(f"Modelo para {pais}:")
    print("Coeficientes: " + str(modelo.coefficients))
    print("Intersección: " + str(modelo.intercept))
    print("\n")