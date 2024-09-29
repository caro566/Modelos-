import requests
from bs4 import BeautifulSoup
import nltk
import re
from google.colab import drive
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Configurar NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Montar Google Drive (si se usa en Google Colab)
drive.mount('/content/drive')

# Parámetros iniciales
initial_url = "/wiki/Ciencias_de_la_computaci%C3%B3n"
base_url = "https://es.wikipedia.org"
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

# Función para rastrear y extraer títulos
def crawl_and_extract_titles(url, max_levels, current_level, document_terms):
    if current_level > max_levels:
        return
    
    # Realizar la solicitud GET
    full_url = base_url + url
    response = requests.get(full_url)

    if response.status_code != 200:
        print(f"Error al acceder a {full_url}")
        return

    # Parsear la página
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extraer párrafos como títulos
    titles = [p.text for p in soup.find_all('p')]
    document_terms[url] = titles

    # Imprimir títulos
    print(f'Títulos en {url}: {titles}')

    # Extraer enlaces dentro de párrafos
    links = [link.get('href') for p in soup.find_all('p') for link in p.find_all('a') if link.get('href')]

    # Filtrar enlaces válidos
    valid_links = [link for link in links if link.startswith("/wiki/") and link not in document_terms]

    # Limitar la cantidad de enlaces por nivel a 10
    valid_links = valid_links[:10]

    # Recursivamente rastrear los enlaces
    for link in valid_links:
        crawl_and_extract_titles(link, max_levels, current_level + 1, document_terms)

# Diccionario para almacenar los títulos por documento
document_terms = {}

# Llamar a la función de rastreo
crawl_and_extract_titles(initial_url, max_levels=2, current_level=0, document_terms=document_terms)

# Ver los títulos en cada página y el diccionario final
print("Diccionario document_terms:")
print(document_terms)

# Función para actualizar el conjunto de términos únicos
def updateSet(document_terms):
    terms_set = set()
    for terminos in document_terms.values():
        terms_set.update(terminos)
    return terms_set

# Preprocesar los términos en cada documento
for document_url, terms in document_terms.items():
    tokenized_terms = []
    for term in terms:
        # Normalizar y tokenizar los términos
        term = re.sub(r'[^\w\s]', '', term)
        term = re.sub(r'\d', '', term)
        if term:
            tokens = word_tokenize(term.lower())  # Convertir a minúsculas
            tokenized_terms.extend(tokens)
    document_terms[document_url] = tokenized_terms

# Guardar el diccionario en un archivo de texto
with open('/content/drive/My Drive/ModeloRI/diccionario.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(list(updateSet(document_terms))))
print(document_terms)

# Filtrar stopwords
for document_url, terms in document_terms.items():
    filtered_terms = [term for term in terms if term not in stop_words]
    document_terms[document_url] = filtered_terms

# Guardar el diccionario filtrado
with open('/content/drive/My Drive/ModeloRI/diccionario_filtrado.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(list(updateSet(document_terms))))
print(document_terms)

# Aplicar Stemming
for document_url, terms in document_terms.items():
    stemmed_terms = [stemmer.stem(term) for term in terms]
    document_terms[document_url] = stemmed_terms

# Guardar el diccionario con stemming
with open('/content/drive/My Drive/ModeloRI/diccionario_stemmed.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(list(updateSet(document_terms))))
print(document_terms)

# Crear matriz binaria
unique_terms = list(updateSet(document_terms))
binary_matrix = np.zeros((len(document_terms), len(unique_terms)), dtype=int)

# Llenar la matriz binaria
for i, (doc, terms) in enumerate(document_terms.items()):
    for j, term in enumerate(unique_terms):
        if term in terms:
            binary_matrix[i, j] = 1

# Crear DataFrame de la matriz binaria
binary_matrix_df = pd.DataFrame(binary_matrix, columns=unique_terms)
binary_matrix_df.insert(0, "Documentos", list(document_terms.keys()))

# Guardar el DataFrame en un archivo CSV
binary_matrix_df.to_csv('/content/drive/My Drive/ModeloRI/matriz_binaria.csv', index=False)
print(binary_matrix)

# Leer la consulta del usuario
consulta_q = input("Ingresa la consulta Q: ")
print(f"Consulta Q: {consulta_q}")

# Función para preprocesar la consulta
def preprocess_query(query):
    words = query.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

# Aplicar preprocesamiento a la consulta
consulta_procesada = preprocess_query(consulta_q)
print(f"Consulta procesada: {consulta_procesada}")

# Vectorizar términos con TF-IDF
documents = list(document_terms.keys())
term_lists = list(document_terms.values())
tfidf_vectorizer = TfidfVectorizer()
documents_as_text = [' '.join(terms) for terms in term_lists]
tfidf_matrix = tfidf_vectorizer.fit_transform(documents_as_text)

# Obtener el vector de la consulta
consulta_vector = tfidf_vectorizer.transform([consulta_procesada])

# Agregar la consulta al conjunto de documentos
documents.append("Consulta")
term_lists.append(consulta_procesada.split())
documents_as_text.append(consulta_procesada)
