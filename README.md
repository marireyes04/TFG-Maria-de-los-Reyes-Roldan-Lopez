# TFG-Maria-de-los-Reyes-Roldan-Lopez

# CDSS · Clinical Decision Support System (Sistema de Soporte para Decisiones Clínicas)

Este repositorio contiene una aplicación de **Soporte para Decisiones Clínicas (CDSS)** basada en **RAG (Retrieval-Augmented Generation)**. La herramienta está diseñada para ayudar a profesionales de la psicología y la salud mental a analizar entrevistas clínicas estructuradas por turnos de conversación, realizar consultas directas y contrastar la información del caso con manuales clínicos y protocolos de seguridad de manera rigurosa y objetiva.

El sistema implementa reglas clínicas muy estrictas para evitar alucinaciones, impedir diagnósticos automáticos no autorizados e impedir la inferencia subjetiva de riesgos (como autolisis o autolesiones) a menos que se mencionen de forma literal y explítica en el texto de la entrevista.

---

## 🛠️ Arquitectura y Tecnologías

El proyecto está construido bajo el siguiente stack tecnológico y librerías clave:
*   **Frontend:** [Streamlit](https://streamlit.io/) (`app.py`), que proporciona una interfaz interactiva y segmentada en diferentes vistas (Entrevistas, Chat Clínico, Evaluación RAG, y Depuración).
*   **Orquestación RAG:** [LangChain](https://www.langchain.com/) (`chat_profesional.py`), para estructurar las consultas, gestionar el historial clínico y formatear los prompts clínicos estrictos.
*   **Base de Datos Vectorial:** [ChromaDB](https://www.trychroma.com/) (`data/chroma`), configurada localmente para almacenar y recuperar de forma semántica los fragmentos de entrevistas y manuales clínicos.
*   **Generador de Embeddings:** El modelo de embeddings local `BAAI/bge-m3` (de Hugging Face, vía `sentence-transformers`).
*   **Modelo de Lenguaje (LLM):** `meta-llama/Llama-3.1-8B-Instruct` consumido a través del `InferenceClient` de Hugging Face.
*   **Ingesta de Documentos:** `unstructured` (con `pdfminer.six` y `docx2txt`) para parsear automáticamente documentos PDF y TXT.

---

## 📂 Estructura del Repositorio

A continuación se detalla la organización de los archivos principales del proyecto:

*   **app.py**: Interfaz gráfica e interactiva de Streamlit para el usuario clínico y administrador.
*   **chat_profesional.py**: Backend RAG que implementa la recuperación semántica, el procesamiento de historial y las llamadas al modelo de Hugging Face bajo reglas estrictas de prompt.
*   **ingest_bulk.py**: Script de ingesta masiva. Parsea y tokeniza las entrevistas (`data/entradas/`) identificando interlocutores (paciente/terapeuta) e indexa los manuales (`data/manuales/`) en colecciones vectoriales diferenciadas.
*   **run_basic_eval.py**: Script de evaluación local rápida (léxica, de estructura y formato) sin requerir llamadas externas o modelos juez.
*   **run_ragas_eval.py**: Script de evaluación avanzado con la biblioteca **Ragas**, evaluando métricas como *Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall* y *Answer Correctness* usando un LLM juez (por defecto `Qwen/Qwen2.5-7B-Instruct`).
*   **requirements.txt**: Listado de librerías y dependencias de Python necesarias.
*   `data/`: Directorio principal de datos y bases vectoriales:
    *   `entradas/`: Contiene las transcripciones de las entrevistas en formato `.txt` (por ejemplo, `chat_random_*.txt`).
    *   `manuales/`: Contiene manuales clínicos en PDF o TXT, organizados por carpetas jerárquicas de nivel (seguridad, guías clínicas, protocolos, etc.).
    *   `chroma/`: Base de datos vectorial persistente.

---

## 🗜️ Restauración de la Base de Datos Vectorial (Chroma)

Debido a los límites de tamaño establecidos por GitHub para subir archivos individuales (restricción estricta de 100 MB y avisos a partir de 50 MB), **algunos archivos grandes de la base de datos de Chroma se han incluido comprimidos en el repositorio**.

Para utilizar la base de datos pre-indexada sin necesidad de volver a ejecutar la ingesta masiva completa, debe extraer estos archivos de la siguiente manera:

1.  **Descomprimir la base de datos SQLite principal:**
    *   Localice el archivo comprimido `chroma.zip` dentro de la carpeta `data/chroma/`.
    *   Descomprímalo en ese mismo directorio para generar el archivo `chroma.sqlite3`.
    *   *Ruta resultante esperada:* `data/chroma/chroma.sqlite3`

2.  **Descomprimir el índice vectorial HNSW:**
    *   Navegue a la subcarpeta del índice UUID en `data/chroma/a2fd49db-2baa-4028-b89b-6fcc66c3f8d4/`.
    *   Localice el archivo comprimido `data_level0.rar`.
    *   Descomprímalo en ese mismo directorio para restaurar el archivo binario indexado `data_level0.bin`.
    *   *Ruta resultante esperada:* `data/chroma/a2fd49db-2baa-4028-b89b-6fcc66c3f8d4/data_level0.bin`

> [!IMPORTANT]
> Si prefiere regenerar la base de datos vectorial por completo desde las fuentes originales, puede eliminar el directorio `data/chroma/` por completo y ejecutar el script de ingesta (consulte la sección de *Uso*).

---

## 🚀 Requisitos e Instalación

### 1. Clonar el repositorio y acceder a la carpeta
```bash
git clone <url-del-repositorio>
cd TFG
```

### 2. Crear y activar el entorno virtual
*   **En Windows (PowerShell):**
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
*   **En macOS/Linux:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
El backend requiere un token de acceso de Hugging Face para realizar inferencias con el modelo Llama. Configure la variable de entorno correspondiente en su sistema o consola:

*   **Windows (PowerShell):**
    ```powershell
    $env:HF_TOKEN="tu_token_de_hugging_face_aqui"
    ```
*   **macOS/Linux:**
    ```bash
    export HF_TOKEN="tu_token_de_hugging_face_aqui"
    ```

*(Opcional)* Si desea activar el modo de depuración detallado en consola, configure:
```powershell
$env:CDSS_DEBUG="1"
```

---

## 💻 Uso de la Aplicación

### Ingesta de Datos (Opcional)
Si ha modificado las entrevistas en `data/entradas/` o los manuales en `data/manuales/`, o si ha eliminado la base de datos previa, puede volver a indexar todo ejecutando:
```bash
python ingest_bulk.py
```
*Nota: Este script no re-ingestará los archivos que ya se encuentren registrados en los metadatas de Chroma para optimizar el tiempo de ejecución.*

### Iniciar la Aplicación Streamlit
Para levantar el servidor local y abrir la interfaz de usuario:
```bash
streamlit run app.py
```
Una vez iniciado, Streamlit abrirá de forma automática una pestaña en su navegador (habitualmente en `http://localhost:8501`).

La aplicación incluye dos roles de acceso iniciales por defecto (los cuales se pueden gestionar o modificar dentro del código de `app.py`):
*   **Terapeuta:** Cuenta con acceso a las pestañas de **Interviews** (visualizar y buscar casos) y **Clinical Chat** (interactuar mediante RAG con el caso seleccionado).
    *   *Usuario:* `therapist` | *Contraseña:* `therapist123`
*   **Administrador:** Acceso completo a todas las pestañas, incluyendo **RAG Evaluation** y la consola de **Admin / Debug**.
    *   *Usuario:* `admin` | *Contraseña:* `admin123`

---

## 📊 Evaluación del Sistema

El proyecto incluye dos flujos de evaluación para medir el desempeño de las respuestas y la fidelidad del contexto recuperado.

### 1. Evaluación Básica (Local y Rápida)
Compara las respuestas generadas o cargadas contra un dataset de validación (`ground_truth_long.csv`). Mide cobertura de formato (cumplir exactamente las 3 líneas requeridas), presencia/ausencia de palabras de riesgo y solapamiento léxico sin necesidad de API de pago de modelos juez.
```bash
python run_basic_eval.py --input ground_truth_long.csv
```
Los resultados detallados se guardarán automáticamente en la carpeta `basic_eval_outputs/`.

### 2. Evaluación RAGAS (Avanzada)
Utiliza la metodología Ragas con modelos juez remotos a través del hub de Hugging Face para evaluar métricas semánticas complejas:
```bash
python run_ragas_eval.py
```
Los resultados e históricos de métricas se almacenarán en `ragas_eval_outputs/`.
