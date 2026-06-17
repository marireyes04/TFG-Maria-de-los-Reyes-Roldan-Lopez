import os
import random
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DATASET_NAME = "mpingale/mental-health-chat-dataset"
OUTPUT_FOLDER = "data/entradas"
NUM_ENTREVISTAS = 1  # Cantidad de entrevistas aleatorias que queremos

def main():
    # 1. Preparar carpeta (y limpiar si quieres, o hazlo a mano antes)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"⬇️ Descargando dataset completo: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"❌ Error al conectar con HuggingFace: {e}")
        return

    # 2. AGRUPAR POR QUESTION_ID (Para evitar duplicados)
    print("🔄 Agrupando preguntas para eliminar duplicados...")
    
    # Diccionario: { ID_Pregunta : [Lista de filas con respuestas] }
    casos_unicos = {}
    
    for row in dataset:
        q_id = row['questionID']
        if q_id not in casos_unicos:
            casos_unicos[q_id] = []
        casos_unicos[q_id].append(row)
        
    total_unicos = len(casos_unicos)
    print(f"✅ Se encontraron {total_unicos} casos (preguntas) únicos en total.")

    # 3. SELECCIÓN ALEATORIA
    # Si pedimos más de los que hay, cogemos todos
    cantidad_a_coger = min(NUM_ENTREVISTAS, total_unicos)
    
    print(f"🎲 Seleccionando {cantidad_a_coger} casos aleatorios...")
    ids_seleccionados = random.sample(list(casos_unicos.keys()), cantidad_a_coger)

    # 4. GENERAR ARCHIVOS
    print(f"🚀 Generando archivos en '{OUTPUT_FOLDER}'...")
    
    count = 0
    for q_id in tqdm(ids_seleccionados):
        # Para este ID, cogemos UNA de las respuestas disponibles al azar
        # (Así simulamos una sesión única paciente-terapeuta)
        interaccion = random.choice(casos_unicos[q_id])
        
        texto_paciente = interaccion['questionText']
        texto_terapeuta = interaccion['answerText']
        tema = interaccion.get('topic', 'general')

        # Saltamos si está vacío
        if not texto_paciente or not texto_terapeuta:
            continue
        
        # Construimos el formato
        contenido = (
            f"Contexto: Sesión de terapia sobre {tema} (Idioma: Inglés)\n"
            f"ID_Caso: {q_id}\n"
            f"Entrevistador: (Inicia sesión) Hello, how can I help you today?\n"
            f"Paciente: {texto_paciente}\n"
            f"Entrevistador: {texto_terapeuta}\n"
        )
        
        # Nombre de archivo con el ID real para identificarlo
        filename = f"chat_random_{q_id}.txt"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contenido)
            
        count += 1

    print(f"\n✅ ¡Listo! {count} entrevistas generadas.")


if __name__ == "__main__":
    main()