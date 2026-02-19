from time import sleep
from PIL import Image
import mss
import asyncio
import io
from google import genai
from google.genai import types

history = []
model_gemini = 'gemini-2.0-flash' 
API_KEY = ""
client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1'})
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
message=[{'role': 'user', 'parts': [{'text':'''Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio. 
Se ti invio un'immagine, commentala solo se accade qualcosa di degno di nota, altrimenti scrivi 'SKIP'.'''}]}]      #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message
vision_message = "Non c'è nessuna descrizione per adesso"
ai_config = types.GenerateContentConfig(
    system_instruction='''Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio. 
Se ti invio un'immagine, commentala solo se accade qualcosa di degno di nota, altrimenti scrivi 'SKIP'.''',
    temperature=0.7 
)
async def chat_ollama_task():
    global history, prompt, vision_message, model_gemini
    while True:
        loop = asyncio.get_event_loop()                             #in questo modo non si dovrebbe fermare tutto il programma
        prompt = await loop.run_in_executor(None, input, "You: ")                                   #Inserisco quello che scrivo(poi farò quello che dico) in più noto che il input cosi ferma il programma nonostante il async
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            print("Arrivederci")
            break
        
        full_prompt = f"[Schermo:{vision_message}] Utente:{prompt}"
        history.append({'role': 'user', 'parts':[{'text':full_prompt}]})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)
                              
        try:
            response = await client.aio.models.generate_content(model=model_gemini, contents=history).text
        except Exception as e:
            print(f"Errore IA: {e}")
            return 0
        #response = await AsyncClient().chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        print(f"\nBot: {response}")                                                             

        history.append({'role': 'model', 'parts':[{'text':response}]})

        if len(history) > 10:
            history = history[-10:]                         # Mantieni il primo (System) + gli ultimi 6, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare

def get_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

async def vision_moondream_task():
    global prompt, vision_message, model_gemini
    model_moondream= 'moondream' 
    while True:
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            break
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(None, get_screenshot)
        prompt_visione = "Descrivi cosa succede a schermo."
        part_img = types.Part.from_bytes(data=image_bytes, mime_type='image/png')
        res_text = (await client.aio.models.generate_content(model=model_gemini, contents=[part_img, prompt_visione])).text
        if "SKIP" not in res_text.strip().upper():
            print(f"\n[Commento Spontaneo]: {res_text}")
            vision_message = res_text.strip()

        await asyncio.sleep(15)

async def main():
    await asyncio.gather(chat_ollama_task(), vision_moondream_task())

#inizio del main di quello che interessa
asyncio.run(main())
