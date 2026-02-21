from time import sleep
from PIL import Image
import mss
import asyncio
import io
from google import genai
from google.genai import types
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
import numpy as np

history = []
model_gemini = 'gemini-2.5-flash-lite' 
API_KEY = ""
client = genai.Client(api_key=API_KEY)
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
vision_message = "Non c'è nessuna descrizione per adesso"
ai_config = types.GenerateContentConfig(
    system_instruction='''Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio. 
Se ti invio un'immagine, commentala solo se accade qualcosa di degno di nota, altrimenti scrivi 'SKIP'.''',
    temperature=0.7 
)
model_transcript_vocal = WhisperModel("small", device="cpu", compute_type="int8")
running = True
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

async def listen_task(input_queue):
    global running
    fs = 16000
    chunk_size = 512 
    
    main_loop = asyncio.get_running_loop()
    audio_buffer = []
    is_speaking = False
    
    print("Silero vad attivo")

    def audio_callback(indata, frames, time, status):
        nonlocal is_speaking, audio_buffer
        if status: print(status)
        audio_chunk = torch.from_numpy(indata.flatten())                                             # Convertiamo l'audio in un tensore per Silero
        speech_prob = model_vad(audio_chunk, fs).item()
        
        if speech_prob > 0.6:                                                                        # Soglia di confidenza 
            if not is_speaking:
                print("Voce rilevata")
                is_speaking = True
            audio_buffer.append(indata.copy())
        elif is_speaking:                                                                           
            audio_buffer.append(indata.copy())
            if len(audio_buffer) > 20:                                                               
                print("Fine frase")
                is_speaking = False
                full_audio = np.concatenate(audio_buffer).flatten()
                audio_buffer = []
                
                asyncio.run_coroutine_threadsafe(process_audio(full_audio, input_queue), main_loop)      # trascrizione

    async def process_audio(audio_data, queue):
        segments, _ = model_transcript_vocal.transcribe(audio_data, language="it")
        text = "".join([s.text for s in segments]).strip()
        if text:
            print(f"Hai detto: {text}")
            await queue.put(text)

    with sd.InputStream(samplerate=fs, channels=1, callback=audio_callback, blocksize=chunk_size):       # Stream continuo
        while running:
            await asyncio.sleep(0.1)

async def chat_gemini_task(input_queue):
    global history, prompt, vision_message, model_gemini, running
    while True:
        #loop = asyncio.get_event_loop()                             #in questo modo non si dovrebbe fermare tutto il programma
        #prompt = await loop.run_in_executor(None, input, "You: ")                                   #Inserisco quello che scrivo(poi farò quello che dico) in più noto che il input cosi ferma il programma nonostante il async
        prompt = await input_queue.get()
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            running = False
            print("Arrivederci")
            break
        
        full_prompt = f"[Schermo:{vision_message}] Utente:{prompt}"
        history.append({'role': 'user', 'parts':[{'text':full_prompt}]})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)
                              
        try:
            response = (await client.aio.models.generate_content(model=model_gemini, contents=history, config=ai_config)).text
        except Exception as e:
            print(f"Errore IA: {e}")
            return 0
        #response = await AsyncClient().chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        print(f"\nBot: {response}")                                                             

        history.append({'role': 'model', 'parts':[{'text':response}]})

        if len(history) > 6:
            history = history[-6:]                         # Mantieni il primo (System) + gli ultimi 6, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare

def get_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.thumbnail((800, 800))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

async def vision_gemini_task():
    global prompt, vision_message, model_gemini, running
    model_moondream= 'moondream' 
    while running:
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(None, get_screenshot)
        prompt_visione = "Descrivi cosa succede a schermo."
        part_img = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
        res_text = (await client.aio.models.generate_content(model=model_gemini, contents=[part_img, prompt_visione], config=ai_config)).text
        if "SKIP" not in res_text.strip().upper():
            #print(f"\n[Commento Spontaneo]: {res_text}")
            vision_message = res_text.strip()

        await asyncio.sleep(15)

async def main():
    input_queue = asyncio.Queue()
    await asyncio.gather(chat_gemini_task(input_queue), vision_gemini_task(), listen_task(input_queue))

#inizio del main di quello che interessa
asyncio.run(main())