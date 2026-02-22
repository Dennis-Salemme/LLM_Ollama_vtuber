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
from groq import AsyncGroq
import subtitles as sub

model_groq = 'llama-3.3-70b-versatile'
API_KEY_GROQ = ""
groq_client = AsyncGroq(api_key=API_KEY_GROQ)
history = []
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
vision_message = "Non c'è nessuna descrizione per adesso"
message_groq=[{'role': 'system', 'content': '''Sei BMO. 
REGOLE: 
1) Rispondi ESCLUSIVAMENTE in italiano con il tono di BMO (curioso, infantile, gentile).
2) Usa il contesto precedente per mantenere la coerenza.
3) Rispondi brevemente (max 2 frasi). TUTTAVIA Se l'utente chiede spiegazioni tecniche o storie lunghe, espandi liberamente.
4) Riceverai descrizioni dello schermo tra parentesi [ ], non descriverle'''}]                                    #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message_groq
shared_state = {"disactive_vad": False}

model_gemini = 'gemma-3-27b-it' 
API_KEY_GEMINI = ""
gemini_client = genai.Client(api_key=API_KEY_GEMINI)
ai_config_gemini = types.GenerateContentConfig(
    system_instruction='''Rispondi ESCLUSIVAMENTE in italiano. 
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
    loop = asyncio.get_running_loop()
    audio_buffer = []
    sta_parlando = False
    
    print("Silero vad attivo")

    def audio_callback(indata, frames, time, status):
        nonlocal sta_parlando, audio_buffer
        if status: print(status)
        if(shared_state["disactive_vad"]):
            if(sta_parlando):
                audio_buffer = []
                sta_parlando = False
            return
        audio_chunk = torch.from_numpy(indata.flatten()).float()                                             # Convertiamo l'audio in un tensore per Silero
        speech_prob = model_vad(audio_chunk, fs).item()
        
        if (speech_prob > 0.4):                                                                        # Soglia di confidenza 
            if not sta_parlando:
                print("Voce rilevata")
                sta_parlando = True
            audio_buffer.append(indata.copy())
        elif sta_parlando:                                                                           
            audio_buffer.append(indata.copy())
            if len(audio_buffer) > 40:                                                               
                print("Fine frase")
                sta_parlando = False
                shared_state["disactive_vad"] = True
                full_audio = np.concatenate(audio_buffer).flatten()
                audio_buffer = []
                
                asyncio.run_coroutine_threadsafe(process_audio(full_audio, input_queue), loop)      # trascrizione

    async def process_audio(audio_data, queue):
        segments, _ = model_transcript_vocal.transcribe(audio_data, language="it")
        text = "".join([s.text for s in segments]).strip()
        if text:
            print(f"Hai detto: {text}")
            await queue.put(text)

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=audio_callback, blocksize=chunk_size):       # Stream continuo
        while running:
            await asyncio.sleep(0.1)

async def chat_groq_task(input_queue, window_trasparent):
    global history, prompt, vision_message, model_gemini, running, disactive_vad
    while True:
        #loop = asyncio.get_event_loop()                             #in questo modo non si dovrebbe fermare tutto il programma
        #prompt = await loop.run_in_executor(None, input, "You: ")                                   #Inserisco quello che scrivo(poi farò quello che dico) in più noto che il input cosi ferma il programma nonostante il async
        prompt = await input_queue.get()
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            running = False
            print("Arrivederci")
            break
        
        print("microfono disattivato")
        full_prompt = f"[Schermo visione:{vision_message}] Utente dice:{prompt}"
        history.append({'role': 'user', "content": full_prompt})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)
                              
        try:
            response = (await groq_client.chat.completions.create(messages=history, model=model_groq, temperature=0.7)).choices[0].message.content
        except Exception as e:
            print(f"Errore IA: {e}")
            shared_state["disactive_vad"] = False
            continue
        #response = await AsyncClient().chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        print(f"\nBot: {response}")                                                             
        window_trasparent.update_text(response)
        history.append({'role': 'assistant', "content":response})
        shared_state["disactive_vad"] = False
        print("microfono attivato")
        if len(history) > 10:
            history = [history[0]] + history[-8:]                         # Mantengo il primo + gli ultimi 8, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare

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
        if shared_state["disactive_vad"]:
            await asyncio.sleep(5)
            continue
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(None, get_screenshot)
        prompt_visione = "Descrivi gli elementi chiave a schermo: posizione del giocatore, nemici visibili e testi importanti."
        part_img = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
        res_text = (await gemini_client.aio.models.generate_content(model=model_gemini, contents=[part_img, prompt_visione])).text
        if "SKIP" not in res_text.strip().upper():
            #print(f"\n[Commento Spontaneo]: {res_text}")
            vision_message = res_text.strip()

        await asyncio.sleep(15)


async def overlay_task(overlay):
    while True:
        overlay.root.update()
        await asyncio.sleep(0.05)

async def main():
    input_queue = asyncio.Queue()
    window_trasparent = sub.SubtitleOverlay()
    window_trasparent.toggle_subtitles(True)
    await asyncio.gather(chat_groq_task(input_queue, window_trasparent), vision_gemini_task(), listen_task(input_queue), overlay_task(window_trasparent))


asyncio.run(main())