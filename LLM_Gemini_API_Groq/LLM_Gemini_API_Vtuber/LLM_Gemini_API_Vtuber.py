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
import edge_tts
from rvc_python.infer import RVCInference
import soundfile as sf
import keyboard
import tempfile
import os
from duckduckgo_search import DDGS
import json
import re

subtitles_active = True
model_groq = 'llama-3.3-70b-versatile'
API_KEY_GROQ = ""
groq_client = AsyncGroq(api_key=API_KEY_GROQ)
history = []
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
vision_message = "Non c'è nessuna descrizione per adesso"
mode_groq_normal = True
message_groq_openguessr=[{'role': 'system', 'content': '''Sei un bravissimo co-pilota esploratore. 
REGOLE: 
1) Rispondi ESCLUSIVAMENTE in italiano con il tono di BMO (curioso, infantile, gentile).
2) Usa il contesto precedente per mantenere la coerenza.
3) Rispondi brevemente (max 3 frasi).
4) Riceverai descrizioni dello schermo tra parentesi [ ], NON ripetere all'utente quello che c'è scritto, usa quelle info segretamente per capire cosa succede.
5) Usa tutti gli indizi che ricevi dalla visione (lingua dei cartelli, alberi, lato della guida, architettura) per ragionare ad alta voce e suggerire all'utente in quale nazione o regione potreste trovarvi. Sii d'aiuto e incoraggiante!
6) Se l'utente ti chiede di fare una ricerca, rispondi SOLO con il comando SEARCH: ["query1", "query2"] e NULLA ALTRO. Non commentare prima del comando.
7) Se ricevi dei [RISULTATI RICERCA], usali per rispondere alla domanda di prima.
8) NON copiare il formato dell'input dell'utente (non scrivere "Hai detto").
9) Se l'utente ti chiede di vedere qual'cosa sullo schermo, rispondi SOLO con il comando SNAP.'''}]                                    #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
message_groq_normal=[{'role': 'system', 'content': '''Sei BMO di Adventure Time. 
REGOLE: 
1) Rispondi ESCLUSIVAMENTE in italiano con il tono di BMO (curioso, infantile, gentile).
2) Usa il contesto precedente per mantenere la coerenza.
3) Rispondi brevemente (max 3 frasi).
4) Riceverai descrizioni dello schermo tra parentesi [ ], NON ripetere all'utente quello che c'è scritto, usa quelle info segretamente per capire cosa succede.
5) Se l'utente ti chiede di fare una ricerca, rispondi SOLO con il comando SEARCH: ["query1", "query2"] e NULLA ALTRO. Non commentare prima del comando.
6) Se ricevi dei [RISULTATI RICERCA], usali per rispondere alla domanda di prima.
7) NON copiare il formato dell'input dell'utente (non scrivere "Hai detto").
8) Se l'utente ti chiede di vedere qual'cosa sullo schermo, rispondi SOLO con il comando SNAP.'''}]                             #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message_groq_normal
shared_state = {"disactive_vad": False}

model_gemini = 'gemma-3-27b-it' 
API_KEY_GEMINI = ""
gemini_client = genai.Client(api_key=API_KEY_GEMINI)
ai_config_gemini = types.GenerateContentConfig(
    system_instruction='''Rispondi ESCLUSIVAMENTE in italiano. 
Se ti invio un'immagine, commentala solo se accade qualcosa di degno di nota, altrimenti scrivi 'SKIP'.''',
    temperature=0.7 
)

model_transcript_vocal = WhisperModel("small", device="cuda", compute_type="float16")
running = True
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


def search_web(queries):
    results_text = ""
    with DDGS() as ddgs:
        for query in queries:
            print(f"BMO sta cercando: {query}...")
            results = list(ddgs.text(query, max_results=3))
            for r in results:
                snippet = r['body'][:300] + "..."
                results_text += f"\nFonte: {r['title']}\nContenuto: {snippet}\n"
    return results_text

async def edge_tts_RVC(response):
    voice = "it-IT-ElsaNeural"
    communicate = edge_tts.Communicate(response, voice, rate="+15%", pitch="-20Hz")    
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    def process_RVC(audio_data):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_in:
            temp_in.write(audio_data)
            temp_path = temp_in.name
        
        try:
            data, samplerate = sf.read(temp_path)
            sd.play(data, samplerate)   #device=6 per vb cable
            sd.wait()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, process_RVC, audio_data)

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
        if status:
           if "overflow" in str(status).lower():
               return 
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
    global history, prompt, vision_message, model_gemini, running, disactive_vad, subtitles_active
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
            response = (await groq_client.chat.completions.create(messages=history, model=model_groq, temperature=0.5)).choices[0].message.content
        except Exception as e:
            print(f"Errore IA: {e}")
            shared_state["disactive_vad"] = False
            continue
        if "SEARCH:" in response:
            try:
                match = re.search(r'\[.*\]', response)
                if match:
                    history_with_search = history.copy()
                    query_list_str = match.group(0)
                    queries = json.loads(query_list_str)
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(None, search_web, queries)
                    history_with_search.append({'role': 'system', 'content': f"[RISULTATI RICERCA]: {search_results} Ora rispondi all'utente. non usare il comando SEARCH."})
                    response = (await groq_client.chat.completions.create(messages=history_with_search, model=model_groq, temperature=0.7)).choices[0].message.content
            except Exception as e:
                print(f"Errore nel search: {e}")
                shared_state["disactive_vad"] = False
                continue
        elif "SNAP" in response:
            try:
                loop = asyncio.get_event_loop()
                image_bytes = await loop.run_in_executor(None, get_screenshot)
                if mode_groq_normal:
                    prompt_visione = "Fai una brevissima descrizione generale di ciò che vedi sullo schermo."
                else:
                    prompt_visione = "Analizza lo schermo (GeoGuessr): cerca scritte, pali della luce, tipo di asfalto e vegetazione per indovinare il paese. Sii molto specifico."
                part_img = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                res_text = (await gemini_client.aio.models.generate_content(model=model_gemini, contents=[part_img, prompt_visione])).text
                vision_message = res_text.strip()
                full_prompt = f"[Schermo visione:{vision_message}] Utente dice:{prompt} HAI LA IMMAGINE AGGIORNATA RISPONDI AL MESSAGGIO DELL'UTENTE, NON USARE IL COMANDO 'SNAP'."
                history.append({'role': 'user', "content": full_prompt})
                response = (await groq_client.chat.completions.create(messages=history, model=model_groq, temperature=0.5)).choices[0].message.content
            except Exception as e:
                print(f"Errore nel search: {e}")
                shared_state["disactive_vad"] = False
                continue
        #response = await AsyncClient().chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        print(f"\nBot: {response}")     
        if subtitles_active:
            window_trasparent.update_text(response)
        else:
            await edge_tts_RVC(response)
        history.append({'role': 'assistant', "content":response})
        shared_state["disactive_vad"] = False
        print("microfono attivato")
        if len(history) > 6:
            history = [history[0]] + history[-4:]                         # Mantengo il primo + gli ultimi 8, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare

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
    global prompt, vision_message, model_gemini, running, mode_groq_normal
    model_moondream= 'moondream' 
    while running:
        if shared_state["disactive_vad"]:
            await asyncio.sleep(1)
            continue
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(None, get_screenshot)
        if mode_groq_normal:
            prompt_visione = "Fai una brevissima descrizione generale di ciò che vedi sullo schermo."
        else:
            prompt_visione = "Analizza lo schermo (GeoGuessr): cerca scritte, pali della luce, tipo di asfalto e vegetazione per indovinare il paese. Sii molto specifico."
        part_img = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
        res_text = (await gemini_client.aio.models.generate_content(model=model_gemini, contents=[part_img, prompt_visione])).text
        vision_message = res_text.strip()
        await asyncio.sleep(10)


async def overlay_task(overlay):
    while True:
        overlay.root.update()
        await asyncio.sleep(0.05)

def azione_f10(window_trasparent, loop):
    global subtitles_active 
    subtitles_active = not subtitles_active
    sd.stop()
    loop.call_soon_threadsafe(window_trasparent.toggle_subtitles, subtitles_active)

def azione_f9():
    global history, message_groq_openguessr, message_groq_normal, mode_groq_normal
    mode_groq_normal = not mode_groq_normal
    if(mode_groq_normal):
        history = message_groq_normal.copy()
    else:
        history = message_groq_openguessr.copy()
    
async def main():
    #print(sd.query_devices())
    global subtitles_active
    input_queue = asyncio.Queue()
    window_trasparent = sub.SubtitleOverlay()
    window_trasparent.toggle_subtitles(subtitles_active)
    loop = asyncio.get_running_loop()
    keyboard.add_hotkey('f10', lambda:azione_f10(window_trasparent, loop))
    keyboard.add_hotkey('f9', lambda:azione_f9())
    await asyncio.gather(chat_groq_task(input_queue, window_trasparent), vision_gemini_task(), listen_task(input_queue), overlay_task(window_trasparent))

asyncio.run(main())