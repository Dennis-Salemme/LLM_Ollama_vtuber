from email import message
from pyexpat import model
from time import sleep
from ollama import chat
import ollama
import moondream as md
from PIL import Image
import mss
import os
import asyncio
from ollama import AsyncClient
import io
import torch
from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd

history = []
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
message=[{'role': 'user', 'content': '''Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio. 
Riceverai descrizioni dello schermo in inglese tra parentesi [ ], usale per commentare il gioco.'''}]      #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message
vision_message = "Non c'è nessuna descrizione per adesso"
model_transcript_vocal = WhisperModel("small", device="cpu", compute_type="int8")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
running = True

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

async def chat_ollama_task(input_queue):
    global history, prompt, vision_message, running
    model_llama = 'llama3'  
    while True:
        #loop = asyncio.get_event_loop()                             #in questo modo non si dovrebbe fermare tutto il programma
        #prompt = await loop.run_in_executor(None, input, "You: ")                                   #Inserisco quello che scrivo(poi farò quello che dico) in più noto che il input cosi ferma il programma nonostante il async
        prompt = await input_queue.get()
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            running = False
            print("Arrivederci")
            break

        full_prompt = f"[{vision_message}] {prompt}"
        history.append({'role': 'user', 'content': full_prompt})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)

        print("Bot: ", end="", flush=True)                               
        bot_mc = ""
        response = await AsyncClient().chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        for chunk in response:
            bot_mc += chunk.message.content
            print(chunk.message.content, end="", flush=True)
        print()                                                           

        history.append({'role': 'assistant', 'content': bot_mc})

        if len(history) > 6:
            history = [history[0]] + history[-5:]                          # Mantieni il primo (System) + gli ultimi 6, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare



async def vision_moondream_task():
    global prompt, vision_message, running
    model_moondream= 'moondream' 
    while running:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            image = sct.grab(monitor)
        #image = open("image", "rb")

        vision_message = await AsyncClient().ollama.generate(model=model_moondream, prompt="Describe what's happening in this screenshot.",images=[image.read()])['response']
        #print(vision_message)
        await asyncio.sleep(15)

async def main():
    input_queue = asyncio.Queue()
    await asyncio.gather(chat_ollama_task(input_queue), vision_moondream_task(), listen_task(input_queue))

#inizio del main di quello che interessa
asyncio.run(main())