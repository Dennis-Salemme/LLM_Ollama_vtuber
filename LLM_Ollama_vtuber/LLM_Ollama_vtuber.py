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

history = []
prompt = ""                                                                      #il modello sarebbe meglio farlo in modo asincrono altrimenti se il modello deve capire di fare un screenshot da quello che deco sarebbe troppo lento
message=[{'role': 'user', 'content': '''Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio. 
Riceverai descrizioni dello schermo in inglese tra parentesi [ ], usale per commentare il gioco.'''}]      #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message
vision_message = "Non c'è nessuna descrizione per adesso"

async def chat_ollama_task():
    global history, prompt, vision_message
    model_llama = 'llama3'  
    while True:
        loop = asyncio.get_event_loop()                             #in questo modo non si dovrebbe fermare tutto il programma
        prompt = await loop.run_in_executor(None, input, "You: ")                                   #Inserisco quello che scrivo(poi farò quello che dico) in più noto che il input cosi ferma il programma nonostante il async
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            print("Arrivederci")
            break

        full_prompt = f"[{vision_message}] {prompt}"
        history.append({'role': 'user', 'content': full_prompt})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)

        print("Bot: ", end="", flush=True)                               
        bot_mc = ""
        response = chat(model=model_llama, messages=history,  stream=True)                     #faccio in modo che il bot mi rispoda parola per parola cosi non aspetto troppo per un messaggio
        for chunk in response:
            bot_mc += chunk.message.content
            print(chunk.message.content, end="", flush=True)
        print()                                                           

        history.append({'role': 'assistant', 'content': bot_mc})

        if len(history) > 6:
            history = [history[0]] + history[-5:]                          # Mantieni il primo (System) + gli ultimi 6, perchè se mantenesse tutti i messaggi non riuscirebbe più a capire cosa fare



async def vision_moondream_task():
    global prompt, vision_message
    model_moondream= 'moondream' 
    client = AsyncClient()
    while True:
        if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
            break
        image = open("C:\\Users\\salem\\Pictures\\Screenshots\\Screenshot.png", "rb")

        vision_message = ollama.generate(model=model_moondream, prompt="Describe what's happening in this screenshot.",images=[image.read()])['response']
        #print(vision_message)
        await asyncio.sleep(15)

async def main():
    await asyncio.gather(chat_ollama_task(), vision_moondream_task())

#inizio del main di quello che interessa
asyncio.run(main())