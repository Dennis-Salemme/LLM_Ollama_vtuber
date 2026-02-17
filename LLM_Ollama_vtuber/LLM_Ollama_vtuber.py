from email import message
from ollama import chat

history = []
model_llama = 'llama3'                                                 #modello ollama3
model_moondream='moondream'                                            #modello moondream
message=[{'role': 'user', 'content': 'Rispondi ESCLUSIVAMENTE in italiano. Rispondi sempre allultimo messaggio'}]      #la domannda che terro sempre all'inizio in modo che il bot si ricordi cosa deve fare
history = message


while True:
    prompt = input("You: ").strip()                                   #Inserisco quello che scrivo(poi farò quello che dico)
    if (prompt.lower() == 'esci'):                                   #lo faccio uscire quando scrivo esci
        print("Arrivederci")
        break

    history.append({'role': 'user', 'content': prompt})                                          #mi tengo una storia dei messaggi in modo che il bot e i miei in modo si ricordi i messaggi vecchi(forse mettere un massimo di ricordo es max 3)

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