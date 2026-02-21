Progetto Ollama visual model
Ho voluto usare Ollama perché lo vollevo in localmente
Ollama serviva per generare il testo per farli vedere lo schermo usavo moondream che fa parte sempre di Ollama ma mi descrive la immagine in modo che poi glielo mandavo per testo.
uno dei problemi trovati e che il bot non si ricordava di quello che avevo scritto poco dopo quindi ho aggiunto una history in modo che gli mandi anche i messaggi prima cosi che abbia una memoria un po limitata ma va bene.
su Ollama per renderlo un po più reattivo nel testo ho attivato la stream in modo che mi arrivavano le parole subito e non debba aspettare che il modello abbia finito di generare tutta la frase.
per farli vedere lo schermo ho voluto usare moondream in modo che anche se è una descrizione della immagine capisce quello che vede all'incirca, il punto è che dovevo girare insieme al Ollama altrimente sarebbe stato troppo lento.
aggiungendo la liberia asyncio fatta apposta per i loop andava ma ho notato che era ancora troppo lento nel generare la risposta e anche nel descrivere la immagine a causa della limitazione per il mio pc, quindi ho voluto passare ad una api 
ho integrato la chat e la visual con la api di gemini, provo a fare la funzione che rileva quando parlo usando faster whisper e ho notato che il modello base non capisce molto bene quello che dico quindi provo ad usare un modello migliore, in più mi serve mettere un rilevamento di quando parlo perché se lo faccio a intermitenza è possibile che non mi prenda una parte della frase.

Link usati:https://github.com/ollama/ollama-python,
https://www.youtube.com/watch?v=rh7JJEfdwVk,
https://ollama.com/library/moondream,
https://github.com/vikhyat/moondream?tab=readme-ov-file, 
https://github.com/m87-labs/moondream-examples/tree/main/quickstart/modal,
https://www.youtube.com/watch?v=6PJBETNsxDk,
