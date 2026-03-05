Ollama e Gemini Visual Assistant
Questo progetto nasce dall'idea di creare un assistente virtuale capace di "vedere" quello che succede sul mio schermo e interagire vocalmente in tempo reale. È un lavoro in continua evoluzione dove ho sperimentato diverse tecnologie per bilanciare prestazioni locali e velocità delle API.

Ho iniziato il progetto con l'obiettivo di far girare tutto in locale, scontrandomi però con i limiti hardware e di compatibilità del software.

1. Fase Locale Ollama e Moondream
	Visione: Ho utilizzato Moondream tramite Ollama per descrivere le immagini dello schermo. Il modello trasforma l'input visivo in 	testo che viene poi elaborato dal modulo di chat.

	Memoria: Inizialmente il bot non aveva memoria. Ho implementato una history dei messaggi per permettere al modello di mantenere il 	contesto della conversazione.

	Ottimizzazione: Per rendere l'interazione più fluida, ho attivato lo streaming della risposta e utilizzato la libreria asyncio per 	gestire i loop senza bloccare il programma.

2. Il passaggio alle API Gemini e Groq
A causa delle limitazioni hardware del mio PC, il calcolo locale era troppo lento per un'interazione naturale.
	Ho integrato le API di Gemini per la parte di visione e chat.

	Ho testato anche Groq con il modello Gemma per testarne la reattività.

3. Audio e Voice Interaction
	Speech-to-Text: Per il rilevamento della voce ho usato Faster Whisper. Per migliorare la precisione, ho aggiunto un sistema di 	Voice Activity Detection (VAD) che rileva quando sto effettivamente parlando, evitando di inviare rumore bianco al modello.

	Text-to-Speech (TTS): Ho cercato di implementare un sistema di clonazione voce tramite RVC e OpenVoice v2. Tuttavia, a causa di 	incompatibilità dei driver GPU su Windows e cambiamenti nel modo in cui il sistema leggeva il codice, ho ripiegato su Edge TTS per 	una maggiore stabilità.

	Feature: Ho aggiunto un tasto di "Mute" che permette di leggere i sottotitoli generati dal bot senza riprodurre l'audio.

Problemi Riscontrati
Sviluppare questo assistente su Windows ha presentato diverse sfide, specialmente nella gestione dell'audio e dei driver GPU per i modelli più vecchi come RVC. Queste difficoltà mi hanno spinto a esplorare soluzioni ibride tra locale e cloud per mantenere il progetto utilizzabile.

Link usati:
https://github.com/ollama/ollama-python
https://www.youtube.com/watch?v=rh7JJEfdwVk
https://ollama.com/library/moondream
https://github.com/vikhyat/moondream?tab=readme-ov-file 
https://github.com/m87-labs/moondream-examples/tree/main/quickstart/modal
https://www.youtube.com/watch?v=6PJBETNsxDk
https://pypi.org/project/faster-whisper/0.3.0/
https://github.com/SYSTRAN/faster-whisper
https://github.com/spatialaudio/python-sounddevice/blob/master/examples/play_stream.py
https://github.com/snakers4/silero-vad
https://github.com/rany2/edge-tts/blob/master/examples/async_audio_streaming_with_predefined_voice_and_subtitles.py
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
https://github.com/JarodMica/tts-pipeline-example/blob/master/test_system_voice.py
https://github.com/R3gm/infer_rvc_python
https://github.com/w-okada/voice-changer
https://www.youtube.com/watch?v=LX5en3pZJwM
