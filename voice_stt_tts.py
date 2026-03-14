from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/voice", methods=['POST'])
def voice_call():
    """Handle incoming call with TTS and STT"""
    resp = VoiceResponse()
    
    # TTS: Speak to the caller
    resp.say("Hello! Please tell me how I can help you.")
    
    # STT: Record and transcribe what caller says
    resp.record(
        transcribe=True,
        transcribe_callback="/transcription",
        max_length=30,
        play_beep=True
    )
    
    return str(resp)

@app.route("/transcription", methods=['POST'])
def transcription():
    """Receive the transcribed text from caller"""
    transcription_text = request.form.get('TranscriptionText', '')
    print(f"Caller said: {transcription_text}")
    
    # You can process the transcription here
    # and respond accordingly
    
    return '', 200

if __name__ == "__main__":
    app.run(debug=True)
