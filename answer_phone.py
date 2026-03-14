from flask import Flask
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_monkey():
    resp = VoiceResponse()
    resp.say("Hello from your pals at Twilio! Have fun.")
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
