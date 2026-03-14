from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
import json

app = Flask(__name__)

# Store grievances (in production, use a database)
grievances = []

@app.route("/voice", methods=['POST'])
def start_call():
    """Initial greeting"""
    resp = VoiceResponse()
    gather = Gather(input='speech', action='/problem', timeout=5, speech_timeout='auto')
    gather.say("Hello, you have reached the grievance helpline. Please tell me what issue you are facing.")
    resp.append(gather)
    return str(resp)

@app.route("/problem", methods=['POST'])
def get_problem():
    """Collect the problem"""
    problem = request.form.get('SpeechResult', '')
    
    resp = VoiceResponse()
    resp.say("I'm sorry you're facing this issue. I'll help you report it.")
    
    gather = Gather(input='speech', action=f'/location?problem={problem}', timeout=5, speech_timeout='auto')
    gather.say("Could you please tell me the exact address or location where this issue is happening?")
    resp.append(gather)
    return str(resp)

@app.route("/location", methods=['POST'])
def get_location():
    """Collect the location"""
    problem = request.args.get('problem', '')
    location = request.form.get('SpeechResult', '')
    
    resp = VoiceResponse()
    gather = Gather(input='speech', action=f'/phone?problem={problem}&location={location}', timeout=5, speech_timeout='auto')
    gather.say("May I confirm your phone number so the authorities can contact you?")
    resp.append(gather)
    return str(resp)

@app.route("/phone", methods=['POST'])
def get_phone():
    """Collect phone and save grievance"""
    problem = request.args.get('problem', '')
    location = request.args.get('location', '')
    phone = request.form.get('SpeechResult', '')
    
    # Save grievance
    grievance = {
        'problem': problem,
        'location': location,
        'phone': phone
    }
    grievances.append(grievance)
    
    # Print formatted output
    print("\n" + "="*50)
    print("NEW GRIEVANCE RECORDED")
    print("="*50)
    print(f"Problem: {problem}")
    print(f"Location: {location}")
    print(f"Phone: {phone}")
    print("="*50 + "\n")
    
    # Save to file
    with open('grievances.json', 'a') as f:
        f.write(json.dumps(grievance) + '\n')
    
    resp = VoiceResponse()
    resp.say("Thank you. I will submit this complaint to the authorities. Your complaint has been recorded. Someone from the department may contact you soon. Goodbye.")
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
