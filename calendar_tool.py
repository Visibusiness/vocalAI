import os.path
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopul permisiunilor (Read/Write)
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_service():
    """Autentificare și conectare la Google Calendar"""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)

def create_appointment(summary, start_time_iso):
    """
    Creează evenimentul.
    start_time_iso trebuie să fie string: '2023-10-27T15:00:00'
    """
    try:
        service = get_service()
        
        # Calculăm ora de sfârșit (automat +1 oră)
        start_dt = datetime.datetime.fromisoformat(start_time_iso)
        end_dt = start_dt + datetime.timedelta(hours=1)
        end_time_iso = end_dt.isoformat()

        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time_iso,
                'timeZone': 'Europe/Bucharest',
            },
            'end': {
                'dateTime': end_time_iso,
                'timeZone': 'Europe/Bucharest',
            },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"✅ Eveniment creat: {event.get('htmlLink')}")
        return True
    except Exception as e:
        print(f"❌ Eroare Calendar: {e}")
        return False

# Test rapid (rulează acest fișier direct ca să generezi token.json)
if __name__ == "__main__":
    print("Testare conexiune calendar...")
    # Creează un eveniment de test peste 24 ore
    maine = (datetime.datetime.now() + datetime.timedelta(days=1)).replace(microsecond=0).isoformat()
    create_appointment("Test Visi Programare", maine)