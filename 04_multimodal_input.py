from base64 import b64encode

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model='gpt-4.1-mini')

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image', 'url': 'https://neuralnine.com/wp-content/uploads/2025/04/neuralnine_logo_transparent-1536x740.png',},
        {
            'type': 'image',
            'base64': b64encode(open('logo.png', 'rb').read()).decode(),
            'mime_type': 'image/png',
        },
    ],
}

response = model.invoke([message])

print(response.content)