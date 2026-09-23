from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
model = init_chat_model(model = 'gpt-4.1-mini', temperature = 0.1)

# response
response = model.invoke('Hello, what is Python?')
print(response)
print(response.content)


# stream
for chunk in model.stream('Hello, what is Python?'):
    print(chunk.text, end='', flush=True)


# conversation 
conversation =[
    SystemMessage('You are a helpful assistant for questions regarding programming'),
    HumanMessage('What is Python?'),
    AIMessage('Python is an interpreted programming language.'),
    HumanMessage('When was it released?'),
]
response = model.invoke(conversation)
print(response)
print(response.content)