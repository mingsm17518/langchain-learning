import requests
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1', timeout=10)
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'


model = init_chat_model('gpt-4.1-mini', temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt=(
        'You are a helpful weather assistant, who always cracks jokes '
        'and is humorous while remaining helpful.'
    ),
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

# 第一轮对话
config = {'configurable': {'thread_id': 1}}

response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'What is the weather like in Vienna?'},
        ]
    },
    config=config,
    context=Context(user_id='ABC123'),
)

print(response['structured_response'])
print(response['structured_response'].summary)


# 继续对话（thread_id 相同 -> 复用同一会话记忆）
config = {'configurable': {'thread_id': 1}}
# 若改成 thread_id=2，则无法继续上面的对话
# config = {'configurable': {'thread_id': 2}}
 
response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'And is this usual?'},
        ]
    },
    config=config,
    context=Context(user_id='ABC123'),
)

print(response['structured_response'])
print(response['structured_response'].summary)