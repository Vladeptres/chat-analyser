from chat_analyser import core

messages = [
    {"user": "Bob", "content": "Hello"},
    {"user": "Alice", "content": "Hi"},
    {"user": "Bob", "content": "I'm soooo wasted this night is super fun"},
    {"user": "Alice", "content": "I want more DICKS in my life, I'm feeling so horny"},
    {"user": "Bob", "content": "GIMMEEEEE DRUUUUGS"},
    {"user": "Alice", "content": "I love my dress, everyone can see my boobies"},
    {"user": "Bob", "content": "I CAN seee elephants LMFAO"},
    {"user": "Alice", "content": "Just did a bathroom check with 2 guys hehe slurp"},
]
users = ["Bob", "Alice"]

print(core.analyser.analyse_chat("party", messages, users))
