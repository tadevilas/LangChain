from typing import TypedDict    

class Person(TypedDict):
    name : str
    age : int

new_P : Person = {'name':'vilas', 'age': 30}

print(new_P)