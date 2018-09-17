import json
import random

def nearest_rebuttal(response, data, default_response="learn the game"):
    response = list(filter(str.isalpha, map(str.lower, response.split())))

    best_idx = -1
    best_score = -1
    for i, interaction in enumerate(data):
        matches = 0
        score = 0
        for w in interaction['context']:
            matches += response.count(w)
        score = matches / (len(response) + 1)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx >= 0:
        return data[best_idx]['response']
    else:
        return default_response


with open('./assignment-b/posts.json') as f:
    initial_posts = json.load(f)
    print(random.choice(initial_posts))

with open('./assignment-b/responses.json') as f:
    data = json.load(f)
    while(True):
        response = input('Fan response: ')
        print("Troll response: ", nearest_rebuttal(response, data))


