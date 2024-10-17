def removearticles(text):


    articles = {'a': '', 'an':'', 'der':'', 'Stadt':''}
    rest = []
    for word in text.split():
        if word not in articles:
            rest.append(word)
    return ' '.join(rest)

TEXT = "Stadt Mülheim an der Ruhr"

print(removearticles(TEXT))