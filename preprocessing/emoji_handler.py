import emoji

def convert_emojis(text):
    return emoji.demojize(text)

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')