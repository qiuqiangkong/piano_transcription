import re
import numpy as np


'''
class BaseTokenizer:
    def __init__(self, strings, vocab_size=None):
        
        self.strings = strings
        
        if vocab_size:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(self.strings)

        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}
        
    def itos(self, token):
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        if string in self.strings:
            return self.string_to_token[string]
'''
class BaseTokenizer:
    def __init__(self, strings):
        
        self.strings = strings
        self.vocab_size = len(self.strings)

        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}
        
    def itos(self, token):
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        if string in self.strings:
            return self.string_to_token[string]


class SpecialTokenizer(BaseTokenizer):
    def __init__(self):
        strings = ["<pad>", "<sos>", "<eos>", "<unk>"]
        BaseTokenizer.__init__(self, strings)
        

class NameTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "note_on", "note_off", "note_sustain",
            "pedal_on", "pedal_off", "pedal_sustain",
            "beat", "downbeat",
        ]
        strings = ["name={}".format(s) for s in strings]
        BaseTokenizer.__init__(self, strings)
        self.vocab_size = 100


class TimeTokenizer:
    def __init__(self):
        self.vocab_size = 6001
        self.frames_per_second = 100

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        time = token / self.frames_per_second
        string = "time={}".format(time)
        return string

    def stoi(self, string):
        if "time=" in string:
            time = float(re.search('time=(.*)', string).group(1))
            token = round(time * self.frames_per_second)
            return token

'''
class MidiProgramTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "inst={}".format(token)
        return string

    def stoi(self, string):
        if "inst=" in string:
            token = int(re.search('inst=(.*)', string).group(1))
            return token
'''

class MaestroLabelTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "Piano",
        ]
        strings = ["label=maestro-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class Slakh2100LabelTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ethnic",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Strings (continued)",
            "Synth Lead",
            "Synth Pad"
        ]
        strings = ["label=slakh2100-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class GtzanLabelTokenizer(BaseTokenizer):
    def __init__(self):

        strings = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
            "metal", "pop", "reggae", "rock"]

        strings = ["label=gtzan-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class PitchTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "pitch={}".format(token)
        return string

    def stoi(self, string):
        if "pitch=" in string:
            token = int(re.search('pitch=(.*)', string).group(1))
            return token


class VelocityTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "velocity={}".format(token)
        return string

    def stoi(self, string):
        if "velocity=" in string:
            token = int(re.search('velocity=(.*)', string).group(1))
            return token


class BeatTokenizer:
    def __init__(self):
        self.vocab_size = 16

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "beat_index={}".format(token)
        return string

    def stoi(self, string):
        if "beat_index=" in string:
            token = int(re.search('beat_index=(.*)', string).group(1))
            return token


class TaskTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "onset", "offset", "velocity"
        ]
        strings = ["task={}".format(s) for s in strings]
        BaseTokenizer.__init__(self, strings)
        self.vocab_size = 100


class Tokenizer:
    def __init__(self, verbose=False):
        self.tokenizers = [
            SpecialTokenizer(),
            NameTokenizer(), 
            TimeTokenizer(), 
            # MidiProgramTokenizer(),
            MaestroLabelTokenizer(),
            Slakh2100LabelTokenizer(),
            GtzanLabelTokenizer(),
            PitchTokenizer(),
            VelocityTokenizer(),
            BeatTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        if verbose:
            print("Vocab size: {}".format(self.vocab_size))
            for tokenizer in self.tokenizers:
                print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))

    def strings_to_tokens(self, strings):

        tokens = []

        for string in strings:
            tokens.append(self.stoi(string))

        return tokens

    def tokens_to_strings(self, tokens):

        strings = []

        for token in tokens:
            strings.append(self.itos(token))

        return strings


class Tokenizer2:
    def __init__(self, verbose=False):
        self.tokenizers = [
            SpecialTokenizer(),
            NameTokenizer(), 
            TimeTokenizer(), 
            # MidiProgramTokenizer(),
            MaestroLabelTokenizer(),
            Slakh2100LabelTokenizer(),
            GtzanLabelTokenizer(),
            PitchTokenizer(),
            VelocityTokenizer(),
            BeatTokenizer(),
            TaskTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        if verbose:
            print("Vocab size: {}".format(self.vocab_size))
            for tokenizer in self.tokenizers:
                print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))

    def strings_to_tokens(self, strings):

        tokens = []

        for string in strings:
            tokens.append(self.stoi(string))

        return tokens

    def tokens_to_strings(self, tokens):

        strings = []

        for token in tokens:
            strings.append(self.itos(token))

        return strings


def test():

    tokenizer = SpecialTokenizer()
    tokenizer = Tokenizer(verbose=True)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    test()