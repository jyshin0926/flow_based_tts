import re
import unicodedata

from g2pk import G2p
from jamo import hangul_to_jamo
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from genesis_speech.data.text.cleaners.korean import KoreanCleaner
# from genesis_speech.models.speech_synthesis.RADTTS.tts_text_processing.text_processing import TextProcessing
from .korean_symbols import get_symbols

# Mappings from symbol to numeric ID and vice versa:
_symbols = get_symbols()
_symbol_to_id = {s: i for i, s in enumerate(_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(_symbols)}
_curly_re = re.compile(r"\<brk\>")


class KoG2p(G2p):
    def __init__(self):
        super().__init__()
        self._word_tokenize = TweetTokenizer().tokenize
        self._g2p_ = G2p()

    def __call__(self, text: str):
        # preprocessing
        words = self._word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)
        # steps
        punc = [".", ",", "!", "?", "<brk>"]
        prons = []
        for word, pos in tokens:
            if re.search("[ㄱ-ㅎ가-힣]", word) is None:
                pron = [word]
            else:  # predict for oov
                pron = list(hangul_to_jamo(self._g2p_(word)))

            if len(pron) >= 1 and len(prons) >= 1 and pron[-1] in punc and prons[-1] == " ":
                del prons[-1]
                prons.extend(pron)
                prons.extend([" "])
            else:
                prons.extend(pron)
                prons.extend([" "])
        return prons[:-1]


def text_to_sequence(text, cleaner: KoreanCleaner, g2p_module: KoG2p):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[" "]]

    # start = time.time()
    clean_text = _clean_text(text, cleaner, g2p_module)
    # print("[total_cleaner]", time.time() - start)
    # for symbol in clean_text:
    #   symbol_id = _symbol_to_id[symbol]
    #   sequence += [symbol_id]

    start = 0
    matches = _curly_re.finditer(clean_text)
    for match in matches:
        sequence += cleaned_text_to_sequence(clean_text[start : match.start()])  # Add the symbols before <brk> token
        sequence += brk_symbol_to_sequence(match.group())  # Add the <brk> token
        start = match.end()

    # Add the remaining symbols after the last <brk> token
    sequence += cleaned_text_to_sequence(clean_text[start:])
    sequence.append(_symbol_to_id[" "])  # TODO:: 여기가 최종 리턴하는 부분이므로 체크필요

    return sequence


# TODO:: 정리 필요
def text_to_sequence_backup(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[" "]]

    clean_text = _clean_text(text, cleaner_names)
    # for symbol in clean_text:
    #   symbol_id = _symbol_to_id[symbol]
    #   sequence += [symbol_id]

    matches = _curly_re.finditer(clean_text)

    start = 0
    for match in matches:
        sequence += cleaned_text_to_sequence(clean_text[start : match.start()])  # Add the symbols before <brk> token
        sequence += brk_symbol_to_sequence(match.group())  # Add the <brk> token
        start = match.end()

    # Add the remaining symbols after the last <brk> token
    sequence += cleaned_text_to_sequence(clean_text[start:])
    sequence.append(_symbol_to_id[" "])  # TODO:: 여기가 최종 리턴하는 부분이므로 체크필요

    return sequence


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # cleaned_text = _clean_text(text, cleaner_names)
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def brk_symbol_to_sequence(self, symbol):
    return [self.symbol_to_id[symbol]]


# def sequence_to_text(sequence):
#   '''Converts a sequence of IDs back to a string'''
#   result = ''
#   for symbol_id in sequence:
#     s = _id_to_symbol[symbol_id]
#     result += s
#   return result


def _clean_text(text, cleaner: KoreanCleaner, g2p_module: KoG2p):
    # for name in cleaner_names:
    #   cleaner = getattr(cleaners, name)
    #   if not cleaner:
    #     raise Exception('Unknown cleaner: %s' % name)
    #   text = cleaner(text)
    # ko_cleaner = KoreanCleaner()
    # g2p = KoG2p()
    # start = time.time()
    text = text.strip("{}")
    text = text.replace("(", "").replace(")", "")
    text = cleaner.normalize(text, False)
    # print("[cleaner]", time.time() - start)

    # start = time.time()
    # text = unicodedata.normalize("NFD", text)
    text = unicodedata.normalize("NFC", text)
    phs = "".join(g2p_module(text))
    # print("[g2p]", time.time() - start)

    return phs