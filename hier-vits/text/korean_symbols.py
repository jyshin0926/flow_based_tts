# coding: utf-8
# Code based on
from beartype.typing import List


def get_symbols(pad: str = '_', eos: str = '~') -> List[str]:
    _punc = '!,.?'
    _space = ' '
    # _silences = ['sp', 'spn', 'sil']
    _silences = ['<brk>']
    _jamo_initial = "".join([chr(_) for _ in range(0x1100, 0x1113)])
    _jamo_middle = "".join([chr(_) for _ in range(0x1161, 0x1176)])
    _jamo_final = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

    # if symbol_set == 'radtts':
    korean_character = _jamo_initial + _jamo_middle + _jamo_final

    return list(pad + eos + korean_character + _punc + _space) + _silences

symbols = get_symbols()

