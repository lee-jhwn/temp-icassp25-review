""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_puncs = '!. '
_ipa = 'abdefhijklmnopstuvwzæðŋɐɑɔəɚɛɜɡɪɹɾʃʊʌʒʔˈˌː̩θᵻ'
_sos = '<s>'
_eos = '</s>'

# Export all symbols:
symbols = [_pad, _sos, _eos] + list(_puncs) + list(_ipa) #+ list(_letters_ipa)

print(f'total {len(symbols)} symbols')

# Special symbol ids
SPACE_ID = symbols.index(" ")
