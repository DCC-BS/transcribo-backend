"""
Language Definition Models

This module defines the supported languages for summary services
and related language detection functionality.
"""

from __future__ import annotations

from enum import Enum


class Language(Enum):
    display_name: str

    def __new__(cls, code: str, display_name: str) -> Language:
        obj = object.__new__(cls)
        obj._value_ = code
        obj.display_name = display_name
        return obj

    AF = ("af", "Afrikaans")
    AM = ("am", "Amharic")
    AR = ("ar", "Arabic")
    AS = ("as", "Assamese")
    AZ = ("az", "Azerbaijani")
    BA = ("ba", "Bashkir")
    BE = ("be", "Belarusian")
    BG = ("bg", "Bulgarian")
    BN = ("bn", "Bengali")
    BO = ("bo", "Tibetan")
    BR = ("br", "Breton")
    BS = ("bs", "Bosnian")
    CA = ("ca", "Catalan")
    CS = ("cs", "Czech")
    CY = ("cy", "Welsh")
    DA = ("da", "Danish")
    DE = ("de", "German")
    EL = ("el", "Greek")
    EN = ("en", "English")
    ES = ("es", "Spanish")
    ET = ("et", "Estonian")
    EU = ("eu", "Basque")
    FA = ("fa", "Persian")
    FI = ("fi", "Finnish")
    FO = ("fo", "Faroese")
    FR = ("fr", "French")
    GL = ("gl", "Galician")
    GU = ("gu", "Gujarati")
    HA = ("ha", "Hausa")
    HAW = ("haw", "Hawaiian")
    HE = ("he", "Hebrew")
    HI = ("hi", "Hindi")
    HR = ("hr", "Croatian")
    HT = ("ht", "Haitian Creole")
    HU = ("hu", "Hungarian")
    HY = ("hy", "Armenian")
    ID = ("id", "Indonesian")
    IS = ("is", "Icelandic")
    IT = ("it", "Italian")
    JA = ("ja", "Japanese")
    JW = ("jw", "Javanese")
    KA = ("ka", "Georgian")
    KK = ("kk", "Kazakh")
    KM = ("km", "Khmer")
    KN = ("kn", "Kannada")
    KO = ("ko", "Korean")
    LA = ("la", "Latin")
    LB = ("lb", "Luxembourgish")
    LN = ("ln", "Lingala")
    LO = ("lo", "Lao")
    LT = ("lt", "Lithuanian")
    LV = ("lv", "Latvian")
    MG = ("mg", "Malagasy")
    MI = ("mi", "Maori")
    MK = ("mk", "Macedonian")
    ML = ("ml", "Malayalam")
    MN = ("mn", "Mongolian")
    MR = ("mr", "Marathi")
    MS = ("ms", "Malay")
    MT = ("mt", "Maltese")
    MY = ("my", "Burmese")
    NE = ("ne", "Nepali")
    NL = ("nl", "Dutch")
    NN = ("nn", "Norwegian Nynorsk")
    NO = ("no", "Norwegian")
    OC = ("oc", "Occitan")
    PA = ("pa", "Punjabi")
    PL = ("pl", "Polish")
    PS = ("ps", "Pashto")
    PT = ("pt", "Portuguese")
    RO = ("ro", "Romanian")
    RU = ("ru", "Russian")
    SA = ("sa", "Sanskrit")
    SD = ("sd", "Sindhi")
    SI = ("si", "Sinhala")
    SK = ("sk", "Slovak")
    SL = ("sl", "Slovenian")
    SN = ("sn", "Shona")
    SO = ("so", "Somali")
    SQ = ("sq", "Albanian")
    SR = ("sr", "Serbian")
    SU = ("su", "Sundanese")
    SV = ("sv", "Swedish")
    SW = ("sw", "Swahili")
    TA = ("ta", "Tamil")
    TE = ("te", "Telugu")
    TG = ("tg", "Tajik")
    TH = ("th", "Thai")
    TK = ("tk", "Turkmen")
    TL = ("tl", "Filipino")
    TR = ("tr", "Turkish")
    TT = ("tt", "Tatar")
    UK = ("uk", "Ukrainian")
    UR = ("ur", "Urdu")
    UZ = ("uz", "Uzbek")
    VI = ("vi", "Vietnamese")
    YI = ("yi", "Yiddish")
    YO = ("yo", "Yoruba")
    YUE = ("yue", "Cantonese")
    ZH = ("zh", "Chinese")


LanguageOrAuto = Language | None


def get_language_name(language: LanguageOrAuto) -> str:
    if language is None:
        return "auto-detected"
    return language.display_name
