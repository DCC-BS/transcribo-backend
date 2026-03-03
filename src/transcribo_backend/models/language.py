"""
Language Definition Models

This module defines the supported languages for summary services
and related language detection functionality.
"""

from __future__ import annotations

from enum import Enum


class Language(Enum):
    """
    Enumeration of supported languages for summary generation.

    Each language is represented by its IETF language tag.
    """

    AF = "af"  # Afrikaans
    AR = "ar"  # Arabic
    BG = "bg"  # Bulgarian
    BN = "bn"  # Bengali
    CA = "ca"  # Catalan
    CS = "cs"  # Czech
    CY = "cy"  # Welsh
    DA = "da"  # Danish
    DE = "de"  # German
    EL = "el"  # Greek
    EN = "en"  # English
    EN_GB = "en-gb"  # English (United Kingdom)
    EN_US = "en-us"  # English (United States)
    ES = "es"  # Spanish
    ET = "et"  # Estonian
    FA = "fa"  # Persian
    FI = "fi"  # Finnish
    FR = "fr"  # French
    GU = "gu"  # Gujarati
    HE = "he"  # Hebrew
    HI = "hi"  # Hindi
    HR = "hr"  # Croatian
    HU = "hu"  # Hungarian
    ID = "id"  # Indonesian
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KN = "kn"  # Kannada
    KO = "ko"  # Korean
    LT = "lt"  # Lithuanian
    LV = "lv"  # Latvian
    MK = "mk"  # Macedonian
    ML = "ml"  # Malayalam
    MR = "mr"  # Marathi
    NE = "ne"  # Nepali
    NL = "nl"  # Dutch
    NO = "no"  # Norwegian
    PA = "pa"  # Punjabi
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    RO = "ro"  # Romanian
    RU = "ru"  # Russian
    SK = "sk"  # Slovak
    SL = "sl"  # Slovenian
    SO = "so"  # Somali
    SQ = "sq"  # Albanian
    SV = "sv"  # Swedish
    SW = "sw"  # Swahili
    TA = "ta"  # Tamil
    TE = "te"  # Telugu
    TH = "th"  # Thai
    TL = "tl"  # Filipino
    TR = "tr"  # Turkish
    UK = "uk"  # Ukrainian
    UR = "ur"  # Urdu
    VI = "vi"  # Vietnamese
    ZH_CN = "zh-cn"  # Chinese (Simplified)
    ZH_TW = "zh-tw"  # Chinese (Traditional)


_LANGUAGE_NAMES = {
    Language.AF: "Afrikaans",
    Language.AR: "Arabic",
    Language.BG: "Bulgarian",
    Language.BN: "Bengali",
    Language.CA: "Catalan",
    Language.CS: "Czech",
    Language.CY: "Welsh",
    Language.DA: "Danish",
    Language.DE: "German",
    Language.EL: "Greek",
    Language.EN: "English",
    Language.EN_GB: "English (United Kingdom)",
    Language.EN_US: "English (United States)",
    Language.ES: "Spanish",
    Language.ET: "Estonian",
    Language.FA: "Persian",
    Language.FI: "Finnish",
    Language.FR: "French",
    Language.GU: "Gujarati",
    Language.HE: "Hebrew",
    Language.HI: "Hindi",
    Language.HR: "Croatian",
    Language.HU: "Hungarian",
    Language.ID: "Indonesian",
    Language.IT: "Italian",
    Language.JA: "Japanese",
    Language.KN: "Kannada",
    Language.KO: "Korean",
    Language.LT: "Lithuanian",
    Language.LV: "Latvian",
    Language.MK: "Macedonian",
    Language.ML: "Malayalam",
    Language.MR: "Marathi",
    Language.NE: "Nepali",
    Language.NL: "Dutch",
    Language.NO: "Norwegian",
    Language.PA: "Punjabi",
    Language.PL: "Polish",
    Language.PT: "Portuguese",
    Language.RO: "Romanian",
    Language.RU: "Russian",
    Language.SK: "Slovak",
    Language.SL: "Slovenian",
    Language.SO: "Somali",
    Language.SQ: "Albanian",
    Language.SV: "Swedish",
    Language.SW: "Swahili",
    Language.TA: "Tamil",
    Language.TE: "Telugu",
    Language.TH: "Thai",
    Language.TL: "Filipino",
    Language.TR: "Turkish",
    Language.UK: "Ukrainian",
    Language.UR: "Urdu",
    Language.VI: "Vietnamese",
    Language.ZH_CN: "Chinese (Simplified)",
    Language.ZH_TW: "Chinese (Traditional)",
}

LanguageOrAuto = Language | None


def get_language_name(language: LanguageOrAuto) -> str:
    """
    Get the human-readable name for a given language code.

    Args:
        language: A Language enum member or None for auto-detect

    Returns:
        The full name of the language (e.g., "English", "German")
        or "auto-detected" if language is None

    Raises:
        ValueError: If the language code is not recognized

    Examples:
        >>> get_language_name(Language.EN)
        'English'
        >>> get_language_name(None)
        'auto-detected'
    """
    if language is None:
        return "auto-detected"

    name = _LANGUAGE_NAMES.get(language)
    if name is None:
        raise ValueError("Unknown language") from None  # noqa: TRY003

    return name
