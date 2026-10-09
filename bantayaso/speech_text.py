"""Speech-only formatting: retain UI text, speak clocks and ranges naturally."""
import re

SMALL = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()

def number(n):
    if n < 20:
        return SMALL[n]
    tens = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty'}
    return tens[n // 10] + (' ' + SMALL[n % 10] if n % 10 else '')

CLOCK = r'(?<![\d:])([01]?\d|2[0-3]):([0-5]?\d)(?:\s*([ap]\.?m\.?))?(?![\d:])'

def clock_words(match):
    h, m = int(match[1]), int(match[2])
    period = (match[3] or '').lower().replace('.', '')
    # Explicit 24-hour clocks convey a period; bare 1..12 clocks do not.
    if not period and (h == 0 or h > 12):
        period = 'am' if h < 12 else 'pm'
    h = h % 12 or 12
    result = number(h) + (" o'clock" if m == 0 else ' oh ' + number(m) if m < 10 else ' ' + number(m))
    return result + (' A M' if period == 'am' else ' P M' if period == 'pm' else '')

def spoken_text(text):
    text = re.sub(r'[*`#]+', '', text)
    # Range separators must be converted before individual clocks.
    text = re.sub(r'(\d{1,2}:[0-5]?\d(?:\s*[ap]\.?m\.?)?)\s*[-–—]\s*(?=\d{1,2}:)', r'\1 to ', text, flags=re.I)
    text = re.sub(r'\bsince\s+(\d{1,2}:[0-5]?\d(?:\s*[ap]\.?m\.?)?\s+to\s+\d{1,2}:)', r'from \1', text, flags=re.I)
    text = re.sub(CLOCK, clock_words, text, flags=re.I)
    text = re.sub(r'\s+[-–—]\s+', '. ', text)
    return re.sub(r'\s+', ' ', text).strip()
