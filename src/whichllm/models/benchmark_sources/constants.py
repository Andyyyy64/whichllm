from __future__ import annotations

import re

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>', re.DOTALL
)

_RSC_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[\d+,"(.*?)"\]\)', re.DOTALL)
