from __future__ import annotations

import re
import unicodedata

from neorag.text import normalize, split_sentences, tokenize

MAIN_CLUBS = [
    "CLB Lập Trình PTIT",
    "CLB Multimedia PTIT",
    "CLB Nhà Sáng Tạo Game PTIT (GCC)",
]
S2B_CLUB = "Cộng đồng S2B"
ALL_CLUBS = [*MAIN_CLUBS, S2B_CLUB]

STOPWORDS = frozenset(
    [
        "và",
        "là",
        "của",
        "có",
        "cho",
        "trong",
        "một",
        "các",
        "được",
        "với",
        "để",
        "khi",
        "từ",
        "này",
        "đó",
        "những",
        "cái",
        "gì",
        "nào",
        "bao",
        "nhiêu",
        "như",
        "thế",
        "tại",
        "sao",
        "vì",
        "nên",
        "hay",
        "hoặc",
        "không",
        "phải",
        "đã",
        "đang",
        "sẽ",
        "bị",
        "bởi",
        "về",
        "theo",
        "trên",
        "dưới",
        "ra",
        "vào",
        "đến",
        "ở",
        "tới",
        "em",
        "mình",
        "bạn",
        "ạ",
        "vậy",
        "nhỉ",
        "thì",
        "mà",
        "clb",
        "câu",
        "lạc",
        "bộ",
    ]
)

_STRIP_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")
_CPP_PATTERN = re.compile(r"(?<!\w)c\+\+(?!\w)", flags=re.IGNORECASE)
_CSHARP_PATTERN = re.compile(r"(?<!\w)c#(?!\w)", flags=re.IGNORECASE)

_CLUB_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    (
        MAIN_CLUBS[0],
        (
            "proptit",
            "pro ptit",
            "programming ptit",
            "clb lap trinh",
            "lap trinh ptit",
            "clb lập trình",
            "clb lập trình ptit",
            "clb lap trinh ptit",
            "clb code",
            "clb lt",
            "lt ptit",
        ),
    ),
    (
        MAIN_CLUBS[1],
        (
            "multimedia",
            "multimedia ptit",
            "mpc",
            "mm",
            "clb mm",
            "ben mm",
            "da phuong tien",
            "đa phương tiện",
            "clb da phuong tien",
            "clb đa phương tiện",
            "clb multimedia",
            "clb multimedia ptit",
            "sang tao da phuong tien",
        ),
    ),
    (
        MAIN_CLUBS[2],
        (
            "gcc",
            "game creators",
            "nha sang tao game",
            "nhà sáng tạo game",
            "clb nha sang tao game",
            "clb game",
            "clb game ptit",
            "clb lam game",
            "clb làm game",
        ),
    ),
    (S2B_CLUB, ("s2b", "cong dong s2b", "cộng đồng s2b")),
]

_CLUB_LEXICAL_HEADER: dict[str, str] = {
    MAIN_CLUBS[0]: "CLB Lập Trình PTIT ProPTIT Programming PTIT clb lt clb lap trinh",
    MAIN_CLUBS[1]: "CLB Multimedia PTIT MPC Multimedia PTIT Club clb đa phương tiện mm",
    MAIN_CLUBS[2]: "CLB Nhà Sáng Tạo Game PTIT GCC Game Creators Club clb game clb làm game",
    S2B_CLUB: "Cộng đồng S2B S2B ba clb",
}

_SLANG_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("len plan chay chuong trinh", "lên kế hoạch tổ chức sự kiện"),
    ("chay chuong trinh", "tổ chức sự kiện"),
    ("pho cn", "phó chủ nhiệm"),
    ("bcn", "ban chủ nhiệm"),
    ("ttv", "tuyển thành viên"),
    ("dpt", "đa phương tiện"),
    ("khmt", "khoa học máy tính"),
    ("cntt", "công nghệ thông tin"),
    ("attt", "an toàn thông tin"),
    ("ctdl", "cấu trúc dữ liệu và giải thuật"),
    ("oop", "lập trình hướng đối tượng"),
    ("nk", "nhiệm kỳ"),
    ("sdt", "số điện thoại"),
    ("dk", "đăng ký"),
    ("hv", "học viện"),
)

_MULTI_MARKERS = (
    "so sanh",
    "so sánh",
    "khac nhau",
    "khác nhau",
    "giong nhau",
    "giống nhau",
    "lau doi",
    "lâu đời",
    "tre nhat",
    "trẻ nhất",
    "ca 3",
    "cả 3",
    "ca ba",
    "cả ba",
    "3 clb",
    "ba clb",
    "3 cau lac bo",
    "ba cau lac bo",
    "cac clb",
    "các clb",
    "nhung clb nao",
    "những clb nào",
    "clb nao trong s2b",
    "clb nào trong s2b",
    "trong s2b clb nao",
    "trong 3 clb",
    "trong ba clb",
)

_TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "Tuyển thành viên": (
        "tuyen",
        "tuyển",
        "ung tuyen",
        "ứng tuyển",
        "dang ky",
        "đăng ký",
        "mo don",
        "mở đơn",
        "gen ",
        "cv",
        "phong van",
        "phỏng vấn",
        "recruit",
    ),
    "Ban chủ nhiệm": (
        "chu nhiem",
        "chủ nhiệm",
        "pho chu nhiem",
        "phó chủ nhiệm",
        "pho cn",
        "ban chu nhiem",
        "ban chủ nhiệm",
        "bcn",
        "nhiem ky",
        "nhiệm kỳ",
        "nk ",
        "truong ban",
        "trưởng ban",
        "ban doi ngoai",
        "ban đối ngoại",
        "ban doi noi",
        "ban đối nội",
        "ban su kien",
        "ban sự kiện",
        "ban tai chinh",
        "ban tài chính",
        "ban nhan su",
        "ban nhân sự",
        "ban dao tao",
        "ban đào tạo",
        "ban truyen thong",
        "ban truyền thông",
    ),
    "Giải thưởng": (
        "giai thuong",
        "giải thưởng",
        "dat giai",
        "đạt giải",
        "vo dich",
        "vô địch",
        "quan quan",
        "quán quân",
        "huy chuong",
        "huy chương",
        "thanh tich",
        "thành tích",
        "icpc",
        "olympic",
    ),
    "Liên hệ": (
        "lien he",
        "liên hệ",
        "facebook",
        "fanpage",
        "email",
        "mail",
        "tiktok",
        "youtube",
        "instagram",
        "github",
        "website",
        "blog",
        "dia chi",
        "địa chỉ",
        "van phong",
        "văn phòng",
        "so dien thoai",
        "số điện thoại",
        "hotline",
        "sdt",
    ),
    "Phí & quỹ": ("hoc phi", "học phí", "chi phi", "chi phí", "quy", "quỹ", "mien phi", "miễn phí"),
    "Nội quy": (
        "noi quy",
        "nội quy",
        "quy dinh",
        "quy định",
        "nghia vu",
        "nghĩa vụ",
        "ky luat",
        "kỷ luật",
        "diem ren luyen",
        "điểm rèn luyện",
    ),
    "Sự kiện": (
        "su kien",
        "sự kiện",
        "hoat dong",
        "hoạt động",
        "cuoc thi",
        "cuộc thi",
        "workshop",
        "talkshow",
        "game jam",
        "projam",
        "coding fest",
        "big game",
        "biggame",
        "sinh nhat",
        "sinh nhật",
    ),
    "Giới thiệu & lịch sử": (
        "thanh lap",
        "thành lập",
        "ra doi",
        "ra đời",
        "lich su",
        "lịch sử",
        "truyen thong",
        "truyền thống",
        "gioi thieu",
        "giới thiệu",
        "phuong cham",
        "phương châm",
        "slogan",
        "viet tat",
        "viết tắt",
        "sang lap",
        "sáng lập",
    ),
    "Đào tạo": (
        "dao tao",
        "đào tạo",
        "hoc gi",
        "học gì",
        "mon hoc",
        "môn học",
        "lo trinh",
        "lộ trình",
        "c/c++",
        "c++",
        "c#",
        "java",
        "unity",
        "photoshop",
        "after effects",
        "spine",
        "cau truc du lieu",
        "cấu trúc dữ liệu",
        "thuat toan",
        "thuật toán",
    ),
    "Cơ cấu": (
        "co cau",
        "cơ cấu",
        "cac ban",
        "các ban",
        "ban nao",
        "ban nào",
        "nhung ban",
        "những ban",
        "ban rieng",
        "ban riêng",
        "ban noi dung",
        "ban nội dung",
        "ban media",
        "ban thiet ke",
        "ban thiết kế",
        "team",
        "cac team",
        "các team",
        "phong ban",
        "phòng ban",
    ),
    "Điều kiện tham gia": (
        "dieu kien",
        "điều kiện",
        "co dat",
        "có đạt",
        "phu hop",
        "phù hợp",
        "moi nganh",
        "mọi ngành",
        "nganh khac",
        "ngành khác",
        "truong khac",
        "trường khác",
        "nam 2",
        "năm 2",
        "nam hai",
        "năm hai",
        "khong biet ve",
        "không biết về",
        "chua biet",
        "chưa biết",
    ),
    "Quyền lợi": (
        "quyen loi",
        "quyền lợi",
        "loi ich",
        "lợi ích",
        "duoc gi",
        "được gì",
        "ho tro gi",
        "hỗ trợ gì",
        "hoc bong",
        "học bổng",
    ),
    "Quan hệ": (
        "truc thuoc",
        "trực thuộc",
        "quan he",
        "quan hệ",
        "lien ket",
        "liên kết",
        "thuoc cong dong",
        "thuộc cộng đồng",
        "bao tro",
        "bảo trợ",
    ),
    "Sản phẩm": (
        "san pham",
        "sản phẩm",
        "du an",
        "dự án",
        "lam ra",
        "làm ra",
        "itch.io",
    ),
}

_TOPIC_GROUPS: dict[str, tuple[str, ...]] = {
    "Tuyển thành viên": ("Tuyển thành viên",),
    "Ban chủ nhiệm": ("Ban chủ nhiệm",),
    "Giải thưởng": ("Giải thưởng", "Giải thưởng & sản phẩm"),
    "Liên hệ": ("Liên hệ",),
    "Phí & quỹ": ("Phí & quỹ",),
    "Nội quy": ("Nội quy & nghĩa vụ",),
    "Sự kiện": ("Sự kiện nội bộ", "Sự kiện đối ngoại", "Sự kiện chung", "Lĩnh vực & hoạt động"),
    "Giới thiệu & lịch sử": (
        "Giới thiệu chung",
        "Lịch sử & truyền thống",
        "Lịch sử & thành lập",
    ),
    "Đào tạo": ("Đào tạo", "Cơ cấu & đào tạo"),
    "Cơ cấu": ("Cơ cấu", "Cơ cấu & team dự án", "Cơ cấu & đào tạo"),
    "Điều kiện tham gia": ("Điều kiện tham gia",),
    "Quyền lợi": ("Quyền lợi",),
    "Quan hệ": (
        "Quan hệ giữa các CLB",
        "Trực thuộc",
        "So sánh & tham gia nhiều CLB",
        "Đối tác & bảo trợ",
    ),
    "Sản phẩm": ("Sản phẩm", "Giải thưởng & sản phẩm"),
}

_GENERIC_META_PREFIX = re.compile(
    r"^(?:Dựa\s+(?:trên|vào)|Căn\s+cứ\s+(?:trên|vào))\s+"
    r"(?:các\s+)?(?:đoạn\s+)?(?:thông\s+tin\s+(?:có\s+)?trong\s+)?"
    r"tài\s+liệu(?:\s+được\s+cung\s+cấp|\s+hiện\s+có|\s+trên)?\s*,\s*",
    flags=re.IGNORECASE,
)
_FALSE_YESNO_BEFORE_ABSTAIN = re.compile(
    r"^(?:Không|Có|Chưa|Không\s+có)\s*[.,;:]\s*"
    r"(?=(?:Tài\s+liệu|Hiện\s+tài\s+liệu|Kho\s+tài\s+liệu)\s+"
    r"(?:hiện\s+có\s+)?(?:không|chưa|chỉ)\b)",
    flags=re.IGNORECASE,
)
_BARE_YESNO_START = re.compile(r"^(?:Không|Có)\s*[.,;:]\s*", flags=re.IGNORECASE)

def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

ASCII_STOPWORDS = frozenset(strip_diacritics(w) for w in STOPWORDS)

def _preserve_tech_tokens(text: str) -> str:
    text = _CPP_PATTERN.sub(" cpp ", text)
    text = _CSHARP_PATTERN.sub(" csharp ", text)
    return text

def ascii_text(text: str) -> str:
    return _SPACES.sub(" ", strip_diacritics(normalize(text))).strip()

def words(text: str) -> list[str]:
    return tokenize(text)

def ascii_words(text: str) -> list[str]:
    return ascii_text(text).split()

def ascii_content_words_raw(text: str) -> list[str]:
    value = unicodedata.normalize("NFC", _preserve_tech_tokens(text)).lower()
    value = _STRIP_PUNCT.sub(" ", value)
    value = strip_diacritics(value)
    return [
        word
        for word in value.split()
        if (len(word) > 1 or word == "c" or word.isdigit()) and word not in ASCII_STOPWORDS
    ]

def accented_bm25_tokens(text: str) -> list[str]:
    value = _STRIP_PUNCT.sub(" ", _preserve_tech_tokens(normalize(text)))
    raw_tokens = [
        tok
        for tok in value.split()
        if (len(tok) > 1 or tok == "c" or tok.isdigit()) and tok not in STOPWORDS
    ]
    bigrams = [f"{raw_tokens[i]}_{raw_tokens[i + 1]}" for i in range(len(raw_tokens) - 1)]
    return [*raw_tokens, *bigrams]

def ascii_bm25_tokens(text: str) -> list[str]:
    raw_tokens = ascii_content_words_raw(text)
    bigrams = [f"{raw_tokens[i]}_{raw_tokens[i + 1]}" for i in range(len(raw_tokens) - 1)]
    return [*raw_tokens, *bigrams]

def club_lexical_aliases(club: str) -> str:
    return _CLUB_LEXICAL_HEADER.get(club, club)

def expand_student_slang(text: str) -> str:
    lowered = ascii_text(text)
    additions: list[str] = []
    for short, full in _SLANG_EXPANSIONS:
        if re.search(rf"(?<!\w){re.escape(short)}(?!\w)", lowered):
            additions.append(full)
    if not additions:
        return text
    return f"{text} ({', '.join(additions)})"

def content_words(text: str, ascii_only: bool = False) -> list[str]:
    if ascii_only:
        return [t for t in ascii_words(text) if len(t) > 1 and t not in ASCII_STOPWORDS]
    return [t for t in words(text) if len(t) > 1 and t not in STOPWORDS]

def char_ngrams(text: str, n: int = 3) -> list[str]:
    value = "_".join(ascii_words(text))
    return [value[i : i + n] for i in range(max(1, len(value) - n + 1))]

def has_diacritics(text: str) -> bool:
    return ascii_text(text).replace(" ", "") != normalize(text).replace(" ", "")

def detect_clubs(text: str) -> list[str]:
    lowered = ascii_text(text)
    found: list[str] = []
    for club, aliases in _CLUB_ALIASES:
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered) for alias in aliases):
            found.append(club)
    return found

def is_multi_club_question(text: str) -> bool:
    lowered = ascii_text(text)
    if any(marker in lowered for marker in _MULTI_MARKERS):
        return True
    return len([c for c in detect_clubs(text) if c in MAIN_CLUBS]) >= 2

def is_founding_comparison(text: str) -> bool:
    lowered = ascii_text(text)
    founding = any(x in lowered for x in ("lau doi", "tre nhat", "thanh lap", "ra doi"))
    return founding and is_multi_club_question(text)

def detect_topics(text: str) -> set[str]:
    lowered = ascii_text(text)
    return {topic for topic, keys in _TOPIC_RULES.items() if any(key in lowered for key in keys)}

def topic_labels(topics: set[str]) -> set[str]:
    labels: set[str] = set()
    for topic in topics:
        labels.update(_TOPIC_GROUPS.get(topic, ()))
    return labels

def clean_answer(text: str, max_chars: int = 2000) -> str:
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    text = re.sub(r"\(?\s*id\s*[:：]?\s*\[?\d+(?:\s*,\s*\d+)*\]?\s*\)?", "", text, flags=re.I)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•+]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    answer = _SPACES.sub(" ", " ".join(lines)).strip()
    for prefix in ("Câu trả lời cuối cùng:", "Câu trả lời:", "Trả lời:", "Đáp án:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()
    return answer[:max_chars]

def sanitize_answer(text: str, status: str = "ANSWERABLE", max_chars: int = 2000) -> str:
    ans = clean_answer(text, max_chars=max_chars * 2)
    ans = _GENERIC_META_PREFIX.sub("", ans).strip()
    ans = _FALSE_YESNO_BEFORE_ABSTAIN.sub("", ans).strip()
    if status.upper() == "UNANSWERABLE":
        ans = _BARE_YESNO_START.sub("", ans).strip()
    if ans and ans[0].islower():
        ans = ans[0].upper() + ans[1:]
    return clean_answer(ans, max_chars=max_chars)

__all__ = [
    "ALL_CLUBS",
    "ASCII_STOPWORDS",
    "MAIN_CLUBS",
    "S2B_CLUB",
    "accented_bm25_tokens",
    "ascii_bm25_tokens",
    "ascii_content_words_raw",
    "ascii_text",
    "ascii_words",
    "char_ngrams",
    "clean_answer",
    "club_lexical_aliases",
    "content_words",
    "detect_clubs",
    "detect_topics",
    "expand_student_slang",
    "has_diacritics",
    "is_founding_comparison",
    "is_multi_club_question",
    "normalize",
    "sanitize_answer",
    "split_sentences",
    "strip_diacritics",
    "tokenize",
    "topic_labels",
    "words",
]
