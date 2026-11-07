"""Chuẩn hóa tiếng Việt, tách từ và nhận diện CLB dùng riêng cho pipeline giải pháp."""

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

# Từ dừng ở mức âm tiết. Không quá lớn để tránh mất từ khóa chuyên môn.
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
        "clb",
        "câu",
        "lạc",
        "bộ",
    ]
)

_STRIP_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")

_CLUB_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    (
        MAIN_CLUBS[0],
        (
            "proptit",
            "pro ptit",
            "clb lap trinh",
            "lap trinh ptit",
            "clb lập trình",
            "clb lập trình ptit",
            "clb lap trinh ptit",
            "clb code",
            "ben code",
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
            "ben game",
            "clb lam game",
            "clb làm game",
        ),
    ),
    (S2B_CLUB, ("s2b", "cong dong s2b", "cộng đồng s2b")),
]

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
)

_TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "Tuyển thành viên": (
        "tuyen",
        "tuyển",
        "ung tuyen",
        "ứng tuyển",
        "dang ky",
        "đăng ký",
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
        "ban chu nhiem",
        "ban chủ nhiệm",
        "nhiem ky",
        "nhiệm kỳ",
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
        "tiktok",
        "youtube",
        "github",
        "website",
        "dia chi",
        "địa chỉ",
        "van phong",
        "văn phòng",
        "so dien thoai",
        "số điện thoại",
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
        "java",
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
    ),
    "Sản phẩm": (
        "san pham",
        "sản phẩm",
        "du an",
        "dự án",
        "lam ra",
        "làm ra",
    ),
}

# Ánh xạ nhóm chủ đề nhận từ câu hỏi sang nhãn topic thật trong corpus.
_TOPIC_GROUPS: dict[str, tuple[str, ...]] = {
    "Tuyển thành viên": ("Tuyển thành viên",),
    "Ban chủ nhiệm": ("Ban chủ nhiệm",),
    "Giải thưởng": ("Giải thưởng", "Giải thưởng & sản phẩm"),
    "Liên hệ": ("Liên hệ",),
    "Phí & quỹ": ("Phí & quỹ",),
    "Nội quy": ("Nội quy & nghĩa vụ",),
    "Sự kiện": ("Sự kiện nội bộ", "Sự kiện đối ngoại", "Sự kiện chung"),
    "Giới thiệu & lịch sử": (
        "Giới thiệu chung",
        "Lịch sử & truyền thống",
        "Lịch sử & thành lập",
    ),
    "Đào tạo": ("Đào tạo", "Cơ cấu & đào tạo"),
    "Cơ cấu": ("Cơ cấu", "Cơ cấu & team dự án", "Cơ cấu & đào tạo"),
    "Điều kiện tham gia": ("Điều kiện tham gia",),
    "Quyền lợi": ("Quyền lợi",),
    "Quan hệ": ("Quan hệ giữa các CLB", "Trực thuộc", "So sánh & tham gia nhiều CLB"),
    "Sản phẩm": ("Sản phẩm", "Giải thưởng & sản phẩm"),
}


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt, xử lý cả chữ đ/Đ (NFD không tách được hai ký tự này)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def ascii_text(text: str) -> str:
    return _SPACES.sub(" ", strip_diacritics(normalize(text))).strip()


def words(text: str) -> list[str]:
    return tokenize(text)


def ascii_words(text: str) -> list[str]:
    return ascii_text(text).split()


def ascii_content_words_raw(text: str) -> list[str]:
    """Tokenizer BM25 khớp với thiết kế thí nghiệm: NFC, bỏ dấu câu, bỏ dấu, lọc stopword.

    Không thống nhất biến thể thanh điệu (oà/òa) để giữ đúng hành vi đã đo trên train.
    """
    value = unicodedata.normalize("NFC", text).lower()
    value = _STRIP_PUNCT.sub(" ", value)
    value = strip_diacritics(value)
    return [word for word in value.split() if len(word) > 1 and word not in STOPWORDS]


def content_words(text: str, ascii_only: bool = False) -> list[str]:
    tokens = ascii_words(text) if ascii_only else words(text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    value = "_".join(ascii_words(text))
    return [value[i : i + n] for i in range(max(1, len(value) - n + 1))]


def has_diacritics(text: str) -> bool:
    return ascii_text(text).replace(" ", "") != normalize(text).replace(" ", "")


def detect_clubs(text: str) -> list[str]:
    """Trả về các CLB được nhắc tới, theo thứ tự xuất hiện trong danh sách chuẩn.

    So khớp theo ranh giới từ để các bí danh ngắn như "mm", "lt" không lọt vào
    giữa một từ khác (ví dụ "comment").
    """
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
    """Đổi các nhóm chủ đề nhận từ câu hỏi thành nhãn topic có trong corpus."""
    labels: set[str] = set()
    for topic in topics:
        labels.update(_TOPIC_GROUPS.get(topic, ()))
    return labels


def clean_answer(text: str, max_chars: int = 2000) -> str:
    """Bỏ markdown/gạch đầu dòng, gộp thành đoạn văn để hợp với ROUGE-L và judge."""
    text = re.sub(r"\*\*|__|`", "", text)
    # Bỏ mọi dạng trích dẫn ID như [12], [1, 2], (id: 12), (id: [1]), [id: 12]
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    text = re.sub(r"\(?\s*id\s*[:：]?\s*\[?\d+(?:\s*,\s*\d+)*\]?\s*\)?", "", text, flags=re.I)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•+]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    answer = _SPACES.sub(" ", " ".join(lines)).strip()
    for prefix in ("Câu trả lời:", "Trả lời:", "Đáp án:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()
    return answer[:max_chars]


__all__ = [
    "ALL_CLUBS",
    "MAIN_CLUBS",
    "S2B_CLUB",
    "ascii_text",
    "ascii_content_words_raw",
    "ascii_words",
    "char_ngrams",
    "clean_answer",
    "content_words",
    "detect_clubs",
    "detect_topics",
    "has_diacritics",
    "is_founding_comparison",
    "is_multi_club_question",
    "normalize",
    "split_sentences",
    "strip_diacritics",
    "tokenize",
    "topic_labels",
    "words",
]
