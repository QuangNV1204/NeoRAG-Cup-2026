from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from neorag.data import DATA_DIR, Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.solution.config import SolutionConfig, _env_bool, _env_int
from neorag.solution.corpus import (
    Chunk,
    build_smart_units,
    format_passage_bm25_text,
    format_passage_dense_text,
)
from neorag.solution.index import BM25Index, DenseIndex, normalize01, saturate_bm25
from neorag.solution.vietnamese import (
    MAIN_CLUBS,
    S2B_CLUB,
    accented_bm25_tokens,
    ascii_bm25_tokens,
    ascii_text,
    content_words,
    detect_clubs,
    expand_student_slang,
    has_diacritics,
    is_multi_club_question,
    parse_temporals,
    sanitize_answer,
)

FALLBACK_ABSTAIN = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."

DOMAIN_GLOSSARY = """BỐI CẢNH CỘNG ĐỒNG S2B TẠI PTIT (Học viện Công nghệ Bưu chính Viễn thông):
1. CLB Lập Trình PTIT (tên viết tắt: PROPTIT / ProPTIT, tiếng Anh: Programming PTIT; hay gọi: clb lập trình, clb lt, clb code):
   - Lập trình C, C++, Cấu trúc dữ liệu và Giải thuật (CTDL&GT), Java OOP; có 5 team dự án (Team AI, Team Data, Team Game, Team Mobile, Team Web; trước đây có Team Backend nay gộp vào Team Web).
   - Lưu ý: "Team Game / nhóm làm game bên CLB Lập Trình / ProPTIT" là team dự án của CLB Lập Trình PTIT, KHÔNG phải GCC.
2. CLB Multimedia PTIT (tên đầy đủ: CLB Sáng tạo Đa phương tiện MULTIMEDIA - PTIT, tiếng Anh: Multimedia PTIT Club, viết tắt: MPC; hay gọi: clb đa phương tiện, clb mm, mm):
   - Đào tạo thiết kế đồ họa, truyền thông, quay phim, chụp ảnh, làm phim, tổ chức sự kiện.
   - Có 5 ban: Ban Nội dung (The Bards), Ban Đối ngoại (The Paladins), Ban Đối nội (The Clerics), Ban Media, Ban Thiết kế. Đợt tuyển Gen 14 (2026) chủ đề ROLLIANCE.
   - Lưu ý: "lên plan chạy chương trình / chạy sự kiện" là lập kế hoạch tổ chức sự kiện, KHÔNG phải chạy code máy tính.
3. CLB Nhà Sáng Tạo Game PTIT (tiếng Anh: Game Creators Club, viết tắt: GCC; hay gọi: clb game, clb làm game):
   - Có 2 nhánh đào tạo: Game Developer (lập trình game: C, C++, CTDL&GT, C# OOP, Unity — sinh viên hay gọi là "nhánh Dev" hoặc "bên code" của GCC) và Game Artist (thiết kế/vẽ game: vẽ cơ bản, Photoshop, After Effects, phối cảnh, shading, 12 nguyên tắc animation, Spine — hay gọi là "nhánh Art" hoặc "bên vẽ" của GCC).
   - Lưu ý: Khi câu hỏi đang hỏi về GCC / CLB game mà nói "bên code" với "bên vẽ", đó là 2 nhánh Dev và Art của GCC, KHÔNG phải CLB Lập Trình PTIT!
4. Cộng đồng S2B: cộng đồng gồm 3 CLB trên (ProPTIT, Multimedia PTIT, GCC), có hoạt động chung như Big Game S2B, Sinh nhật cộng đồng S2B (khác với Big Game Training và Sinh nhật riêng của từng CLB).
5. Viết tắt sinh viên thường gặp: hv = Học viện, tv = thành viên, ttv = tuyển thành viên, dk = đăng ký, nk = nhiệm kỳ, bcn = Ban chủ nhiệm, cn = Chủ nhiệm, pcn/pho cn = Phó Chủ nhiệm, dpt = Đa phương tiện, khmt = Khoa học Máy tính, cntt = Công nghệ Thông tin, attt = An toàn Thông tin, sdt = số điện thoại, d26/k26 = khóa D26 (năm 2026)."""

QUERY_PLAN_PROMPT = f"""{DOMAIN_GLOSSARY}

NHIỆM VỤ: Phân tích câu hỏi của người dùng và lập kế hoạch tìm kiếm tài liệu.
- Giữ nguyên tuyệt đối mọi con số, ngày tháng, năm, nhiệm kỳ, tên riêng, tên phần mềm/cuộc thi trong câu hỏi gốc.
- Giải nghĩa đúng từ không dấu, viết tắt, tiếng lóng theo đúng CLB và ngữ cảnh ở trên. Không tự bịa câu trả lời.
- Phân biệt rõ SCOPE:
  + `SCOPE: MULTI_CLUB`: CHỈ dùng khi câu hỏi so sánh giữa nhiều CLB, hỏi cả 3 CLB, hoặc hỏi so sánh nhất trong S2B (ví dụ: CLB nào lâu đời nhất, trẻ nhất, tuyển ít nhất, đều có blog/slogan/link đăng ký không...). Khi đó tạo truy vấn riêng cho TỪNG CLB liên quan (đủ cả 3 CLB nếu hỏi phạm vi S2B/3 CLB).
  + `SCOPE: SINGLE`: Khi câu hỏi chỉ hỏi về 1 CLB, 1 sự kiện cụ thể, hoặc 1 ngành học cụ thể (ví dụ sinh viên ngành đa phương tiện/DPT, sinh viên ngành KHMT, sinh viên ngành Marketing, học sinh năm 2, chưa có laptop...). Tuyệt đối không nhầm câu hỏi về một ngành học thành MULTI_CLUB!
- Tạo 3 đến 5 truy vấn tìm kiếm ngắn gọn, giàu từ khóa khớp với văn phong tài liệu (kể cả thuật ngữ tiếng Anh liên quan nếu câu hỏi mô tả gián tiếp).

Trả về ĐÚNG định dạng sau (không giải thích thêm):
SCOPE: <SINGLE hoặc MULTI_CLUB>
NORMALIZED: <câu hỏi viết lại rõ ràng bằng tiếng Việt có dấu, đầy đủ tên CLB>
Q1: <truy vấn 1>
Q2: <truy vấn 2>
Q3: <truy vấn 3>
Q4: <truy vấn 4 (tùy chọn)>
Q5: <truy vấn 5 (tùy chọn)>

CÂU HỎI GỐC: {{query}}"""

EVIDENCE_SELECT_PROMPT = f"""{DOMAIN_GLOSSARY}

Bạn là chuyên gia thẩm định bằng chứng cho hệ thống hỏi đáp S2B.
Dưới đây là CÂU HỎI của người dùng và CÁC ĐOẠN TÀI LIỆU ỨNG VIÊN (mỗi đoạn có dạng `[id] CLB — chủ đề` rồi nội dung đầy đủ).

CÂU HỎI GỐC: {{query}}
DIỄN GIẢI RÕ NGHĨA: {{normalized_query}}

CÁC ĐOẠN TÀI LIỆU ỨNG VIÊN:
{{candidates}}

NHIỆM VỤ: Đọc kỹ toàn bộ ứng viên và xác định:
1. `STATUS`: chọn đúng 1 trong 3 nhãn:
   - `ANSWERABLE`: Các đoạn ứng viên có đủ dữ kiện để trả lời câu hỏi (kể cả khi đáp án dựa trên tài liệu là "Không", "Chưa mở", "Không cố định", hoặc tính toán trực tiếp được từ ngày tháng/số lượng trong đoạn, hoặc tổng hợp từ các đoạn có ghi khác nhau).
   - `PARTIAL`: Câu hỏi có nhiều ý nhưng tài liệu chỉ có thông tin cho một phần, HOẶC tài liệu có ghi danh sách/lộ trình/nhiệm kỳ/đợt tuyển của đúng đối tượng nhưng khuyết chi tiết cụ thể được hỏi (ví dụ: có BCN nhiệm kỳ được hỏi nhưng không ghi chức danh đó; hỏi lịch năm 2027 trong khi tài liệu chỉ có lịch 2026 và ghi rõ chỉ áp dụng năm 2026; so sánh 3 CLB nhưng chỉ 2 CLB có số liệu).
   - `UNANSWERABLE`: Tài liệu HOÀN TOÀN KHÔNG CÓ thông tin trả lời câu hỏi chính (ví dụ: hỏi phí/quỹ hằng tháng của Multimedia hay GCC; hỏi giải thưởng của Multimedia; hỏi S2B viết tắt của từ gì; hỏi biệt danh của một ban không được nêu biệt danh; hỏi tên bài hát truyền thống khi tài liệu chỉ nói số lượng; hỏi phòng/chỗ ngồi riêng của GCC; hỏi công cụ/engine của một nhóm khi tài liệu chỉ nêu tên nhóm; hỏi phần mềm cụ thể như Premiere khi tài liệu chỉ nói chung về quay phim/làm phim; hỏi Phó Chủ nhiệm GCC; hỏi fanpage/website riêng của S2B; hỏi địa điểm tổ chức một sự kiện năm cụ thể không được ghi).
   *Lưu ý quan trọng*: Câu hỏi dạng "Có ... không?" mà tài liệu KHÔNG HỀ NHẮC TỚI điều được hỏi thì KHÔNG PHẢI là đáp án "Không", mà phải xếp là `UNANSWERABLE` (hoặc `PARTIAL` nếu có ý khác trả lời được)!

2. `CORE_IDS`: Danh sách ID tối thiểu (1 đến 4 ID, xếp theo mức độ quan trọng giảm dần, cách nhau bởi dấu phẩy) trực tiếp dùng để viết câu trả lời:
   - Với `ANSWERABLE` / `PARTIAL`:
     + CHỈ chọn những đoạn THỰC SỰ CẦN THIẾT và TRỰC TIẾP chứa dữ kiện trả lời câu hỏi.
     + Đoạn chứa câu trả lời trực tiếp cho câu hỏi chính PHẢI ĐỨNG Ở VỊ TRÍ ĐẦU TIÊN (rank 1).
     + Với câu hỏi 1 ý / 1 CLB / 1 ngành học: CHỈ CHỌN 1 ĐẾN 2 ID (tối đa 3 ID nếu câu hỏi có 2 ý nhỏ). TUYỆT ĐỐI KHÔNG chọn đoạn của CLB khác (ví dụ: hỏi ngành Đa phương tiện thì CHỈ chọn CLB Multimedia PTIT, tuyệt đối không chọn đoạn của ProPTIT hay GCC; hỏi sinh viên Marketing thì CHỈ chọn GCC)!
     + Với câu hỏi so sánh 3 CLB: CHỌN ĐÚNG 1 đoạn tiêu biểu nhất cho MỖI CLB (tổng 3 đoạn, mỗi CLB 1 đoạn).
     + Các đoạn chỉ mang tính bối cảnh phụ hoặc liên quan xa thì đưa vào `EXTRA_IDS`, KHÔNG đưa vào `CORE_IDS`.
   - Với `UNANSWERABLE`: Nếu có 1-2 đoạn nhắc tới đối tượng/khái niệm gần nhất liên quan đến câu hỏi (ví dụ đoạn giới thiệu S2B khi hỏi S2B viết tắt của gì; đoạn liệt kê các ban khi hỏi biệt danh một ban; đoạn nói về vòng Training khi hỏi thời lượng vòng đó; đoạn ghi Chủ nhiệm khi hỏi Phó Chủ nhiệm), hãy ghi 1-2 ID đó vào `CORE_IDS`. Nếu hoàn toàn không có đoạn nào liên quan trực tiếp (như phí của Multimedia/GCC, giải thưởng Multimedia, fanpage riêng của S2B), ghi `NONE`.

3. `EXTRA_IDS`: Tối đa 4 ID liên quan tiếp theo (không trùng `CORE_IDS`) để dự phòng xếp hạng truy xuất, hoặc `NONE`.

Trả về ĐÚNG 3 dòng theo định dạng (không giải thích thêm):
STATUS: <ANSWERABLE | PARTIAL | UNANSWERABLE>
CORE_IDS: <các số ID cách nhau dấu phẩy, hoặc NONE>
EXTRA_IDS: <các số ID cách nhau dấu phẩy, hoặc NONE>"""

COMPOSE_SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp chuẩn xác của cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT), gồm CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).
Bạn CHỈ sử dụng thông tin có trong các đoạn tài liệu được cung cấp để trả lời đúng trọng tâm câu hỏi."""

COMPOSE_USER_PROMPT = """CÁC VÍ DỤ ĐÁP ÁN CHUẨN THAM KHẢO (học cách diễn đạt súc tích, độ dài ngắn gọn và cách dùng từ chuẩn của BTC):
{fewshot_block}

---
THÔNG TIN CÂU HỎI HIỆN TẠI:
- Câu hỏi gốc của người dùng: {query}
- Diễn giải câu hỏi: {normalized_query}
- Trạng thái dữ liệu thẩm định: {status}

ĐOẠN TÀI LIỆU CHÍNH ĐƯỢC PHÉP DÙNG:
{context_block}

QUY TẮC BẮT BUỘC KHI VIẾT CÂU TRẢ LỜI:
1. **Chiến lược Extractive Span (Trích xuất nguyên văn tối đa & Tối ưu Groundedness/ROUGE-L)**:
   - Tối đa hóa việc trích xuất NGUYÊN VĂN các mệnh đề, cụm từ chứa dữ kiện trực tiếp từ `ĐOẠN TÀI LIỆU CHÍNH`.
   - TUYỆT ĐỐI KHÔNG tự tiện diễn đạt lại (paraphrase) bằng từ đồng nghĩa khi trong tài liệu đã có cụm từ chính xác.
   - Giữ NGUYÊN VĂN 100%: mọi tên riêng, con số, mốc ngày/tháng/năm, tên giải thưởng, tên ban, biệt danh, tên cuộc thi, liên kết fanpage/website/github/tiktok.
   - Mọi từ ngữ, sự kiện trong câu trả lời đều phải có căn cứ nguyên văn trong đoạn tài liệu chính (đảm bảo Groundedness tuyệt đối).

2. **Đồng bộ chuẩn hóa tên CLB & Mốc thời gian (Chuẩn theo tập Train)**:
   - Tên CLB: Bắt buộc dùng đúng danh xưng chính thức: `CLB Lập Trình PTIT`, `CLB Multimedia PTIT`, `CLB Nhà Sáng Tạo Game PTIT (GCC)`, `Cộng đồng S2B`.
   - Mốc thời gian: Luôn ghi rõ `nhiệm kỳ YYYY-YYYY` (không viết tắt `nk`), `giai đoạn YYYY-YYYY`, `năm YYYY`, ngày `DD/MM/YYYY`, `Gen X` (viết hoa chữ G).

3. **Phạm vi & Trả lời trực tiếp (Chống nhiễu tuyệt đối)**:
   - CHỈ dùng thông tin trong `ĐOẠN TÀI LIỆU CHÍNH` ở trên và CHỈ trả lời đúng những gì câu hỏi hỏi.
   - Trả lời trực tiếp, tự tin, đi thẳng vào đáp án.
   - TUYỆT ĐỐI KHÔNG thêm các câu phân bua, giải thích thừa thãi ngoài lề như "Tài liệu không so sánh với Python", "Tài liệu không nêu học phí trung tâm ngoài", "Tài liệu không ghi quy định riêng cho anh dạy", "Tài liệu không nêu yêu cầu kỹ năng của mảng này", "nhưng không ghi rõ sự kiện này có diễn ra trong vòng training hay không". Khi tài liệu đã có thông tin trả lời ý chính, hãy nêu rõ thông tin đó một cách súc tích.
   - TUYỆT ĐỐI KHÔNG đưa thêm thông tin phụ ngoài phạm vi câu hỏi (như học bổng, điểm rèn luyện, quỹ, các sự kiện khác...) dù có xuất hiện trong đoạn tài liệu.
   - TUYỆT ĐỐI KHÔNG khuyên nhủ ("bạn nên liên hệ fanpage...", "hãy hỏi trực tiếp..."), trừ khi nguyên văn đoạn tài liệu có câu hướng dẫn theo dõi fanpage để cập nhật (như thông báo tuyển thành viên).

2. **Quy tắc Từ chối (`UNANSWERABLE`) & Thiếu một phần dữ liệu (`PARTIAL`)**:
   - Khi `Trạng thái dữ liệu` là `UNANSWERABLE` (hoặc tài liệu không hề đề cập chi tiết được hỏi, kể cả câu hỏi dạng "Có ... không?"):
     + **TUYỆT ĐỐI KHÔNG** mở đầu bằng "Có.", "Không.", "Chưa." hay khẳng định phủ định!
     + Mở đầu trực tiếp bằng: `"Tài liệu hiện có không có thông tin về ..."` hoặc `"Tài liệu không nêu ..."` / `"Tài liệu không ghi ..."`.
     + Nếu trong `ĐOẠN TÀI LIỆU CHÍNH` có thông tin liên quan gần nhất (ví dụ hỏi biệt danh Ban Thiết kế mà đoạn văn chỉ nêu biệt danh 3 ban khác; hoặc hỏi thời lượng vòng Training Multimedia mà chỉ có thời điểm bắt đầu; hoặc hỏi Phó Chủ nhiệm GCC mà chỉ ghi Chủ nhiệm), hãy nêu ngắn gọn trong câu tiếp theo: `"Tài liệu chỉ cho biết [thông tin liên quan ngắn gọn]."` và không coi đó là đáp án cho điều bị thiếu.
   - Khi `Trạng thái dữ liệu` là `PARTIAL`: trả lời đầy đủ phần thông tin tài liệu có ghi trong 1-2 câu, và nói rõ phần nào tài liệu không ghi/không có.

3. **Quy tắc Câu hỏi Có/Không khi ĐỦ dữ liệu (`ANSWERABLE`)**:
   - CHỈ khi `ĐOẠN TÀI LIỆU CHÍNH` thực sự có thông tin khẳng định hoặc phủ định điều được hỏi thì mới mở đầu ngắn gọn bằng `"Có."`, `"Không."`, `"Đúng."`, `"Được."`, `"Không cố định:"`, `"Chưa trễ."`, `"Chưa chắc."`, `"Không bắt buộc."` tùy đúng ngữ nghĩa tài liệu, rồi nêu ngay dữ kiện chứng minh từ tài liệu.

4. **Độ dài súc tích & Tối ưu ROUGE-L (RẤT QUAN TRỌNG)**:
   - Đáp án chuẩn của BTC có độ dài trung bình rất ngắn (~150–250 ký tự, 1–2 câu).
   - Câu hỏi một ý / một CLB: BẮT BUỘC gói gọn trong đúng 1 đến 2 câu súc tích (khoảng 140–240 ký tự). Tránh viết dài, lan man vì sẽ làm giảm mạnh điểm ROUGE-L!
   - Câu hỏi nhiều ý / so sánh 3 CLB: Tối đa 2 đến 3 câu (khoảng 220–350 ký tự), trả lời đủ từng ý.
   - Câu đầu tiên phải nêu rõ tên đầy đủ của CLB / chủ thể được hỏi (ví dụ `CLB Lập Trình PTIT`, `CLB Multimedia PTIT`, `CLB Nhà Sáng Tạo Game PTIT (GCC)`, `Cộng đồng S2B`) để câu trả lời tự trọn nghĩa và đạt điểm Response Relevancy cao.

5. **Bám sát nguyên văn & Giữ mốc thời gian/nguồn**:
   - Chép NGUYÊN VĂN cụm từ quan trọng trong tài liệu cho ngày/tháng/năm (`09/10/2011`), khoảng thời gian (`hai tháng`, `khoảng 3 tháng`, `khoảng 1 tháng`, `24 giờ`), tên riêng, tên giải thưởng (`Giải Ba Chuyên Tin 2022`), tên ban & biệt danh (`Ban Nội dung (The Bards)`), tên cuộc thi (`ProJApp`, `ProPixelGameJam`), khẩu hiệu, email, số điện thoại, đường link. Không tự chèn từ mở ngoặc vào giữa cụm từ gốc.
   - Giữ rõ bối cảnh nguồn hoặc thời điểm khi tài liệu có gắn mốc (ví dụ: `"Theo Phòng truyền thống của CLB Lập Trình PTIT..."`, `"Theo bài tuyển thành viên năm 2026..."`, `"nhiệm kỳ 2025-2026"`, `"Các mốc thời gian này chỉ áp dụng cho đợt tuyển năm 2026"`).
   - Với câu hỏi tính số năm/khoảng thời gian: kiểm tra kỹ phép trừ ngày/tháng/năm (ví dụ thành lập `09/10/2011` thì tháng `8/2026` là đã qua `14 năm` hoạt động và đến đúng ngày `09/10/2026` là tròn `15 năm`).
   - KHÔNG dùng markdown, KHÔNG in đậm, KHÔNG gạch đầu dòng, KHÔNG ghi mã `[id]` trong câu trả lời.

Hãy thực hiện theo ĐÚNG 2 mục dưới đây:
KIỂM TRA: <1 dòng ngắn kiểm tra: câu hỏi hỏi ý gì, dữ kiện trả lời trực tiếp trong đoạn chính là gì, độ dài dự kiến 1-2 câu>
TRẢ LỜI: <đoạn văn câu trả lời cuối cùng duy nhất, súc tích 1-2 câu>"""

_SCOPE_RE = re.compile(r"^\s*SCOPE\s*[:：]\s*(SINGLE|MULTI_CLUB)\b", re.I | re.M)
_NORMALIZED_RE = re.compile(r"^\s*NORMALIZED\s*[:：]\s*(.+)$", re.I | re.M)
_QUERY_LINE_RE = re.compile(r"^\s*Q\d+\s*[:：]\s*(.+)$", re.I | re.M)

_STATUS_RE = re.compile(r"^\s*\**\s*STATUS\s*\**\s*[:：]\s*(ANSWERABLE|PARTIAL|UNANSWERABLE)\b", re.I | re.M)
_CORE_IDS_RE = re.compile(r"^\s*\**\s*CORE_IDS\s*\**\s*[:：]\s*(.+)$", re.I | re.M)
_EXTRA_IDS_RE = re.compile(r"^\s*\**\s*EXTRA_IDS\s*\**\s*[:：]\s*(.+)$", re.I | re.M)
_ANSWER_SPLIT_RE = re.compile(
    r"^\s*\**\s*(?:TRẢ\s*LỜI|CÂU\s*TRẢ\s*LỜI|ANSWER)\s*\**\s*[:：]\s*",
    re.I | re.M,
)
_DIGITS_RE = re.compile(r"\d+")

_SERIES_TOPICS = frozenset(
    {
        "Giải thưởng",
        "Giải thưởng & sản phẩm",
        "Ban chủ nhiệm",
        "Tuyển thành viên",
        "Đào tạo",
        "Cơ cấu & đào tạo",
        "Cơ cấu",
        "Cơ cấu & team dự án",
        "Sự kiện chung",
        "Liên hệ",
    }
)

_FALLBACK_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "CLB Lập Trình PTIT được thành lập ngày nào và phương châm hoạt động là gì?",
        'CLB Lập Trình PTIT (PROPTIT) được thành lập ngày 09/10/2011, với phương châm hoạt động "Chia sẻ để cùng nhau phát triển".',
    ),
    (
        "Ban Thiết kế của CLB Multimedia PTIT có biệt danh là gì?",
        "Tài liệu không nêu biệt danh của Ban Thiết kế CLB Multimedia PTIT. Trong đợt tuyển Gen 14, các biệt danh được nêu là của Ban Nội dung (The Bards), Ban Đối ngoại (The Paladins) và Ban Đối nội (The Clerics).",
    ),
    (
        "Phí hoặc quỹ hằng tháng khi tham gia CLB Multimedia PTIT là bao nhiêu?",
        "Tài liệu hiện có không có thông tin về phí hoặc quỹ của CLB Multimedia PTIT.",
    ),
)

@dataclass(frozen=True)
class QueryPlan:
    raw_query: str
    normalized_query: str
    scope: str
    search_queries: list[tuple[str, float]]

@dataclass(frozen=True)
class EvidenceDecision:
    status: str
    core_ids: list[int]
    extra_ids: list[int]
    raw_output: str = ""

def _parse_id_list(text: str, valid_ids: set[int], limit: int) -> list[int]:
    if "NONE" in text.upper():
        return []
    result: list[int] = []
    for match in _DIGITS_RE.findall(text):
        doc_id = int(match)
        if doc_id in valid_ids and doc_id not in result:
            result.append(doc_id)
            if len(result) >= limit:
                break
    return result

def format_doc_block(doc: Document) -> str:
    return f"[{doc.id}] {doc.club} — {doc.topic}\n{doc.text}"

class TrainFewShotBank:

    def __init__(self, dense_factory: Any, enabled: bool = True):
        self.rows: list[dict[str, Any]] = []
        self.ascii_queries: list[str] = []
        self.index: DenseIndex | None = None
        if not enabled:
            return
        path = DATA_DIR / "train" / "questions.jsonl"
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("query") and row.get("answer"):
                            self.rows.append(row)
                            self.ascii_queries.append(ascii_text(row["query"]))
            if self.rows:
                self.index = dense_factory().fit(
                    [row["query"] for row in self.rows],
                    name="train_fewshot_queries_v2",
                )
        except Exception:
            self.rows = []
            self.index = None

    def select(
        self,
        raw_query: str,
        normalized_query: str,
        query_vector: np.ndarray | None,
        status: str,
        k: int = 5,
    ) -> list[tuple[str, str]]:
        if not self.rows or self.index is None or query_vector is None or k <= 0:
            return list(_FALLBACK_EXAMPLES[: max(1, k)])

        raw_key = ascii_text(raw_query)
        norm_key = ascii_text(normalized_query)
        sims = self.index.scores(query_vector)
        order = [int(i) for i in np.argsort(-sims)]

        eligible = [
            i
            for i in order
            if self.ascii_queries[i] != raw_key and self.ascii_queries[i] != norm_key
        ]
        if not eligible:
            return list(_FALLBACK_EXAMPLES[:k])

        answerable_idxs = [i for i in eligible if self.rows[i].get("answerable", True)]
        partial_idxs = [
            i
            for i in answerable_idxs
            if self.rows[i].get("type") in {"thiếu dữ liệu", "mâu thuẫn nguồn", "thời gian"}
        ]
        abstain_idxs = [i for i in eligible if not self.rows[i].get("answerable", True)]

        chosen: list[int] = []

        def _take(pool: list[int], count: int) -> None:
            for idx in pool:
                if len(chosen) >= k:
                    return
                if idx not in chosen:
                    chosen.append(idx)
                    if sum(1 for c in chosen if c in pool) >= count:
                        break

        upper_status = status.upper()
        if upper_status == "UNANSWERABLE":
            _take(abstain_idxs, min(4, k))
            _take(partial_idxs, 1)
            _take(answerable_idxs, k)
        elif upper_status == "PARTIAL":
            _take(partial_idxs, 3)
            _take(abstain_idxs, 1)
            _take(answerable_idxs, k)
        else:
            _take(answerable_idxs, k)
            _take(partial_idxs, 1)
            _take(eligible, k)

        return [(self.rows[i]["query"], self.rows[i]["answer"]) for i in chosen[:k]]

class QueryPlanner:

    def __init__(self, llm: LLMClient | None, model: str, enabled: bool = True):
        self.llm = llm
        self.model = model
        self.enabled = enabled

    def plan(self, query: str) -> QueryPlan:
        det_multi = is_multi_club_question(query)
        slang_expanded = expand_student_slang(query)

        if not self.enabled or self.llm is None:
            return self._fallback_plan(query, slang_expanded, det_multi)

        try:
            raw = self.llm.chat(
                [{"role": "user", "content": QUERY_PLAN_PROMPT.format(query=query)}],
                model=self.model,
            )
        except Exception:
            return self._fallback_plan(query, slang_expanded, det_multi)

        scope_match = _SCOPE_RE.search(raw)
        scope = scope_match.group(1).upper() if scope_match else ("MULTI_CLUB" if det_multi else "SINGLE")
        if det_multi:
            scope = "MULTI_CLUB"

        norm_match = _NORMALIZED_RE.search(raw)
        normalized = norm_match.group(1).strip() if norm_match else slang_expanded
        if not normalized:
            normalized = slang_expanded

        sub_queries = [m.strip() for m in _QUERY_LINE_RE.findall(raw) if len(m.strip()) >= 3]
        weighted: list[tuple[str, float]] = [
            (query, 1.0),
            (normalized, 1.15),
        ]
        if slang_expanded != query:
            weighted.append((slang_expanded, 0.95))
        for sq in sub_queries[:5]:
            weighted.append((sq, 1.0))

        if scope == "MULTI_CLUB":
            weighted.extend(self._multi_club_queries(normalized))

        return QueryPlan(
            raw_query=query,
            normalized_query=normalized,
            scope=scope,
            search_queries=self._dedup_queries(weighted, limit=9),
        )

    def _fallback_plan(self, query: str, slang_expanded: str, is_multi: bool) -> QueryPlan:
        scope = "MULTI_CLUB" if is_multi else "SINGLE"
        weighted: list[tuple[str, float]] = [(query, 1.0)]
        if slang_expanded != query:
            weighted.append((slang_expanded, 1.05))
        if is_multi:
            weighted.extend(self._multi_club_queries(slang_expanded))
        return QueryPlan(
            raw_query=query,
            normalized_query=slang_expanded,
            scope=scope,
            search_queries=self._dedup_queries(weighted, limit=8),
        )

    @staticmethod
    def _multi_club_queries(base_query: str) -> list[tuple[str, float]]:
        mentioned = [c for c in detect_clubs(base_query) if c in MAIN_CLUBS]
        clubs = mentioned if len(mentioned) >= 2 else list(MAIN_CLUBS)
        return [(f"{club}: {base_query}", 0.92) for club in clubs]

    @staticmethod
    def _dedup_queries(items: list[tuple[str, float]], limit: int) -> list[tuple[str, float]]:
        deduped: list[tuple[str, float]] = []
        seen: set[str] = set()
        for text, weight in items:
            key = ascii_text(text)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append((text, weight))
            if len(deduped) >= limit:
                break
        return deduped or [(items[0][0], 1.0)]

class WholePassageHybridRetriever:

    def __init__(
        self,
        documents: list[Document],
        units: list[Chunk],
        whole_dense: DenseIndex,
        unit_dense: DenseIndex,
        bm25_accented: BM25Index,
        bm25_ascii: BM25Index,
        candidate_pool_size: int = 36,
    ):
        self.documents = documents
        self.units = units
        self.whole_dense = whole_dense
        self.unit_dense = unit_dense
        self.bm25_accented = bm25_accented
        self.bm25_ascii = bm25_ascii
        self.candidate_pool_size = candidate_pool_size

        self.doc_pos = {doc.id: idx for idx, doc in enumerate(documents)}
        self.unit_groups: list[np.ndarray] = []
        grouped: dict[int, list[int]] = {}
        for u_idx, unit in enumerate(units):
            grouped.setdefault(unit.doc_id, []).append(u_idx)
        for doc in documents:
            self.unit_groups.append(np.asarray(grouped.get(doc.id, []), dtype=np.int32))

        self.siblings_by_group: dict[tuple[str, str], list[int]] = {}
        for idx, doc in enumerate(documents):
            self.siblings_by_group.setdefault((doc.club, doc.topic), []).append(idx)
        self.doc_temporals = [parse_temporals(doc.text) for doc in documents]

    def retrieve_candidates(
        self, plan: QueryPlan
    ) -> tuple[list[Document], list[int], dict[int, float], np.ndarray]:
        texts = [q for q, _ in plan.search_queries]

        query_vecs = self.whole_dense.embed_queries(texts)
        whole_mat = self.whole_dense.score_matrix(query_vecs)
        unit_mat = self.unit_dense.score_matrix(query_vecs)

        n_queries = len(texts)
        n_docs = len(self.documents)
        per_query_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
        rrf_scores = np.zeros(n_docs, dtype=np.float32)

        for q_idx, (q_text, q_weight) in enumerate(plan.search_queries):
            whole_s = normalize01(whole_mat[q_idx])
            unit_raw = np.zeros(n_docs, dtype=np.float32)
            u_row = unit_mat[q_idx]
            for d_idx, u_indices in enumerate(self.unit_groups):
                if u_indices.size > 0:
                    unit_raw[d_idx] = float(u_row[u_indices].max())
                else:
                    unit_raw[d_idx] = float(whole_mat[q_idx, d_idx])
            unit_s = normalize01(unit_raw)

            dense_s = 0.65 * whole_s + 0.35 * unit_s

            acc_s = saturate_bm25(self.bm25_accented.scores(q_text), k=12.0)
            asc_s = saturate_bm25(self.bm25_ascii.scores(q_text), k=12.0)
            sparse_s = (0.55 * acc_s + 0.45 * asc_s) if has_diacritics(q_text) else (0.20 * acc_s + 0.80 * asc_s)

            combined = (0.62 * dense_s + 0.38 * sparse_s) * float(q_weight)
            per_query_scores[q_idx] = combined

            ranking = np.argsort(-combined)
            for rank, doc_idx in enumerate(ranking):
                rrf_scores[int(doc_idx)] += float(q_weight) / (35.0 + rank)

        max_s = normalize01(per_query_scores.max(axis=0))
        mean_s = normalize01(per_query_scores.mean(axis=0))
        rrf_norm = normalize01(rrf_scores)
        final_scores = 0.55 * max_s + 0.15 * mean_s + 0.30 * rrf_norm

        if plan.scope == "SINGLE":
            clubs = detect_clubs(plan.normalized_query)
            if len(clubs) == 1:
                target_club = clubs[0]
                for idx, doc in enumerate(self.documents):
                    if doc.club == target_club:
                        final_scores[idx] += 0.04

        q_temp = parse_temporals(f"{plan.raw_query} {plan.normalized_query}")
        if q_temp["years"] or q_temp["ranges"]:
            q_years = q_temp["years"]
            q_ranges = set(q_temp["ranges"])
            for idx, doc in enumerate(self.documents):
                d_temp = self.doc_temporals[idx]
                d_years = d_temp["years"]
                d_ranges = set(d_temp["ranges"])
                is_temporal_doc = (
                    doc.topic in _SERIES_TOPICS
                    or bool(d_ranges)
                    or "nhiệm kỳ" in doc.text.lower()
                    or "giai đoạn" in doc.text.lower()
                )
                if is_temporal_doc and d_years:
                    common_years = d_years & q_years
                    if not common_years:
                        final_scores[idx] -= 0.14
                    else:
                        final_scores[idx] += 0.08
                        if q_ranges & d_ranges:
                            final_scores[idx] += 0.08

        global_order = [int(i) for i in np.argsort(-final_scores)]
        pool_indices = self._build_candidate_pool(global_order, final_scores, plan.scope)

        candidates = [self.documents[i] for i in pool_indices]
        hybrid_ids = [self.documents[i].id for i in global_order]
        score_map = {self.documents[i].id: float(final_scores[i]) for i in global_order[:20]}

        rep_vec = query_vecs[min(1, len(query_vecs) - 1)]
        if len(query_vecs) >= 2:
            blended = 0.45 * query_vecs[0] + 0.55 * query_vecs[1]
            norm = float(np.linalg.norm(blended))
            if norm > 1e-9:
                rep_vec = blended / norm

        return candidates, hybrid_ids, score_map, rep_vec

    def _build_candidate_pool(
        self, global_order: list[int], scores: np.ndarray, scope: str
    ) -> list[int]:
        limit = min(len(self.documents), self.candidate_pool_size)
        selected: list[int] = list(global_order[:limit])
        selected_set = set(selected)

        if scope == "MULTI_CLUB":
            quotas = [(MAIN_CLUBS[0], 6), (MAIN_CLUBS[1], 6), (MAIN_CLUBS[2], 6), (S2B_CLUB, 3)]
            for club_name, quota in quotas:
                club_docs = [i for i in global_order if self.documents[i].club == club_name][:quota]
                for idx in club_docs:
                    if idx not in selected_set:
                        selected.append(idx)
                        selected_set.add(idx)

        for idx in global_order[:10]:
            doc = self.documents[idx]
            if doc.topic not in _SERIES_TOPICS:
                continue
            siblings = sorted(
                self.siblings_by_group.get((doc.club, doc.topic), []),
                key=lambda s_idx: -float(scores[s_idx]),
            )
            added = 0
            for s_idx in siblings:
                if s_idx not in selected_set:
                    selected.append(s_idx)
                    selected_set.add(s_idx)
                    added += 1
                    if added >= 4 or len(selected) >= min(len(self.documents), limit + 8):
                        break

        selected.sort(key=lambda i: (-float(scores[i]), i))
        return selected[: min(len(self.documents), limit + 8)]

class EvidenceSelector:

    def __init__(self, llm: LLMClient | None, model: str, max_core_ids: int = 6):
        self.llm = llm
        self.model = model
        self.max_core_ids = max_core_ids

    @staticmethod
    def _rerank_core_ids(
        core_ids: list[int],
        doc_map: dict[int, Document],
        plan: QueryPlan,
        hybrid_rank: dict[int, int],
    ) -> list[int]:
        if len(core_ids) <= 1:
            return core_ids

        q_temp = parse_temporals(f"{plan.raw_query} {plan.normalized_query}")
        q_years = q_temp["years"]
        q_ranges = set(q_temp["ranges"])
        q_words = set(content_words(plan.normalized_query.lower()))

        scored_ids: list[tuple[float, int]] = []
        for orig_idx, cid in enumerate(core_ids):
            if cid not in doc_map:
                scored_ids.append((-999.0, cid))
                continue
            doc = doc_map[cid]
            d_temp = parse_temporals(doc.text)
            d_years = d_temp["years"]
            d_ranges = set(d_temp["ranges"])

            score = 0.0

            if q_years:
                if d_years & q_years:
                    score += 25.0
                    if q_ranges & d_ranges:
                        score += 15.0
                elif d_years:
                    score -= 25.0

            d_words = set(content_words(doc.text.lower()))
            overlap = len(q_words & d_words)
            score += float(overlap) * 2.0

            h_rank = hybrid_rank.get(cid, 999)
            score += max(0.0, 20.0 - h_rank * 0.5)

            score += 10.0 / (1.0 + orig_idx)

            scored_ids.append((score, cid))

        scored_ids.sort(key=lambda item: (-item[0], item[1]))
        return [cid for _, cid in scored_ids]

    @staticmethod
    def _prioritize_sibling_extras(
        core_ids: list[int],
        llm_extra_ids: list[int],
        candidates: list[Document],
        hybrid_ids: list[int],
        limit: int = 5,
    ) -> list[int]:
        doc_map = {d.id: d for d in candidates}
        hybrid_rank = {doc_id: rank for rank, doc_id in enumerate(hybrid_ids)}
        core_set = set(core_ids)

        sibling_candidates: list[int] = []
        for cid in core_ids:
            if cid not in doc_map:
                continue
            cdoc = doc_map[cid]
            for doc in candidates:
                if (
                    doc.id != cid
                    and doc.id not in core_set
                    and doc.id not in sibling_candidates
                    and doc.club == cdoc.club
                    and doc.topic == cdoc.topic
                ):
                    sibling_candidates.append(doc.id)

        sibling_candidates.sort(key=lambda did: hybrid_rank.get(did, 999))

        extras: list[int] = []
        for did in [*sibling_candidates, *llm_extra_ids, *hybrid_ids]:
            if did in doc_map and did not in core_set and did not in extras:
                extras.append(did)
                if len(extras) >= limit:
                    break
        return extras

    def select(
        self,
        plan: QueryPlan,
        candidates: list[Document],
        hybrid_ids: list[int],
    ) -> EvidenceDecision:
        if not candidates:
            return EvidenceDecision("UNANSWERABLE", [], [])

        doc_map = {doc.id: doc for doc in candidates}
        hybrid_rank = {doc_id: rank for rank, doc_id in enumerate(hybrid_ids)}

        if self.llm is None:
            fallback = hybrid_ids[: min(4, len(hybrid_ids))]
            core = self._rerank_core_ids(fallback, doc_map, plan, hybrid_rank)
            extras = self._prioritize_sibling_extras(core, [], candidates, hybrid_ids, limit=5)
            return EvidenceDecision("ANSWERABLE", core, extras)

        valid_ids = {doc.id for doc in candidates}
        candidate_text = "\n\n".join(format_doc_block(doc) for doc in candidates)
        prompt = EVIDENCE_SELECT_PROMPT.format(
            query=plan.raw_query,
            normalized_query=plan.normalized_query,
            candidates=candidate_text,
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], model=self.model)
        except Exception as exc:
            fallback = hybrid_ids[: min(4, len(hybrid_ids))]
            core = self._rerank_core_ids(fallback, doc_map, plan, hybrid_rank)
            extras = self._prioritize_sibling_extras(core, [], candidates, hybrid_ids, limit=5)
            return EvidenceDecision(
                "ANSWERABLE",
                core,
                extras,
                raw_output=f"ERROR {type(exc).__name__}: {exc}",
            )

        status_match = _STATUS_RE.search(raw)
        core_match = _CORE_IDS_RE.search(raw)
        extra_match = _EXTRA_IDS_RE.search(raw)

        if not status_match and not core_match:
            legacy_ids = _parse_id_list(raw, valid_ids, self.max_core_ids)
            core = legacy_ids or hybrid_ids[: min(3, len(hybrid_ids))]
            core = self._rerank_core_ids(core, doc_map, plan, hybrid_rank)
            extras = self._prioritize_sibling_extras(core, [], candidates, hybrid_ids, limit=5)
            return EvidenceDecision("ANSWERABLE", core, extras, raw_output=raw)

        status = status_match.group(1).upper() if status_match else "ANSWERABLE"
        core_ids = (
            _parse_id_list(core_match.group(1), valid_ids, self.max_core_ids) if core_match else []
        )
        llm_extra_ids = _parse_id_list(extra_match.group(1), valid_ids, 5) if extra_match else []
        llm_extra_ids = [i for i in llm_extra_ids if i not in core_ids]

        if status != "UNANSWERABLE" and not core_ids:
            core_ids = [doc.id for doc in candidates[: min(4, len(candidates))]]

        core_ids = self._rerank_core_ids(core_ids, doc_map, plan, hybrid_rank)

        extra_ids = self._prioritize_sibling_extras(
            core_ids=core_ids,
            llm_extra_ids=llm_extra_ids,
            candidates=candidates,
            hybrid_ids=hybrid_ids,
            limit=5,
        )

        return EvidenceDecision(
            status=status,
            core_ids=core_ids,
            extra_ids=extra_ids,
            raw_output=raw,
        )

class GroundedComposer:

    def __init__(self, llm: LLMClient, model: str, fewshot_count: int = 5):
        self.llm = llm
        self.model = model
        self.fewshot_count = fewshot_count

    def compose(
        self,
        plan: QueryPlan,
        decision: EvidenceDecision,
        core_docs: list[Document],
        fewshot_examples: list[tuple[str, str]],
    ) -> tuple[str, str]:
        if decision.status == "UNANSWERABLE" and not core_docs:
            context_block = "(Không có đoạn tài liệu nào trong kho dữ liệu chứa thông tin về nội dung được hỏi.)"
        else:
            context_block = "\n\n".join(format_doc_block(doc) for doc in core_docs)

        fewshot_lines = [
            f"Ví dụ {idx}:\nCâu hỏi: {q_ex}\nĐáp án chuẩn: {a_ex}"
            for idx, (q_ex, a_ex) in enumerate(fewshot_examples[: self.fewshot_count], 1)
        ]
        fewshot_block = "\n\n".join(fewshot_lines) if fewshot_lines else "(Không có)"

        user_prompt = COMPOSE_USER_PROMPT.format(
            fewshot_block=fewshot_block,
            query=plan.raw_query,
            normalized_query=plan.normalized_query,
            status=decision.status,
            context_block=context_block,
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": COMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
            )
        except Exception as exc:
            return FALLBACK_ABSTAIN, f"ERROR {type(exc).__name__}: {exc}"

        answer = self._extract_answer(raw, decision.status)
        return answer or FALLBACK_ABSTAIN, raw

    @staticmethod
    def _extract_answer(raw: str, status: str) -> str:
        text = raw.strip()
        parts = _ANSWER_SPLIT_RE.split(text)
        if len(parts) >= 2:
            candidate = parts[-1].strip()
        else:
            lines = [
                line
                for line in text.splitlines()
                if not re.match(r"^\s*\**\s*KIỂM\s*TRA\s*\**\s*[:：]", line, flags=re.I)
            ]
            candidate = "\n".join(lines).strip()
        return sanitize_answer(candidate, status=status, max_chars=2000)

class NeoRAGPipeline(RAGPipeline):

    def __init__(
        self,
        config: SolutionConfig | None = None,
        llm: LLMClient | None = None,
        embedder: Any | None = None,
        **_: Any,
    ):
        self.config = config or SolutionConfig()
        self.config.ensure_dirs()
        self.llm = llm or LLMClient(cache_path=self.config.cache_dir / "llm.sqlite")
        self.embedder = embedder
        self.llm_model = self.config.llm_model
        self.embedding_model = self.config.embedding_model

        self.use_query_planner = _env_bool("NEORAG_QUERY_PLANNER", True)
        self.candidate_pool_size = _env_int("NEORAG_CANDIDATE_POOL", 36)
        self.max_core_ids = _env_int("NEORAG_MAX_CORE_IDS", 6)
        self.fewshot_count = _env_int("NEORAG_FEWSHOT_COUNT", 5)
        self.use_train_fewshot = _env_bool("NEORAG_TRAIN_FEWSHOT", True)

        self.documents: list[Document] = []
        self.doc_by_id: dict[int, Document] = {}
        self.retriever: WholePassageHybridRetriever | None = None
        self.planner: QueryPlanner | None = None
        self.selector: EvidenceSelector | None = None
        self.composer: GroundedComposer | None = None
        self.fewshot_bank: TrainFewShotBank | None = None
        self.last_debug: dict[str, Any] = {}

    def prepare(self) -> None:
        self.config.ensure_dirs()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder is not None:
            vectors = self.embedder.embed(texts)
        else:
            vectors = self.llm.embed(
                texts,
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
        return np.asarray(vectors, dtype=np.float32)

    def _make_dense_index(self) -> DenseIndex:
        return DenseIndex(
            embed=self._embed,
            cache_dir=self.config.cache_dir,
            embedding_model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )

    def index(self, corpus: list[Document]) -> None:
        self.documents = list(corpus)
        if not self.documents:
            raise ValueError("Corpus rỗng")
        self.doc_by_id = {doc.id: doc for doc in self.documents}

        whole_texts = [format_passage_dense_text(doc) for doc in self.documents]
        whole_dense = self._make_dense_index().fit(whole_texts, name="whole_passages_v2")

        units = build_smart_units(self.documents)
        unit_dense = self._make_dense_index().fit(
            [unit.context for unit in units], name="smart_units_v3"
        )

        bm25_texts = [format_passage_bm25_text(doc) for doc in self.documents]
        bm25_accented = BM25Index(bm25_texts, tokenizer=accented_bm25_tokens)
        bm25_ascii = BM25Index(bm25_texts, tokenizer=ascii_bm25_tokens)

        self.retriever = WholePassageHybridRetriever(
            documents=self.documents,
            units=units,
            whole_dense=whole_dense,
            unit_dense=unit_dense,
            bm25_accented=bm25_accented,
            bm25_ascii=bm25_ascii,
            candidate_pool_size=self.candidate_pool_size,
        )
        self.planner = QueryPlanner(
            llm=self.llm,
            model=self.llm_model,
            enabled=self.use_query_planner,
        )
        self.selector = EvidenceSelector(
            llm=self.llm,
            model=self.llm_model,
            max_core_ids=self.max_core_ids,
        )
        self.composer = GroundedComposer(
            llm=self.llm,
            model=self.llm_model,
            fewshot_count=self.fewshot_count,
        )
        enable_bank = self.use_train_fewshot and len(self.documents) >= 20
        self.fewshot_bank = TrainFewShotBank(
            dense_factory=self._make_dense_index,
            enabled=enable_bank,
        )

    def run(self, query: str) -> RAGOutput:
        if (
            self.retriever is None
            or self.planner is None
            or self.selector is None
            or self.composer is None
        ):
            raise RuntimeError("Pipeline chưa được index. Gọi index(corpus) trước.")

        plan = self.planner.plan(query)

        candidates, hybrid_ids, score_map, query_vec = self.retriever.retrieve_candidates(plan)

        decision = self.selector.select(plan, candidates, hybrid_ids)

        core_docs = [self.doc_by_id[doc_id] for doc_id in decision.core_ids if doc_id in self.doc_by_id]
        if decision.status != "UNANSWERABLE" and not core_docs:
            core_docs = candidates[: min(4, len(candidates))]

        fewshots = (
            self.fewshot_bank.select(
                raw_query=plan.raw_query,
                normalized_query=plan.normalized_query,
                query_vector=query_vec,
                status=decision.status,
                k=self.fewshot_count,
            )
            if self.fewshot_bank is not None
            else list(_FALLBACK_EXAMPLES[: self.fewshot_count])
        )
        answer, compose_raw = self.composer.compose(plan, decision, core_docs, fewshots)

        retrieved_ids: list[int] = []
        for doc_id in [*decision.core_ids, *decision.extra_ids, *hybrid_ids]:
            if doc_id in self.doc_by_id and doc_id not in retrieved_ids:
                retrieved_ids.append(doc_id)
                if len(retrieved_ids) >= self.config.max_retrieved:
                    break
        if not retrieved_ids:
            retrieved_ids = [self.documents[0].id]

        if plan.scope == "MULTI_CLUB":
            retrieved_ids = self._ensure_multi_club_top5(retrieved_ids, candidates, plan)

        self.last_debug = {
            "query": query,
            "normalized_query": plan.normalized_query,
            "scope": plan.scope,
            "search_queries": plan.search_queries,
            "candidate_ids": [d.id for d in candidates],
            "status": decision.status,
            "core_ids": decision.core_ids,
            "extra_ids": decision.extra_ids,
            "retrieved_ids": retrieved_ids,
            "top_hybrid_scores": score_map,
            "selector_raw": decision.raw_output,
            "composer_raw": compose_raw,
            "answer": answer,
        }
        return RAGOutput(answer=answer, retrieved_ids=retrieved_ids)

    def _ensure_multi_club_top5(
        self,
        retrieved_ids: list[int],
        candidates: list[Document],
        plan: QueryPlan,
    ) -> list[int]:
        if plan.scope != "MULTI_CLUB":
            return retrieved_ids

        detected = [c for c in detect_clubs(plan.normalized_query) if c in MAIN_CLUBS]
        target_clubs = detected if len(detected) >= 2 else list(MAIN_CLUBS)

        top5 = retrieved_ids[:5]
        top5_clubs = {self.doc_by_id[doc_id].club for doc_id in top5 if doc_id in self.doc_by_id}

        missing_clubs = [c for c in target_clubs if c not in top5_clubs]
        if not missing_clubs:
            return retrieved_ids

        insertions: list[int] = []
        for club in missing_clubs:
            found_id = None
            for doc_id in retrieved_ids[5:]:
                if self.doc_by_id.get(doc_id) and self.doc_by_id[doc_id].club == club:
                    found_id = doc_id
                    break
            if found_id is None:
                for doc in candidates:
                    if doc.club == club and doc.id not in top5 and doc.id not in insertions:
                        found_id = doc.id
                        break
            if found_id is not None and found_id not in insertions:
                insertions.append(found_id)

        if not insertions:
            return retrieved_ids

        head = list(top5)
        rem = [i for i in retrieved_ids[5:] if i not in insertions]

        for ins_id in insertions:
            if len(head) >= 5:
                club_counts: dict[str, int] = {}
                for d_id in head:
                    c = self.doc_by_id[d_id].club
                    club_counts[c] = club_counts.get(c, 0) + 1

                replace_idx = None
                for idx in range(len(head) - 1, 1, -1):
                    d_id = head[idx]
                    c = self.doc_by_id[d_id].club
                    if club_counts.get(c, 0) > 1:
                        replace_idx = idx
                        break
                if replace_idx is not None:
                    removed = head.pop(replace_idx)
                    rem.insert(0, removed)
                    head.append(ins_id)
                else:
                    head.append(ins_id)
            else:
                head.append(ins_id)

        new_order: list[int] = []
        for doc_id in [*head, *rem]:
            if doc_id not in new_order:
                new_order.append(doc_id)
        return new_order[: self.config.max_retrieved]

__all__ = [
    "EvidenceDecision",
    "EvidenceSelector",
    "GroundedComposer",
    "NeoRAGPipeline",
    "QueryPlan",
    "QueryPlanner",
    "TrainFewShotBank",
    "WholePassageHybridRetriever",
]
