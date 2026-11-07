# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cài đặt và lệnh

```bash
python scripts/install.py          # cách chuẩn của repo; pip install -r requirements.txt tương đương
python scripts/install.py --local  # thêm sentence-transformers (kéo PyTorch/CUDA) cho embedding local
cp .env.example .env               # rồi điền OPENAI_API_KEY do BTC cấp
```

Chạy pipeline (`<split>` ∈ `train`, `public_test`, `private_test`, hoặc đường dẫn `.jsonl`):

```bash
neorag run train        # predict + evaluate (chỉ train có ground truth)
neorag predict public_test
neorag evaluate train
neorag validate public_test
```

`--no-judge` bỏ các metric cần LLM (nhanh, không tốn budget). `--pipeline module.path:ClassName` chọn pipeline.

**CLI mặc định chạy baseline, chatbot mặc định chạy solution — hai default khác nhau:**

```bash
neorag run train --pipeline neorag.solution.cag:CAGPipeline
neorag predict public_test --pipeline neorag.solution.cag:CAGPipeline
python -m streamlit run app/chat.py                      # mặc định CAGPipeline
NEORAG_PIPELINE=neorag.solution.pipeline:RAGAnythingPipeline python -m streamlit run app/chat.py
NEORAG_PIPELINE=neorag.pipelines.baseline:BaselinePipeline python -m streamlit run app/chat.py
```

Test (hoàn toàn offline, dùng `FakeLLM`/`FakeEmbedder` trong `tests/conftest.py`):

```bash
pytest
pytest tests/test_solution_core.py::test_pipeline_offline_end_to_end
```

Lint: `ruff check` (line-length 100, rule `E,F,W,I,B,UP,SIM`, bỏ `E501`). Không có CI, không có pre-commit hook.

## Ràng buộc cuộc thi

- LLM duy nhất được phép là `claude-haiku-5-5` qua gateway `https://proptit.dekthe.dev/v1` (OpenAI-compatible). **Không được truyền `temperature`** — gateway trả lỗi. `LLMClient.chat` bỏ hẳn tham số khi `temperature=None`.
- Embedding là `bge-m3` qua cùng gateway, hoặc model open-source chạy local.
- Budget key là $5. Mọi call LLM được cache SQLite trong `.cache/`, nên chạy lại không tốn thêm.
- Submission hợp lệ: 1–10 `retrieved_ids` duy nhất **trỏ tới id passage gốc trong corpus** (không phải chunk id), answer không rỗng và ≤ 2000 ký tự, đúng một dòng mỗi câu hỏi. `predict()` tự dedup và cắt theo `max_retrieved`.
- Điểm = 0.5 × retrieval (Recall/MRR/NDCG@5) + 0.5 × generation (String Presence, ROUGE-L, Groundedness, Response Relevancy, 1 − Noise Sensitivity), cộng abstention. `configs/eval.toml` là cấu hình chính thức của BTC — không sửa.

## Kiến trúc

Repo là starter kit của BTC: dữ liệu + baseline + bộ chấm chính thức + chatbot mẫu. Thí sinh thay phần pipeline.

`RAGPipeline` (`src/neorag/pipelines/base.py`) là interface chung: `prepare()`, `index(corpus)`, `run(query) -> RAGOutput(answer, retrieved_ids)`. `load_pipeline("module:Class")` resolve bằng importlib và **gọi constructor không kèm kwargs** — nên mọi tùy chỉnh phải nằm trong default của constructor hoặc đọc từ env. Ba entry point cùng đi qua interface này: CLI (`src/neorag/cli.py`), `app/chat.py`, và test.

- `src/neorag/llm.py` — `LLMClient` bọc OpenAI SDK trỏ gateway, cache SQLite keyed SHA-256 của (endpoint, model, payload).
- `src/neorag/evaluate.py` + `src/neorag/metrics/` — bộ chấm chính thức. Prompt của LLM judge cố định trong `metrics/judge.py` để điểm giữa các thí sinh so sánh được.
- `src/neorag/pipelines/baseline.py` — embed nguyên passage, cosine top-10, một prompt một lượt. Cố ý chưa tối ưu.

### Pipeline solution (`src/neorag/solution/`)

```
corpus → corpus.py build_chunks   (chunk 1 câu + cửa sổ 2 câu, context = "CLB — topic. text")
       → index.py DenseIndex      (bge-m3, cache .npz trong .cache/solution/embeddings/)
         index.py BM25Index       (tokenizer bỏ dấu, index trên NGUYÊN passage)
         graph.py KnowledgeGraph  (regex entity + IDF term, thuần tất định)
       → query.py QueryAnalyzer/QueryExpander (1 call LLM sinh ≤3 query + query per-CLB tất định + 1 call HyDE)
       → retrieve.py HybridRetriever (0.65 dense / 0.35 BM25, fuse max + mean, LLM rerank top-20)
       → generate.py AnswerGenerator (draft → verifier pass)
```

Những điểm đọc riêng từng file sẽ dễ hiểu nhầm:

- **Dense index trên chunk, BM25 index trên nguyên passage.** Chủ ý: dense cần hạt mịn để bắt mệnh đề, BM25 cần hạt thô để không loãng IDF.
- Điểm dense của một passage là **trung bình top-2 chunk score** của nó (`retrieve.py` `_dense_doc_scores`). Sự tồn tại của chunk cửa sổ 2 câu là để luôn có ứng viên thứ hai ổn định cho phép tính này.
- **Graph neighbours hiện không bao giờ vào prompt.** `HybridRetriever.retrieve` chỉ chèn chúng sau `context_docs` phần tử đầu (`len(selected) < max_retrieved`), còn `AnswerGenerator._context` chỉ đọc `documents[:context_docs]`. Với default (8/10) graph docs rơi vào index 8–9, tức là chỉ xuất hiện trong `retrieved_ids`, không tới LLM. Đây là lỗi tiềm ẩn đã biết, không phải thiết kế.
- `SolutionConfig` (`solution/config.py`) là dataclass frozen, đọc toàn bộ từ env (`NEORAG_*`, `LLM_MODEL`, `EMBEDDING_MODEL`). Vì `load_pipeline` không truyền kwargs, **tinh chỉnh runtime chỉ qua env**.
- Không có vector database và không có cross-encoder: toàn bộ là NumPy in-memory (cosine = dot product sau khi L2-normalize) cộng BM25 tự viết.

## Dữ liệu

- `data/corpus.jsonl`: 118 passage. Id 1–78 ProPTIT, 79–84 S2B chung, 85–96 MPC, 97–118 GCC.
- `data/train/questions.jsonl` có ground truth (`relevant_ids`, `key_facts`, `answerable`, `difficulty`, `club`). `public_test` chỉ có `id` + `query`. `private_test` phát hành 08/11.
- `data/*/gold.jsonl` và `data/private_test/questions.jsonl` bị gitignore (BTC giữ).
- Câu trả lời đúng nên giữ nguyên mốc thời gian đi kèm thông tin ("đợt 2026", nhiệm kỳ cụ thể) thay vì trình bày như quy định cố định.

## Trạng thái nhánh `rag_et`

`src/neorag/solution/`, `docs/solution.md` và `tests/test_solution_core.py` **chưa được track** — cần `git add` thủ công trước khi commit.
