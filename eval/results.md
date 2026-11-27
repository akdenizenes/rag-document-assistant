## Retrieval evaluation (2026-11-27)

49 labelled questions; a hit means a chunk from the correct document and page appears in the top-k results.

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Median latency |
|---|---:|---:|---:|---:|---:|
| Vector only (pgvector) | 63.3% | 71.4% | 81.6% | 0.695 | 16 ms |
| Full-text only (tsvector) | 34.7% | 49.0% | 55.1% | 0.430 | 15 ms |
| Hybrid (vector + FTS, RRF) | 57.1% | 75.5% | 83.7% | 0.672 | 18 ms |
| Hybrid + score filter (app default) | 57.1% | 75.5% | 83.7% | 0.672 | 19 ms |
| Hybrid + cross-encoder rerank | 83.7% | 89.8% | 91.8% | 0.868 | 290 ms |

### Language `en` (20 questions)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Vector only (pgvector) | 70.0% | 80.0% | 85.0% | 0.754 |
| Full-text only (tsvector) | 35.0% | 50.0% | 60.0% | 0.450 |
| Hybrid (vector + FTS, RRF) | 50.0% | 80.0% | 85.0% | 0.643 |
| Hybrid + score filter (app default) | 50.0% | 80.0% | 85.0% | 0.643 |
| Hybrid + cross-encoder rerank | 90.0% | 95.0% | 100.0% | 0.935 |

### Language `tr` (29 questions)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Vector only (pgvector) | 58.6% | 65.5% | 79.3% | 0.653 |
| Full-text only (tsvector) | 34.5% | 48.3% | 51.7% | 0.417 |
| Hybrid (vector + FTS, RRF) | 62.1% | 72.4% | 82.8% | 0.693 |
| Hybrid + score filter (app default) | 62.1% | 72.4% | 82.8% | 0.693 |
| Hybrid + cross-encoder rerank | 79.3% | 86.2% | 86.2% | 0.822 |

### Misses for `hybrid` (not in top 5)

- `q02` (tr): Bir yazılım ürününün geliştirilmesine ne yol açar?  → expected Software Products, p. [7]
- `q11` (en): Show an example of a product vision for a CRM product.  → expected Software Products, p. [14]
- `q19` (en): What are the two stages of prototyping?  → expected Software Products, p. [24]
- `q23` (tr): Artımlı geliştirmede hangi özellikler önce gerçekleştirilir?  → expected Agile Software Engineering, p. [5]
- `q25` (tr): Çevik ilkelerde 'değişimi kucakla' ne anlama gelir?  → expected Agile Software Engineering, p. [8]
- `q31` (tr): Sprint çıktısının müşteriye dağıtılabilecek kalitede olması ne demektir?  → expected Agile Software Engineering, p. [16, 18, 47]
- `q36` (tr): Product backlog öğeleri hangi durumlardan geçer?  → expected Agile Software Engineering, p. [24]
- `q47` (en): Why can't external communication simply be rotated among team members?  → expected Agile Software Engineering, p. [44]

<details><summary>Setup</summary>

- Documents / chunks in library: 2 / 115
- Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Chunking: 600 chars, 100 overlap
- Hybrid: RRF k=60, weights vector/FTS=0.85/0.15, candidates per branch ≥ 20
- Min cosine score: 0.1 (FTS-only: 0.05)
- Score filter: cutoff 0.5, keep ≥ 2
- Cross-encoder: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (15 candidates)

</details>

