# AutoRAG Vega-Lite retrieval comparison

Rows: **103**

## `0`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.71 | 0.344 | 0.434052 | 0.236667 | 0.142 | 0.00155 |
| 2 | `` | `lexical_retrieval` | `` |  | `` | `` |  |  |  |  |  |  |

## `qa_all`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 3 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.5 | 0.333333 | 0.0019 |
| 2 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.333333 | 0.2 | 0.002187 |
| 3 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.333333 | 0.2 | 0.00215 |
| 4 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.181818 | 0.1 | 0.00323 |
| 5 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.181818 | 0.1 | 0.00351 |
| 6 | `` | `` | `BM25` | 20 | `porter_stemmer` | `` | 1 | 1 | 1 | 0.095238 | 0.05 | 0.005531 |
| 7 | `` | `` | `BM25` | 5 | `space` | `` | 1 | 0.995 | 0.996309 | 0.333333 | 0.2 | 0.001916 |
| 8 | `` | `` | `BM25` | 10 | `space` | `` | 1 | 0.995 | 0.996309 | 0.181818 | 0.1 | 0.003119 |
| 9 | `` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 10 | `retrieve_node_line` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 11 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 12 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 13 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 14 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 15 | `` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 16 | `retrieve_node_line` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 17 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 18 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 19 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 20 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |

## `qa_chart_pattern`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.793262 | 0.844546 | 0.181818 | 0.1 | 0.002997 |
| 2 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.793262 | 0.844546 | 0.181818 | 0.1 | 0.002702 |
| 3 | `` | `` | `BM25` | 20 | `porter_stemmer` | `` | 1 | 0.793262 | 0.844546 | 0.095238 | 0.05 | 0.004871 |
| 4 | `` | `` | `BM25` | 10 | `space` | `` | 1 | 0.788762 | 0.841499 | 0.181818 | 0.1 | 0.002636 |
| 5 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.97 | 0.7885 | 0.834088 | 0.323333 | 0.194 | 0.00169 |
| 6 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.97 | 0.7885 | 0.834088 | 0.323333 | 0.194 | 0.002026 |
| 7 | `` | `` | `BM25` | 5 | `space` | `` | 0.98 | 0.785667 | 0.834604 | 0.326667 | 0.196 | 0.00217 |
| 8 | `` | `` | `BM25` | 3 | `porter_stemmer` | `` | 0.89 | 0.77 | 0.800949 | 0.445 | 0.296667 | 0.001429 |
| 9 | `` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 10 | `retrieve_node_line` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 11 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 12 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 13 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 14 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 15 | `` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 16 | `retrieve_node_line` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 17 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 18 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 19 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 20 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |

## `qa_instruction`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.386202 | 0.531398 | 0.181818 | 0.1 | 0.00324 |
| 2 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.386202 | 0.531398 | 0.181818 | 0.1 | 0.002565 |
| 3 | `` | `` | `BM25` | 20 | `porter_stemmer` | `` | 1 | 0.386202 | 0.531398 | 0.095238 | 0.05 | 0.00469 |
| 4 | `` | `` | `BM25` | 10 | `space` | `` | 1 | 0.385869 | 0.531092 | 0.181818 | 0.1 | 0.002393 |
| 5 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.71 | 0.344 | 0.434052 | 0.236667 | 0.142 | 0.001754 |
| 6 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.71 | 0.344 | 0.434052 | 0.236667 | 0.142 | 0.00154 |
| 7 | `` | `` | `BM25` | 5 | `space` | `` | 0.7 | 0.342 | 0.430184 | 0.233333 | 0.14 | 0.00144 |
| 8 | `` | `` | `BM25` | 3 | `porter_stemmer` | `` | 0.47 | 0.29 | 0.335949 | 0.235 | 0.156667 | 0.00122 |
| 9 | `` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 10 | `retrieve_node_line` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 11 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 12 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 13 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 14 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 15 | `` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 16 | `retrieve_node_line` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 17 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 18 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 19 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 20 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |

## `qa_mixed_technical`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.722155 | 0.788707 | 0.181818 | 0.1 | 0.002555 |
| 2 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.722155 | 0.788707 | 0.181818 | 0.1 | 0.002798 |
| 3 | `` | `` | `BM25` | 20 | `porter_stemmer` | `` | 1 | 0.722155 | 0.788707 | 0.095238 | 0.05 | 0.004793 |
| 4 | `` | `` | `BM25` | 10 | `space` | `` | 1 | 0.718766 | 0.786257 | 0.181818 | 0.1 | 0.00249 |
| 5 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.893333 | 0.7065 | 0.752773 | 0.297778 | 0.178667 | 0.001604 |
| 6 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 0.893333 | 0.7065 | 0.752773 | 0.297778 | 0.178667 | 0.001567 |
| 7 | `` | `` | `BM25` | 5 | `space` | `` | 0.89 | 0.702556 | 0.749135 | 0.296667 | 0.178 | 0.001465 |
| 8 | `` | `` | `BM25` | 3 | `porter_stemmer` | `` | 0.783333 | 0.681667 | 0.707736 | 0.391667 | 0.261111 | 0.001215 |
| 9 | `` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 10 | `retrieve_node_line` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 11 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 12 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 13 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 14 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 15 | `` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 16 | `retrieve_node_line` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 17 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 18 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 19 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 20 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |

## `qa_title_query`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 1 | 0.987 | 0.990178 | 0.333333 | 0.2 | 0.001736 |
| 2 | `` | `` | `BM25` | 5 | `porter_stemmer` | `` | 1 | 0.987 | 0.990178 | 0.333333 | 0.2 | 0.001715 |
| 3 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.987 | 0.990178 | 0.181818 | 0.1 | 0.00271 |
| 4 | `` | `` | `BM25` | 10 | `porter_stemmer` | `` | 1 | 0.987 | 0.990178 | 0.181818 | 0.1 | 0.002784 |
| 5 | `` | `` | `BM25` | 20 | `porter_stemmer` | `` | 1 | 0.987 | 0.990178 | 0.095238 | 0.05 | 0.00479 |
| 6 | `` | `` | `BM25` | 3 | `porter_stemmer` | `` | 0.99 | 0.985 | 0.986309 | 0.495 | 0.33 | 0.001295 |
| 7 | `` | `` | `BM25` | 10 | `space` | `` | 1 | 0.981667 | 0.986181 | 0.181818 | 0.1 | 0.002585 |
| 8 | `` | `` | `BM25` | 5 | `space` | `` | 0.99 | 0.98 | 0.982619 | 0.33 | 0.198 | 0.001519 |
| 9 | `` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 10 | `retrieve_node_line` | `lexical_retrieval` | `` | 3 | `` | `` |  |  |  |  |  |  |
| 11 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 12 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 13 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 14 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 15 | `` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 16 | `retrieve_node_line` | `lexical_retrieval` | `` | 20 | `` | `` |  |  |  |  |  |  |
| 17 | `` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 18 | `retrieve_node_line` | `lexical_retrieval` | `` | 5 | `` | `` |  |  |  |  |  |  |
| 19 | `` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |
| 20 | `retrieve_node_line` | `lexical_retrieval` | `` | 10 | `` | `` |  |  |  |  |  |  |

## `unknown_qa`

| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |
|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `retrieve_node_line` | `lexical_retrieval` | `` |  | `` | `` |  |  |  |  |  |  |
