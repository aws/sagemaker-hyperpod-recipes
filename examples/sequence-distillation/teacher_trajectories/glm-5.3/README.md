# Published GLM-5.3 teacher trajectories

The teacher trajectories behind the reported results. GLM-5.3 on Amazon Bedrock played ShoppingBench's agent on the 425 training problems (`split/*.json`), two samples per problem. Set `USE_PUBLISHED_TEACHER_TRAJECTORIES = True` in notebook 1 to start from these files instead of calling Bedrock. Notebook 1 unpacks them into its generation folder and skips every problem they cover.

| File | Problems | Solved (kept by rejection sampling) |
|---|---|---|
| `rollout_teacher_s0_product.jsonl.gz` | 147 | 104 |
| `rollout_teacher_s0_shop.jsonl.gz` | 145 | 103 |
| `rollout_teacher_s0_voucher.jsonl.gz` | 133 | 90 |
| `rollout_teacher_s1_product.jsonl.gz` | 147 | 103 |
| `rollout_teacher_s1_shop.jsonl.gz` | 145 | 122 |
| `rollout_teacher_s1_voucher.jsonl.gz` | 133 | 90 |

- **Teacher:** `us.zai.glm-5.3`, `reasoning_effort: high`, temperature 1.0, top_p 0.95, at most 8,192 tokens per step, ShoppingBench's `rollout.md` system prompt.
- **Format:** one trajectory per line, in the harness's rollout format. Each line is a list of steps (prompt, completion, extra_info), the same format notebook 1 writes.
- **Result:** notebook 1 on these files gives `seqkd_train.jsonl` (2,600 rows) and `seqkd_val.jsonl` (130 rows), the dataset of the reported student.
