# Cost and latency

Numbers below come from the three live traces in `traces/samples/`,
run on local Ollama `qwen3:8b`. The model was already loaded, so the
warm-up time was 0.0 seconds and is not included in the case times.
API fees for this run are $0.00. Electricity and the host machine are
not priced here.

The hosted column is a comparison only. It prices the same token counts
as claude-haiku-5.5 for prompts under 100,000 tokens:
$0.10 per million input tokens and
$0.50 per million output tokens.
Source: https://platform.claude.com/docs/en/about-claude/pricing
List prices dated 2026-10-07, checked 2026-10-11.
The counts come from Ollama's tokenizer, not Claude's, so the dollar
figure is approximate. Every call in these traces reported a non-zero
input count.

## Per case

| Case | Seconds | Input tokens | Output tokens | Local API cost | Hosted equivalent |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean_auto | 16.98 | 1229 | 393 | $0.00 | $0.000319 |
| high_risk_home | 19.07 | 1275 | 426 | $0.00 | $0.000340 |
| no_hit_life | 14.95 | 1192 | 325 | $0.00 | $0.000282 |
| Mean | 17.00 | | | $0.00 | $0.000314 |
| Max | 19.07 | | | | |

## Per step, mean of the three cases

| Step | Mean seconds | Budget seconds | Mean tokens | Token budget |
| --- | ---: | ---: | ---: | ---: |
| intake | 4.19 | 8 | 271 | 850 |
| enrichment | 1.87 | 7 | 293 | 720 |
| risk_scoring | 8.59 | 12 | 570 | 1050 |
| recommendation | 2.35 | 10 | 480 | 1100 |
| whole case | 17.00 | 37 | | 3,720 |

Every step is inside the budget written in `docs/design.md` before these
runs. Risk scoring is the slow step. It is about half of each case.

## Worked hosted cost

`clean_auto` used 1229 input tokens and
393 output tokens.

1229 * 0.10 / 1,000,000 = $0.0001229
393 * 0.50 / 1,000,000 = $0.0001965
Total = $0.0003194

## Verdict

Mean 17.00 seconds per case is 8.5 times the
2 second real-time budget. Real time fails. The four calls
run one after another, and the next agent needs the previous agent's full
JSON, so streaming the reply would not shorten the wait.

At 17.00 seconds each, 10,000 cases take 47.2 hours
on this one machine. The overnight window is 8 hours,
so that batch fails too. This machine can finish about 1,694 cases in
8 hours. Batch is still the right way to run the pipeline, because no
applicant is waiting on their own case. More machines, a smaller model,
or combining intake and enrichment into one call would raise that number.
The hosted equivalent is well under one cent per case, so cost is not
what makes either use case fail.
