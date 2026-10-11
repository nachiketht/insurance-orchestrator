# State and memory

Written before any agent code. The pipeline is a fixed sequence: intake, enrichment, risk scoring, recommendation. The steps are known in advance, so this is a workflow, not an agent that chooses its own next step.

## Choice: a shared case record

Each application is one `CaseRecord`. It holds the raw application plus one section per agent: `intake`, `enrichment`, `risk`, and `recommendation`.

The agents do not share a bag of mutable state. Each agent is a function that receives only the sections it needs and returns its own result. Only the pipeline writes that result onto the case record.

A shared record is a better fit than passing each output straight into the next function:

- The recommendation needs intake, enrichment, and the risk score together. Message passing would make every agent forward data it does not use, which is a case record with extra steps.
- If a later step fails, the earlier sections are still on the record for a human reviewer.
- One JSON file per case is the audit trail.

## What each agent reads and writes

| Agent | Reads | Writes |
| --- | --- | --- |
| `intake_agent` | the raw application | `case.intake`: normalized name, date of birth, address, product (`auto`, `home`, or `life`), coverage amount, income, claims text, and `missing_fields` |
| `enrichment_agent` | `case.intake` and a bureau lookup | `case.enrichment`: credit band, prior-claim count, and asset age copied from the bureau, plus flags and notes |
| `risk_scoring_agent` | `case.intake` and `case.enrichment` | `case.risk`: risk factors and a score from 0 to 100, higher meaning worse |
| `recommendation_agent` | `case.intake`, `case.enrichment`, and `case.risk` | `case.recommendation`: `approve`, `deny`, or `refer`, plus reasons |

The pipeline also sets `case.status` (`completed` or `escalated`) and, on failure, `case.escalation`.

The bureau lookup is a local table keyed by `applicant_id`. It is not a model call. Unknown ids come back as `no_hit`.

## Memory

Memory is one case. Applications are independent, so nothing is kept across cases: no conversation history and no shared store. Each model call sends only that call's messages. A later test runs two cases back to back and checks that nothing from the first appears in the second.

## Parallelism

The bureau lookup needs only `applicant_id` from the raw application, so it could run beside intake. Risk scoring cannot start until enrichment has finished, because it reads the enrichment result. The lookup is an in-memory read, so running it in parallel would not save meaningful time. The four model calls stay in order.

## Budgets, set before measuring

Local model: Ollama `qwen3:8b`. API cost is $0. These are the limits Part 4 will compare with measured traces.

| Step | Input tokens | Output tokens | Latency |
| --- | ---: | ---: | ---: |
| Intake | 700 | 150 | 8 s |
| Enrichment | 600 | 120 | 7 s |
| Risk scoring | 800 | 250 | 12 s |
| Recommendation | 900 | 200 | 10 s |
| Whole case | 3,000 | 720 | 37 s |

Use-case limits, also fixed before measuring:

- Real time, an applicant waiting on a quote: 2 seconds or less per case.
- Overnight batch, on this one machine, one case at a time: 10,000 applications inside 8 hours, which means a mean of 2.88 seconds or less per case.
