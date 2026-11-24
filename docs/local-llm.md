# Running omop-agent on a self-hosted open-weight model (CPU)

omop-agent needs an LLM for four things: interpreting the source profile, drafting the ETL design,
choosing standard concepts for source codes, and the repair agent's tool loop. Since this version
the default is a **self-hosted open-weight model** served on your own machine through an
OpenAI-compatible API (`llm.provider: openai_compatible`). Prompts never leave your
infrastructure, so no data goes to a third party and the privacy review gate is not required
(see "Privacy" below). Claude (`provider: anthropic`) remains available.

## Choosing a model for a CPU-only server

On a CPU, generation speed is set by memory bandwidth divided by the bytes read per generated
token. Mixture-of-experts (MoE) models read only their active experts per token, so a large MoE
model with few active parameters is both better and faster than a dense model of similar size.
Prompt processing (prefill) on a CPU runs at roughly 100 to 250 tokens per second, so long prompts
cost real time; see "Speed" below.

| Server RAM | Model | Licence | Weights (4-bit) | Notes |
|---|---|---|---|---|
| 96 GB or more | **gpt-oss-120b** (117B total, 5.1B active) | Apache 2.0 | ~63 GB (MXFP4) | **The default.** Mature support in llama.cpp, Ollama and vLLM; reliable JSON-schema output; reasoning level settable (low/medium/high). |
| 64 GB | Qwen3.6-35B-A3B (35B, 3B active) | Apache 2.0 | ~21 GB | Fast on CPU; weaker on long design answers. |
| 24-32 GB | gpt-oss-20b (21B, 3.6B active) | Apache 2.0 | ~12 GB | Same chat format as the 120b. Good for testing the set-up; check quality with `llm-check` and the demo before real use. |
| 160 GB or more | Qwen3.8-Flash-Next (125B, 6B active) | Qwen Community License | ~120 GB | Stronger on tool use than gpt-oss-120b in published benchmarks, but llama.cpp support was recent and unstable in November 2026. Re-evaluate once it is stable. |

These are the options as of November 2026. Open-weight models improve quickly: rather than trusting
this table, run `omop-agent llm-check` and the demo (below) on any candidate, and compare the
mappings and the design against the recorded reference answers.

Use a 4-bit quantisation (MXFP4 for gpt-oss; Q4_K_M or IQ4_XS for others). 8-bit is barely
better on these tasks and about twice as slow on a CPU.

## Serving the model

### llama.cpp (recommended for CPU)

Build `llama-server` from https://github.com/ggml-org/llama.cpp (or use the
`ghcr.io/ggml-org/llama.cpp:server` image), download the GGUF weights from Hugging Face, and run:

```bash
llama-server -m /models/gpt-oss-120b-MXFP4.gguf \
  --jinja -c 32768 -np 1 -t 32 --host 127.0.0.1 --port 8080
```

* `--jinja` uses the model's own chat template. It is required: without it llama-server does not
  return tool calls, and the repair agent cannot work.
* `-c 32768` is the context length. The design prompts are the longest the pipeline sends.
* `-np 1` serves one request at a time: omop-agent calls the model sequentially, and one slot
  gets the whole context.
* `-t` is the number of threads: the number of physical cores (not hyper-threads).
* Keep the server on `127.0.0.1` (or a private address) and do not expose it to the internet. To
  require a key, add `--api-key <key>` and set `OMOP_AGENT_LLM_API_KEY` for omop-agent.
* On a two-socket server, `numactl --interleave=all llama-server ...` (or llama.cpp's `--numa`
  options) is usually 20-30 % faster. [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), a
  CPU-optimised fork with the same command line, is often faster still.
* llama-server caches the prompt prefix. omop-agent keeps each stage's system prompt identical
  across calls, so after the first call of a stage only the new part of each prompt is processed.

### Ollama

```bash
ollama pull gpt-oss:120b
ollama serve   # http://127.0.0.1:11434
```

Then set `base_url: http://127.0.0.1:11434/v1` and `model: gpt-oss:120b`. Ollama's default
context is small: set `OLLAMA_CONTEXT_LENGTH=32768` (or a `num_ctx` in a Modelfile).

### vLLM (if a GPU becomes available)

```bash
vllm serve openai/gpt-oss-120b --enable-auto-tool-choice --tool-call-parser openai
```

## Configuring omop-agent

```yaml
llm:
  provider: openai_compatible
  base_url: http://127.0.0.1:8080/v1   # llama-server; Ollama: http://127.0.0.1:11434/v1
  model: gpt-oss-120b                  # the name the server knows the model by
  reasoning_effort: high               # gpt-oss: low | medium | high (unset: the server's default)
  # temperature: 0.2
  # max_tokens: 16384                  # longest answer
  # timeout_seconds: 3600              # one request; CPU answers can take many minutes
  # structured_output: json_schema     # json_schema | json_object | prompt
  # structured_retries: 2              # invalid answers are sent back with the errors
  # extra_body: {chat_template_kwargs: {reasoning_effort: high}}   # anything else the server reads
  # record_dir: ./recordings           # keep the answers for replay and audit
```

How answers are kept to the pipeline's schemas:

* **Constrained decoding.** Every structured call sends its JSON Schema as `response_format`. The
  server compiles it to a grammar, so the model can only produce JSON of that shape.
* **Validate and retry.** An answer that still fails validation (a value out of range, a rule the
  grammar cannot express) goes back to the model with the validation errors, up to
  `structured_retries` times. Every attempt is in the call's record under `work/llm_calls/`.
* **Mechanical fixes.** Mistakes whose right answer follows from the CDM specification alone
  are fixed in code before validation, so a slow CPU model does not spend a revision call on
  them: a missing surrogate primary key gets `auto_id`, helper functions written `etl.try_...`
  get the `{etl}` schema placeholder, and a foreign key to another CDM table set to `0` (provider,
  care site, location) is left NULL. Each fix is listed under the spec's assumptions
  (`omop_agent/design/tidy.py`).
* **Code checks everything anyway**, as with any model: concept ids must exist in your
  vocabulary, spec expressions are parsed and `EXPLAIN`ed, and the repair agent's tools validate
  their input.

## Checking a model: `omop-agent llm-check`

```bash
omop-agent llm-check -c omop_agent.yaml
```

This uses synthetic prompts only, so it is safe before any review. It checks that the server
accepts every answer schema the pipeline uses, that the model maps three test codes correctly (one
of which has no right candidate), and that a tool call round trip works. It prints the time each
check took. Then run the demo end to end against the model:

```bash
omop-agent demo ./demo-local --db-url "$OMOP_AGENT_DB_URL" --provider openai_compatible \
  --llm-url http://127.0.0.1:8080/v1 --llm-model gpt-oss-120b
```

The demo's report shows whether the CDM passes the gate and how many repair iterations it
needed. Compare `demo-local/work/mappings/*.csv` and `etl_spec.yaml` with a replay run
(`--provider replay`) to see where the model differs from the reference answers.

## Speed

A rough guide for gpt-oss-120b on a recent two-socket Xeon or EPYC server (llama.cpp, 4-bit): about
15-20 generated tokens per second, and 100-250 prompt tokens per second. A project of the size of
the Nairobi HDSS needs on the order of 30-80 model calls (one annotation, one plan, one design
call per CDM table, one mapping call per 20-30 codes, and a few repair turns), so expect the
LLM-driven stages to take one to several hours on a CPU. That is fine for an ETL that is designed
once and reviewed by people, and the build, the checks and re-runs do not call the model.

The calls are independent of each other's timing, so a slow server only makes the stages slower.
Raise `timeout_seconds` if a single call (a large design table) times out.

## Privacy

With `provider: openai_compatible` the privacy review gate depends on where the model runs:

* `base_url` on this machine or a private network (`127.0.0.1`, `localhost`, `10.x`, `172.16-31.x`,
  `192.168.x`, `*.internal`, `*.local`): prompts stay on your infrastructure, so no review is
  required.
* any other address: the review is required, as with Claude. If that address is your own server
  (an internal DNS name, say), set `llm.on_premises: true`.

The data-minimisation rules apply whatever the model: no patient rows, small-cell suppression,
identifier columns withheld, and the repair agent can only count.

## Improving quality: retrieval (RAG)

A smaller open model knows less than a frontier model, but much of what these tasks need is
looked up, not reasoned. omop-agent retrieves where it matters most:

* **Concept mapping is retrieval-first**: candidates come from your Athena vocabulary (exact,
  lexical and trigram search), and the model only picks among real candidates or asks for a
  re-search with better terms. It cannot invent a concept id.
* **The CDM conventions** for each target table (the field-level ETL conventions of the OMOP CDM
  v5.4 specification) are put into each design prompt.
* **The mapping memory** (`mapping.memory_path`) keeps every mapping a person approved, across
  projects. `omop-agent init` sets it to `~/.omop-agent/mapping_memory.sqlite` (in the Docker
  deployment `/srv/data/mapping_memory.sqlite`, shared by all projects), or to
  `$OMOP_AGENT_MAPPING_MEMORY`.
  * A term whose description reviewers already mapped (the same text after normalising case,
    spacing and punctuation; every reviewer chose the same concept; the concept is still a valid
    standard concept of a domain the field accepts) takes that concept with no model call. The
    decision says who approved it in which project. Set `mapping.memory_auto_approve: false` to
    have such decisions wait for a reviewer.
  * For the other terms, the concepts reviewers chose for the three most similar remembered terms
    join the candidates, and the prompt lists them as reviewed examples.
  * Local codes without a word of three or more letters ("1", "M") are never matched: they mean
    different things in different sources.
  * Only decisions a person approved are kept (a reviewer's own concept, or a suggestion approved
    in the web app), only for terms seen at least `privacy.min_cell_count` times, and each
    project's entries are replaced whenever its mappings are written, so a corrected mapping
    replaces the old one. With an external model (`anthropic`, or a server that is not on your
    network), the remembered concepts are still offered but other projects' term descriptions
    are not put into prompts.
  * The memory is written by `map`, `apply-mappings` and `run`; `omop-agent mapping-memory` records
    the current project's reviewed mappings and shows the memory's size.
* **Planned**: semantic concept search (a small local embedding model and `pgvector`) so that
  candidates are found by meaning as well as spelling; approved designs from earlier projects
  retrieved as examples for the design stage; and the data-quality check descriptions and known
  fixes retrieved for the repair agent.

Retrieved text lengthens prompts, and on a CPU prompt processing is the slow part, so each
addition is measured against the demo and the HDSS test set before it is enabled by default.
