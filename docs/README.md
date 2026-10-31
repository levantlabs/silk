# silk documentation — which page answers which question

Use this index to find guides, API references, and records of where silk's
components and data come from. Start with the quickstart if you are new to silk.

## Start here

| Your question | Read |
| --- | --- |
| I'm new — how do I build and simulate a circuit? | [quickstart.md](quickstart.md) |
| I'm an agent about to write silk code. | [agent-guide.md](agent-guide.md) |
| I'm changing silk's internals — who owns what? | [architecture.md](architecture.md) |
| What does this function/class take and return? | [api-reference.md](api-reference.md) |
| Is there already an example that does this? | [examples.md](examples.md) |
| What's broken or missing right now? | [known_issues.md](known_issues.md) |
| What work is queued, and where is its detail? | Maintainers: `backlog task list --plain` (Backlog.md); public release: the [issue tracker](https://github.com/levantlabs/silk/issues). |
| What can silk do, and where is it going? | [capability-roadmap.md](capability-roadmap.md) |

## Running designs

| Your question | Read |
| --- | --- |
| How do I sweep a parameter, or explore a design space? | [optimizer.md](optimizer.md) — define parameters with `Variable`, sample and apply values with `VariableSpace`, and automate evaluations with `Optimizer` |
| How do I measure gain, noise, jitter, ENOB…? | [metrics-library.md](metrics-library.md) |
| How do I declare probes, bind metrics, and read or plot runs and corners? | [probes.md](probes.md) — runnable ideal-RC example, current probes, analysis selection and result lifetime |
| Which ngspice `.measure` form should agree with which silk metric, and where do the conventions differ? | [metrics-ngspice-measure-catalog.md](metrics-ngspice-measure-catalog.md) |
| How do I run a test as a Monte Carlo campaign — mismatch, process, yield, what causes the spread? | [monte-carlo.md](monte-carlo.md) — workflow and current limits; worked example `mc_diff_pair_offset.py` |
| How do I collect a module's structure, measurements, and specs in a report? | [reports.md](reports.md) |
| How do I get gm/Id/Vth/region per device from a run? | [operating-point.md](operating-point.md) — `Module.save_device_params` + `SpecReport.op_table()` |
| How do I declare specs and reconcile them with measurements? | [design_stage_bridge_user_guide.md](design_stage_bridge_user_guide.md) |
| The simulator "succeeded" — do I believe it? | [simulation-error-handling.md](simulation-error-handling.md) |
| My test declares two analyses — which one does a metric read? | [multi-analysis-tests.md](multi-analysis-tests.md) |
| My run writes a huge rawfile. How do I save less? | [simulator-save-list.md](simulator-save-list.md) |
| How do I record runs, evidence and results in a database? | [database.md](database.md) |

## Checking designs

| Your question | Read |
| --- | --- |
| What do the lint rules check, and what is coverage? | [lint_and_coverage_user_guide.md](lint_and_coverage_user_guide.md) |
| How does silk guess a port's role from its name? | [authoring-port-role-rules.md](authoring-port-role-rules.md) |

## PDKs and simulators

| Your question | Read |
| --- | --- |
| How do I install a PDK, and which are supported? | [installing-the-pdk.md](installing-the-pdk.md) |
| Where do I tell silk my PDK, ngspice or Xyce location? | [settings.md](settings.md) — the settings file (`~/.config/silk/silk.env`), precedence, and checking it with the doctor |
| How do I add support for a new PDK? | [adding-a-pdk.md](adding-a-pdk.md) |
| Is my device-details YAML correct? | [pdk-validation.md](pdk-validation.md) |
| What are Xyce's support limits? | [known_issues.md](known_issues.md) |

## Waveforms

| Your question | Read |
| --- | --- |
| How do I open and drive the waveform explorer? | [waveform-explorer.md](waveform-explorer.md) |
| How do I embed the explorer in my own page? | [explorer-integration.md](explorer-integration.md) |
| I want just the reader/plot core for a browser app. | [browser-waveform-core.md](browser-waveform-core.md) |
| What licence is the nutmeg reader, and where is it from? | [mit-nutmeg-reader-provenance.md](mit-nutmeg-reader-provenance.md) |

## Tooling, integration and process

| Your question | Read |
| --- | --- |
| What command-line tools ship with silk? | [cli.md](cli.md) |
| I'm writing a tool that *generates* silk Python. | [codegen-contract.md](codegen-contract.md) |
| I'm writing a tool that *reads* silk's schema APIs. | [schema-export.md](schema-export.md) |
| What metrics are available? | [metrics-library.md](metrics-library.md) — functions and usage; see [schema-export.md](schema-export.md) for the JSON format |
| What style do I write silk code in? | [coding-conventions.md](coding-conventions.md) |

## Reference assets

- `images/` contains figures used by these pages.
- `reference/` contains the generated API documentation configuration.
