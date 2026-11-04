# Roadmap

This page is for maintainers. It is the product's rolling roadmap and the only place that records planned and
deferred work and its state. The user documentation describes what the product does today; nothing on this page is
available until its row says so.

## Status values

**Contract**

- `pending`: the item needs a reviewed design before work starts; the link names the current source.
- `ready`: the linked design holds a reviewed contract for the item.
- `not needed`: the Work cell states the change completely.
- `—`: delivered before this roadmap; the code and user documentation define the behaviour.

**Implementation**

- `planned`: none of the item is in the product.
- `partly implemented`: part of it is; the evidence names which part.
- `implemented`: the item is in the product.

## Maintaining the roadmap

- One row per planned item. IDs never change and are never reused; a split gets new IDs and the old one leaves the
  table. A need in another repository is written `<repository>:<ID>`.
- A row changes in the commit that changes its item: its scope, its needs, its contract or its implementation. Its
  evidence names in-repository witnesses (tests, documents, reports); a row never names the commit that changes it.
- An implemented row may leave the table once a release has shipped it and no row needs it; the changelog keeps it.
- Status words live only on this page. Designs state requirements; the changelog says what each release shipped.
  When this page and a design disagree on order or needs, this page wins; on behaviour, the design wins.
- The page records product facts, not how the work is run: an effort's plans, reviews and CI results stay in its own
  records.
- An item moves from Deferred to Planned on the maintainer's decision, and gets its ID then.

## Planned

Nothing is scheduled yet.

| ID | Work | Needs | Contract | Implementation | Evidence |
|---|---|---|---|---|---|

## Deferred

Not scheduled. One entry per item: what it is, and why it is not scheduled or what would make it worth doing.

- **Large configs on real geography**: qualify all four configurations in `topogen_configs/` using the Census and road
  inputs and newly generated geography, retaining a reproducible maintainer report; saved small graphs do not exercise
  that path, and the full run needs an appropriate data and execution window.
- **Python 3.14 and free-threaded NetLab**: add Python 3.14 and its free-threaded build to the supported dependency and
  CI matrix after checking the complete build, simulation and metrics path; the present CI covers 3.11–3.13, so upstream
  package support alone does not establish this support.
- **Delivered-path TE latency**: evaluate latency from actual delivered path costs for single-source TE demands if
  consumers need path-based percentiles rather than marginal-cost summaries; path decomposition can change the
  percentiles and previously cost about 1.9 times as much placement time, so agree the meaning and measure the cost
  first.
- **Long mock autoresearch run**: reproduce and diagnose `netlab autoresearch run --backend mock` stopping early,
  including backend response consumption by reflection and the circuit breaker, and make a requested run finish or
  explain its intended stopping condition; the exact historical stop at 42 experiments has not been re-established on
  the current release.
- **Publishing trigger and assets**: document the published-GitHub-release trigger and its required wheel, source
  distribution and `SHA256SUMS`, matching `.github/workflows/publish.yml`; the workflow already enforces the three
  assets, and maintainers need the preparation instructions before their next release.
