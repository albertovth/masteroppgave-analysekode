# Python-kode for masterprosjekt i offentlig forvaltning

Dette repositoriet inneholder sentral analysekode brukt i et masterprosjekt om OCAI-profiler og KI-strategiprofiler, med Dirichlet-modellering, ILR/CLR-transformasjoner og tilhørende sanity checks.

## Struktur

`run/` inneholder Python-koden for analyse, sanity checks og hjelpefunksjoner.

## Innhold

Følgende filer inngår:
- `analyse_skript_masteroppgave11.py`
- `model_metrics.py`
- `export_helpers.py`
- `smoke_test_pipeline.py`
- `sanity_checks_ilr.py`
- `sanity_checks_dirichlet.py`
- `sanity_checks_model_metrics.py`
- `run_real.sh`

## Kjøre sanity checks

Grunnleggende metrikksjekker:
`python run/sanity_checks_model_metrics.py`

Full røyktest:
`python run/smoke_test_pipeline.py`

## Reproduserbarhet

Den fullstendige empiriske analysen bygger på data som ikke kan deles offentlig.

Dette repositoriet inneholder derfor ikke:
- begrensede/reelle datafiler
- private logger
- genererte outputfiler fra kjøringer på reelle data
- personlige arbeidsnotater

## Formål

Formålet med repositoriet er å dokumentere analyseopplegget og gjøre sentral kode tilgjengelig for innsyn og etterprøvbarhet.
