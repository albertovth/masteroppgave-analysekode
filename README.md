# Python-kode for masteroppgave i offentlig forvaltning

Dette repoet inneholder sentral analysekode brukt i en masteroppgave om OCAI-profiler og strategiprofiler for KI, med regresjonsanalyser, Dirichlet-modellering, ILR/CLR-transformasjoner og tilhørende konsistenskontroller.

## Struktur

`run/` inneholder Python-koden for analyse, regresjonsanalyser, konsistenskontroller, oppstarttester og hjelpefunksjoner.

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
- `lda_fritekst.py`
- `lda_dio_bruksomrader_ki.py`

I tillegg inneholder repoet to separate skript for temamodeller (LDA) av fritekstsvar.

## Kjøre konsistenskontroller

Grunnleggende metrikksjekker:
`python run/sanity_checks_model_metrics.py`

Full oppstartstest:
`python run/smoke_test_pipeline.py`

## Datastruktur

Se `docs/STRUKTUR_TIL_DATAFIL.md` for strukturen datafilen må ha for at analysekoden skal fungere.

## Reproduserbarhet

Den fullstendige empiriske analysen bygger på data som ikke kan deles offentlig.

Dette repoet inneholder derfor ikke:
- begrensede/reelle datafiler
- private loggfiler
- genererte resultatfiler fra kjøringer på reelle data
- personlige arbeidsnotater

## Formål

Formålet med repoet er å dokumentere analyseopplegget og gjøre sentral kode tilgjengelig for innsyn og etterprøvbarhet.
