# Struktur til datafil

Dette dokumentet beskriver datastrukturen til Excel-filen som analysekoden er laget for å lese.

## Filformat

Datafilen skal være en Excel-fil (`.xlsx`).

## Generell struktur

Arbeidsboken skal ha 9 ark:

1. `Bakgrunn`
2. `Kontroll`
3. `OCAI - dominerende egenskaper`
4. `OCAI - strategiske prioritering`
5. `OCAI - suksesskriterier`
6. `Strategi - dominerende egenskap`
7. `Strategi - strategiske priorite`
8. `Strategi - suksesskriterier`
9. `Åpne spørsmål`

Alle ark skal ha overskriftsrad i rad 1.

Alle ark skal ha `ID` som første kolonne.

## Arkstruktur

### `Bakgrunn`

Arket skal ha 6 kolonner:

- `ID`
- `Departement`
- `Ansiennitet`
- `Alder`
- `Kjønn`
- `Stilling`

### `Kontroll`

Arket skal ha data i 6 kolonner:

- `ID`
- `A`
- `B`
- `C`
- `D`
- `E`

### `OCAI - dominerende egenskaper`

Arket skal ha 5 kolonner:

- `ID`
- `Klan`
- `Adhockrati`
- `Marked`
- `Hierarki`

Kolonnene `Klan`, `Adhockrati`, `Marked` og `Hierarki` skal summere til 100 per rad.

### `OCAI - strategiske prioritering`

Arket skal ha 5 kolonner:

- `ID`
- `Klan`
- `Adhockrati`
- `Marked`
- `Hierarki`

Kolonnene `Klan`, `Adhockrati`, `Marked` og `Hierarki` skal summere til 100 per rad.

### `OCAI - suksesskriterier`

Arket skal ha 5 kolonner:

- `ID`
- `Klan`
- `Adhockrati`
- `Marked`
- `Hierarki`

Kolonnene `Klan`, `Adhockrati`, `Marked` og `Hierarki` skal summere til 100 per rad.

### `Strategi - dominerende egenskap`

Arket skal ha 5 kolonner:

- `ID`
- `Opportunist`
- `Entreprenør`
- `Spekulant`
- `Konservativ`

Kolonnene `Opportunist`, `Entreprenør`, `Spekulant` og `Konservativ` skal summere til 100 per rad.

### `Strategi - strategiske priorite`

Arket skal ha 5 kolonner:

- `ID`
- `Opportunist`
- `Entreprenør`
- `Spekulant`
- `Konservativ`

Kolonnene `Opportunist`, `Entreprenør`, `Spekulant` og `Konservativ` skal summere til 100 per rad.

### `Strategi - suksesskriterier`

Arket skal ha 5 kolonner:

- `ID`
- `Opportunist`
- `Entreprenør`
- `Spekulant`
- `Konservativ`

Kolonnene `Opportunist`, `Entreprenør`, `Spekulant` og `Konservativ` skal summere til 100 per rad.

### `Åpne spørsmål`

Arket skal ha data i 3 kolonner:

- `ID`
- `A`
- `B`

## Krav til struktur

For at analysekoden skal fungere, skal datafilen:

- være en `.xlsx`-fil
- ha de 9 arkene med navnene oppgitt over
- ha `ID` som første kolonne i alle ark
- ha overskriftsrad i rad 1 i alle ark
- ha de kolonnenavnene som er oppgitt over
- ha fire profilkolonner i hvert av de tre OCAI-arkene og de tre strategiarkene
- ha profilsummer lik 100 per rad i de seks ipsative arkene
