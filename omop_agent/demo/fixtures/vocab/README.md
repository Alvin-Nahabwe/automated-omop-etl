# Demo fixture vocabulary: synthetic, not Athena content

These files use the Athena download layout (tab-delimited, `YYYYMMDD` dates), so the
vocabulary loader reads them exactly like a real download. The content is **synthetic**. It
exists only so the demo and the tests can run without a licensed Athena download.

* Only these concept ids are real OMOP concepts: `0`, `8507` (MALE), `8532` (FEMALE),
  `9201` (Inpatient Visit), `9202` (Outpatient Visit), `9203` (Emergency Room Visit), and
  `32817` (EHR). Their `concept_code` values are placeholders.
* Every other concept uses the local-concept range (`>= 2,000,000,000`). Its name reads like
  the real vocabulary so that mapping behaves realistically, but ids, codes and relationships
  are made up. ICD-10 and LOINC codes are real codes, used only as lookup keys.
* `vocabulary_id` values (`SNOMED`, `ICD10`, `RxNorm`, `LOINC`, `UCUM`, ...) keep their real
  names because the pipeline's logic depends on them (for example ICD-10 `Maps to` SNOMED).

For a real conversion, download the vocabularies you need from https://athena.ohdsi.org and
set `vocabulary.athena_dir` in `omop_agent.yaml`. Never load this fixture into a production
CDM.
