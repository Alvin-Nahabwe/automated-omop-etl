-- Extra concepts for the HDSS fixture (tests/fixtures/hdss/), loaded into the fixture
-- vocabulary schema after load_athena. Synthetic, like the demo fixture vocabulary
-- (omop_agent/demo/fixtures/vocab/README.md): ids in the local range from 2000100000, made-up
-- relationships; ICD-10 codes are real codes, used only as lookup keys.
INSERT INTO {vocab}.vocabulary (vocabulary_id, vocabulary_name, vocabulary_reference, vocabulary_version,
                                vocabulary_concept_id)
SELECT v.* FROM (VALUES
    ('RxNorm Extension', 'RxNorm Extension (synthetic fixture)', 'omop-agent HDSS fixture', 'fixture', 2000100900),
    ('CVX', 'CVX (synthetic fixture)', 'omop-agent HDSS fixture', 'fixture', 2000100901)
) v (vocabulary_id, vocabulary_name, vocabulary_reference, vocabulary_version, vocabulary_concept_id)
WHERE NOT EXISTS (SELECT 1 FROM {vocab}.vocabulary x WHERE x.vocabulary_id = v.vocabulary_id);

INSERT INTO {vocab}.concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, standard_concept,
                             concept_code, valid_start_date, valid_end_date, invalid_reason)
VALUES
    -- standard targets in the domains a diagnosis list or a lab list also holds
    (2000100001, 'Normal pregnancy', 'Observation', 'SNOMED', 'Clinical Finding', 'S', 'FX2000100001',
     DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100002, 'Patient currently pregnant', 'Observation', 'SNOMED', 'Clinical Finding', 'S', 'FX2000100002',
     DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100003, 'Insertion of intrauterine contraceptive device', 'Procedure', 'SNOMED', 'Procedure', 'S',
     'FX2000100003', DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100004, 'Pregnancy test result', 'Observation', 'SNOMED', 'Observable Entity', 'S', 'FX2000100004',
     DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100005, 'Immigrant', 'Observation', 'SNOMED', 'Social Context', 'S', 'FX2000100005',
     DATE '1970-01-01', DATE '2099-12-31', NULL),
    -- ICD-10 source codes: Z34.9 maps to an observation, O98.7 to a condition and an observation,
    -- Z30.1 to a procedure
    (2000100101, 'Encounter for supervision of normal pregnancy, unspecified', 'Observation', 'ICD10', 'ICD10 code',
     NULL, 'Z34.9', DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100102, 'Human immunodeficiency virus [HIV] disease complicating pregnancy, childbirth and the puerperium',
     'Condition', 'ICD10', 'ICD10 code', NULL, 'O98.7', DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100103, 'Encounter for insertion of intrauterine contraceptive device', 'Procedure', 'ICD10', 'ICD10 code',
     NULL, 'Z30.1', DATE '1970-01-01', DATE '2099-12-31', NULL),
    -- a drug whose only ingredient is in RxNorm Extension, and a vaccine without an ingredient
    (2000100020, 'artesunate', 'Drug', 'RxNorm Extension', 'Ingredient', 'S', 'OMOP2000100020',
     DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100021, 'artesunate 60 MG Injectable Solution', 'Drug', 'RxNorm Extension', 'Clinical Drug', 'S',
     'OMOP2000100021', DATE '1970-01-01', DATE '2099-12-31', NULL),
    (2000100030, 'measles virus vaccine', 'Drug', 'CVX', 'CVX', 'S', '05', DATE '1970-01-01', DATE '2099-12-31', NULL);

INSERT INTO {vocab}.concept_relationship (concept_id_1, concept_id_2, relationship_id, valid_start_date,
                                          valid_end_date, invalid_reason)
SELECT r.concept_id_1, r.concept_id_2, r.relationship_id, DATE '1970-01-01', DATE '2099-12-31', NULL
FROM (VALUES
    (2000100101, 2000100001, 'Maps to'), (2000100001, 2000100101, 'Mapped from'),
    (2000100102, 2000001006, 'Maps to'), (2000001006, 2000100102, 'Mapped from'),
    (2000100102, 2000100002, 'Maps to'), (2000100002, 2000100102, 'Mapped from'),
    (2000100103, 2000100003, 'Maps to'), (2000100003, 2000100103, 'Mapped from'),
    (2000100021, 2000100020, 'RxNorm has ing'), (2000100020, 2000100021, 'RxNorm ing of')
) r (concept_id_1, concept_id_2, relationship_id)
UNION ALL  -- every standard concept maps to itself
SELECT c, c, rel, DATE '1970-01-01', DATE '2099-12-31', NULL
FROM unnest(ARRAY[2000100001, 2000100002, 2000100003, 2000100004, 2000100005, 2000100020, 2000100021,
                  2000100030]) c
CROSS JOIN (VALUES ('Maps to'), ('Mapped from')) m (rel);

INSERT INTO {vocab}.concept_ancestor (ancestor_concept_id, descendant_concept_id, min_levels_of_separation,
                                      max_levels_of_separation)
SELECT c, c, 0, 0
FROM unnest(ARRAY[2000100001, 2000100002, 2000100003, 2000100004, 2000100005, 2000100020, 2000100021,
                  2000100030]) c
UNION ALL
SELECT 2000100020, 2000100021, 1, 1;
