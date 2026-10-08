-- The reviewed local-code mappings of the HDSS fixture (tests/fixtures/hdss/), as the vocab_map
-- stage would leave them in source_to_concept_map, so the fixture builds without an LLM.
INSERT INTO {etl}.source_to_concept_map (source_code, source_concept_id, source_vocabulary_id,
                                         source_code_description, target_concept_id, target_vocabulary_id)
VALUES
    ('AL', 0, 'SRC_DRUG', 'artemether-lumefantrine', 2000003001, 'RxNorm'),
    ('PARACETAMOL', 0, 'SRC_DRUG', 'paracetamol', 2000003002, 'RxNorm'),
    ('AMOXICILLIN', 0, 'SRC_DRUG', 'amoxicillin', 2000003003, 'RxNorm'),
    ('ARTESUNATE', 0, 'SRC_DRUG', 'artesunate injection', 2000100021, 'RxNorm Extension'),
    ('MEASLES VACCINE', 0, 'SRC_DRUG', 'measles vaccine', 2000100030, 'CVX'),
    ('HB', 0, 'SRC_LAB', 'haemoglobin', 2000004001, 'LOINC'),
    ('MRDT', 0, 'SRC_LAB', 'malaria rapid test', 2000004002, 'LOINC'),
    ('PREG', 0, 'SRC_LAB', 'pregnancy test', 2000100004, 'SNOMED'),
    ('POS', 0, 'SRC_RESULT', 'positive', 2000006001, 'LOINC'),
    ('NEG', 0, 'SRC_RESULT', 'negative', 2000006002, 'LOINC'),
    ('g/dL', 0, 'SRC_UNIT', 'gram per decilitre', 2000005001, 'UCUM'),
    ('MALARIA', 0, 'SRC_VA_CAUSE', 'malaria', 2000001001, 'SNOMED'),
    ('HIV', 0, 'SRC_VA_CAUSE', 'HIV/AIDS', 2000001006, 'SNOMED'),
    ('TB', 0, 'SRC_VA_CAUSE', 'tuberculosis', 2000001007, 'SNOMED'),
    -- a cause the verbal autopsy records as two: one code with two targets (etlKeyFanOut)
    ('HIV/TB', 0, 'SRC_VA_CAUSE', 'HIV/AIDS and tuberculosis', 2000001006, 'SNOMED'),
    ('HIV/TB', 0, 'SRC_VA_CAUSE', 'HIV/AIDS and tuberculosis', 2000001007, 'SNOMED');
