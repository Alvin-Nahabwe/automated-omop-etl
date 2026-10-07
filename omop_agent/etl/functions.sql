-- Helper functions of the generated ETL SQL, installed in the ETL schema ({etl}) by
-- cdm.ddl.create_etl_tables. They are (re)installed only when omop_agent_functions_version()
-- in the schema differs from this file's version, so a build that runs as another role does not
-- need to own them; the role needs EXECUTE on them (granted to PUBLIC by default).
--
-- Value counters. A compiled INSERT counts, per CDM field (its "site"), the values it could not
-- load as written, in the transaction-local settings omop_agent.<category>__<site>, which the
-- runner reads at the end of the step (etl.runner). Each one is only called for a row it counts
-- (on the NULL branch of a CASE or COALESCE), so a clean row costs nothing. They are internal:
-- spec SQL may not call them (sqlguard).

-- add 1 to the counter <what> of <site>
CREATE OR REPLACE FUNCTION {etl}.omop_count(site text, what text) RETURNS boolean
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  IF site IS NOT NULL THEN
    PERFORM set_config('omop_agent.' || what || '__' || site,
      (coalesce(nullif(current_setting('omop_agent.' || what || '__' || site, true), ''), '0')::bigint + 1)::text,
      true);
  END IF;
  RETURN true;
END $f$;

-- a row dropped because required fields are NULL: required_null of each NULL field, and
-- row.dropped once. Returns false, for the WHERE that drops the row.
CREATE OR REPLACE FUNCTION {etl}.omop_drop(fields text[], nulls boolean[]) RETURNS boolean
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  FOR i IN 1 .. coalesce(array_length(fields, 1), 0) LOOP
    IF nulls[i] THEN
      PERFORM {etl}.omop_count(fields[i], 'required_null');
    END IF;
  END LOOP;
  PERFORM {etl}.omop_count('row', 'dropped');
  RETURN false;
END $f$;

-- a NULL required value replaced by the CDM default (a concept: 0)
CREATE OR REPLACE FUNCTION {etl}.omop_default_int(site text, val integer) RETURNS integer
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  PERFORM {etl}.omop_count(site, 'defaulted');
  RETURN val;
END $f$;

-- a NULL required end replaced by the value it is derived from (the start); counted only when
-- that value exists
CREATE OR REPLACE FUNCTION {etl}.omop_derived_date(site text, val date) RETURNS date
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  IF val IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'defaulted');
  END IF;
  RETURN val;
END $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_derived_ts(site text, val timestamp) RETURNS timestamp
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  IF val IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'defaulted');
  END IF;
  RETURN val;
END $f$;

-- installed last: a schema that answers with this version has every function above
CREATE OR REPLACE FUNCTION {etl}.omop_agent_functions_version() RETURNS text
LANGUAGE sql IMMUTABLE AS $f$ SELECT '{functions_version}' $f$;
