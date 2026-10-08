-- Helper functions of the generated ETL SQL, installed in the ETL schema ({etl}) by
-- cdm.ddl.create_etl_tables. They are (re)installed only when omop_agent_functions_version()
-- in the schema differs from this file's version, so a build that runs as another role does not
-- need to own them; the role needs EXECUTE on them (granted to PUBLIC by default).
--
-- Value counters. A compiled INSERT counts, per CDM field (its "site"), the values it could not
-- load as written, in the transaction-local settings omop_agent.<category>__<site>, which the
-- runner reads at the end of the step (etl.runner). Each one is only called for a row it counts
-- (on the NULL branch of a CASE or COALESCE), so a clean row costs nothing. They are internal:
-- spec SQL may not call them (sqlguard), nor any other omop_* function. The conversions below
-- ("Tolerant conversion") use them; their uncounted try_* forms may be called by spec SQL.

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

-- ---------------------------------------------------------------------------------------------
-- Tolerant conversion. The compiler rewrites the hard casts and to_date / to_timestamp calls of
-- column and expression fields (etl.tolerant) into the omop_as_*_at front ends below: a value that
-- cannot be read becomes NULL and is counted at the field's site (invalid), instead of failing the
-- whole table. Dates are read strictly (31/02/2019 is invalid, not 2019-03-03). An unknown day or
-- month (00, 97-99, or a template without DD / MM) is completed by the field's policy
-- (etl.policies.partial_policy) and counted as imputed, or left NULL and counted as partial_null:
--   first / last / mid  the first day, the last day, or the 15th (June 15 when the month is unknown)
--   birth               a known month without a day: the 1st; an unknown month: NULL (THEMIS)
--   month / year        only the month / the year is used (month_of_birth, year_of_birth): kept,
--                       not counted
--   reject              NULL
-- An imputed date is never later than omop_agent.imputation_cap (the configured data extract
-- date, else the current date; etl.session), unless the known part of the date is.
-- Offsets ('Z', '+03:00') follow etl.timezone (the setting omop_agent.timezone): when it is set,
-- timestamps and dates are converted to that zone (offset_converted); unset, the wall clock is
-- kept as written (offset_ignored). Only an offset that is not the zone's own (unset: the
-- session's) is counted: the zone's own changes nothing. A ::timestamptz cast, and to_timestamp
-- with a TZH / OF template, keep the instant the offset gives; to_timestamp with another template
-- reads the wall clock as a timestamp does, then in the session zone.
--
-- The try_* functions are the same conversions without counters (STABLE or IMMUTABLE, PARALLEL
-- SAFE): the build's count queries use them, and spec SQL may use them in keys and joins. No
-- function here raises for any input value, and none needs an exception block (which a parallel
-- worker cannot run): every field is range-checked before it is used. The clean values of the
-- common layouts (ISO, also with fractional seconds or an offset, and numeric or month-name
-- templates with one-character separators) take SQL fast paths, which call no counter; the other
-- values go through one plpgsql parser (omop_walk, about 13 us a value).

-- the tokens of a to_date / to_timestamp template (ARRAY['ISO'] for 'ISO'); literals are
-- 'L<text>'. An unsupported template is an error: with a constant template it is raised when the
-- statement is planned (EXPLAIN), not per row. The compiler checks literal templates first.
CREATE OR REPLACE FUNCTION {etl}.omop_fmt_spec(fmt text) RETURNS text[]
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $f$
DECLARE out text[] := '{}'; i int := 1; n int := length(fmt); u text; t text; w int; j int;
BEGIN
  IF upper(fmt) = 'ISO' THEN RETURN ARRAY['ISO']; END IF;
  IF strpos(fmt, '%') > 0 THEN
    RAISE EXCEPTION 'unsupported date template "%": a strftime pattern, not a PostgreSQL template', fmt
      USING HINT = 'Write the PostgreSQL template, e.g. DD/MM/YYYY for %d/%m/%Y.';
  END IF;
  WHILE i <= n LOOP
    u := upper(substr(fmt, i, 7));
    IF u LIKE 'FM%' THEN i := i + 2; CONTINUE; END IF;
    t := NULL;
    SELECT tk, wd INTO t, w FROM (VALUES ('YYYY', 'Y4', 4), ('YY', 'Y2', 2), ('MONTH', 'MON', 5), ('MON', 'MON', 3),
        ('MM', 'M', 2), ('MI', 'MI', 2), ('DD', 'D', 2), ('HH24', 'H', 4), ('HH12', 'H12', 4), ('HH', 'H12', 2),
        ('SS', 'S', 2), ('MS', 'F', 2), ('US', 'F', 2), ('AM', 'AP', 2), ('PM', 'AP', 2), ('TZH:TZM', 'TZ', 7),
        ('TZH', 'TZ', 3), ('OF', 'TZ', 2)) v(p, tk, wd)
      WHERE u LIKE p || '%' ORDER BY length(p) DESC LIMIT 1;
    IF t IS NOT NULL THEN
      out := out || t; i := i + w;
    ELSIF substr(fmt, i, 1) = '"' THEN
      j := strpos(substr(fmt, i + 1), '"');
      IF j = 0 THEN RAISE EXCEPTION 'unsupported date template "%": unterminated quote', fmt; END IF;
      out := out || ('L' || substr(fmt, i + 1, j - 1)); i := i + j + 1;
    ELSIF substr(fmt, i, 1) ~ '[[:alnum:]]' THEN
      RAISE EXCEPTION 'unsupported date template "%" at "%"', fmt, substr(fmt, i)
        USING HINT = 'Supported: YYYY YY MM DD Mon Month HH24 HH12 HH MI SS MS US AM PM TZH:TZM TZH OF FM, '
                     'separators and "quoted" text, or ISO.';
    ELSE
      out := out || ('L' || substr(fmt, i, 1)); i := i + 1;
    END IF;
  END LOOP;
  RETURN out;
END $f$;

-- the shape of template tokens (omop_fmt_spec, without an offset) as a regular expression, for
-- omop_walk: ARRAY[regex, the numeric tokens in order, the alphabetic tokens in order, the widths
-- of the numeric tokens in each run of digits]. Day, month and time fields may be unpadded unless
-- two numbers touch (DDMMYYYY: one run '4.2.2' read by width; '0' is a run of one field). A
-- separator matches any one separator, as in to_date ('DD/MM/YYYY' reads 05-03-2019). A date-only
-- template also takes a trailing time ('15/06/2019 10:30'), and a template with a time also takes
-- the date alone (midnight).
CREATE OR REPLACE FUNCTION {etl}.omop_spec_rx(spec text[]) RETURNS text[]
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $f$
DECLARE
  rx text := ''; ntoks text[] := '{}'; atoks text[] := '{}';
  runs text[] := '{}'; w text; tok text; i int; n int := coalesce(array_length(spec, 1), 0); first_time int;
  opt_from int; date_after_time boolean := false;
  time_toks constant text[] := '{H,H12,MI,S,F,AP}';
  num_toks constant text[] := '{Y4,Y2,M,D,H,H12,MI,S,F}';
BEGIN
  FOR i IN 1 .. n LOOP
    IF spec[i] = ANY (time_toks) THEN
      first_time := coalesce(first_time, i);
    ELSIF first_time IS NOT NULL AND spec[i] IN ('Y4', 'Y2', 'M', 'D', 'MON') THEN
      date_after_time := true;
    END IF;
  END LOOP;
  IF first_time > 1 AND NOT date_after_time THEN  -- the time part and the separators before it
    opt_from := first_time;
    WHILE opt_from > 2 AND left(spec[opt_from - 1], 1) = 'L' LOOP opt_from := opt_from - 1; END LOOP;
  END IF;
  FOR i IN 1 .. n LOOP
    tok := spec[i];
    IF i = opt_from THEN rx := rx || '(?:'; END IF;
    IF left(tok, 1) = 'L' THEN
      rx := rx || CASE WHEN substr(tok, 2) = ' ' THEN ' +'
                       WHEN length(tok) = 2 AND substr(tok, 2) !~ '[[:alnum:]]' THEN '[^[:alnum:][:space:]]'
                       ELSE regexp_replace(substr(tok, 2), '([^[:alnum:] ])', '\\\1', 'g') END;
      -- letters of a quoted literal ("T") keep their place among the alphabetic tokens
      atoks := atoks || array_fill('L'::text, ARRAY[coalesce(array_length(
                 array_remove(regexp_split_to_array(substr(tok, 2), '[^[:alpha:]]+'), ''), 1), 0)]);
    ELSIF tok IN ('MON', 'AP') THEN
      rx := rx || CASE tok WHEN 'MON' THEN '[[:alpha:]]+\.?' ELSE '[AaPp]\.?[Mm]\.?' END;
      atoks := atoks || tok;
    ELSE
      rx := rx || CASE tok
        WHEN 'Y4' THEN '[0-9]{4}'
        WHEN 'Y2' THEN '[0-9]{2}'
        WHEN 'F' THEN '[0-9]+'
        ELSE CASE WHEN spec[i + 1] = ANY (num_toks) OR spec[i - 1] = ANY (num_toks) THEN '[0-9]{2}'
                  ELSE '[0-9]{1,2}' END
      END;
      ntoks := ntoks || tok;
      w := CASE tok WHEN 'Y4' THEN '4' WHEN 'F' THEN '9' ELSE '2' END;
      IF spec[i - 1] = ANY (num_toks) THEN
        runs[array_length(runs, 1)] := runs[array_length(runs, 1)] || '.' || w;
      ELSE
        runs := runs || CASE WHEN spec[i + 1] = ANY (num_toks) THEN w ELSE '0' END;
      END IF;
    END IF;
  END LOOP;
  IF opt_from IS NOT NULL THEN
    rx := rx || ')?';
  ELSIF first_time IS NULL THEN
    rx := rx || '(?:(?:[Tt]| +)[0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?)?';
    ntoks := ntoks || '{H,MI,S,F}'::text[];
    runs := runs || '{0,0,0,0}'::text[];
  END IF;
  RETURN ARRAY['^' || rx || '$', ntoks::text, atoks::text, runs::text];
END $f$;

-- a template for omop_walk (ARRAY['ISO'] for 'ISO'): omop_spec_rx of its tokens, then 'tz' when it
-- reads an offset (TZH:TZM, TZH, OF; it must come last, and omop_walk reads it before the rest),
-- then, for a date-only template with a day, omop_spec_rx of the template without the day: a
-- value without its day ('06/1975' for DD/MM/YYYY) is a partial date.
CREATE OR REPLACE FUNCTION {etl}.omop_fmt_rx(fmt text) RETURNS text[]
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $f$
DECLARE spec text[] := {etl}.omop_fmt_spec(fmt); k int; tz text := ''; out text[];
BEGIN
  IF spec = ARRAY['ISO'] THEN RETURN spec; END IF;
  k := array_position(spec, 'TZ');
  IF k IS NOT NULL THEN
    IF EXISTS (SELECT 1 FROM unnest(spec[k + 1:]) t WHERE left(t, 1) <> 'L') THEN
      RAISE EXCEPTION 'unsupported date template "%": the offset (TZH, OF) must come last', fmt;
    END IF;
    WHILE k > 1 AND left(spec[k - 1], 1) = 'L' LOOP k := k - 1; END LOOP;
    spec := spec[1:k - 1];
    tz := 'tz';
  END IF;
  out := {etl}.omop_spec_rx(spec) || tz;
  k := array_position(spec, 'D');
  IF k IS NOT NULL AND NOT spec && '{H,H12,MI,S,F,AP}'::text[] THEN  -- (a day between separators only:
    IF k > 1 AND left(spec[k - 1], 1) = 'L' THEN                      -- 6 digits are not DDMMYYYY)
      out := out || {etl}.omop_spec_rx(spec[1:k - 2] || spec[k + 1:]);  -- the day and the separator before it
    ELSIF left(spec[k + 1], 1) = 'L' THEN
      out := out || {etl}.omop_spec_rx(spec[1:k - 1] || spec[k + 2:]);  -- or after it
    END IF;
  END IF;
  RETURN out;
END $f$;

-- the layout of a template the SQL fast path reads: 'ISO' for ISO and the ISO templates; for a
-- template of one-character separators between numeric YYYY, MM (or Mon), DD, HH24, MI and SS
-- fields: the number of fields, the positions of YYYY, MM, DD, HH24, MI and SS (0: absent), 'N'
-- if the month is a name (else '-'), the separators in order, then the other separator
-- characters, which are read as the first one ('DD/MM/YYYY HH24:MI' gives '5321450-// : :'). NULL
-- for any other template (the slow path reads it).
CREATE OR REPLACE FUNCTION {etl}.omop_fast_layout(fmt text) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $f$
DECLARE spec text[]; n int; i int; pos int[] := '{0,0,0,0,0,0}'; k int; seps text := ''; mon boolean := false;
BEGIN
  IF upper(fmt) IN ('ISO', 'YYYY-MM-DD', 'YYYY-MM-DD HH24:MI:SS', 'YYYY-MM-DD HH24:MI', 'YYYY-MM-DD"T"HH24:MI:SS',
                    'YYYY-MM-DD"T"HH24:MI') THEN
    RETURN 'ISO';
  END IF;
  spec := {etl}.omop_fmt_spec(fmt);
  n := coalesce(array_length(spec, 1), 0);
  IF n % 2 = 0 OR n > 13 THEN RETURN NULL; END IF;
  FOR i IN 1 .. n LOOP
    IF i % 2 = 0 THEN
      IF length(spec[i]) <> 2 OR substr(spec[i], 2) ~ '[[:alnum:]]' THEN RETURN NULL; END IF;
      seps := seps || substr(spec[i], 2);
    ELSE
      k := array_position('{Y4,M,D,H,MI,S}'::text[], CASE spec[i] WHEN 'MON' THEN 'M' ELSE spec[i] END);
      IF k IS NULL OR pos[k] <> 0 THEN RETURN NULL; END IF;
      pos[k] := (i + 1) / 2;
      mon := mon OR spec[i] = 'MON';
    END IF;
  END LOOP;
  IF 0 IN (pos[1], pos[2], pos[3]) THEN RETURN NULL; END IF;  -- a complete date
  RETURN ((n + 1) / 2)::text || array_to_string(pos, '') || CASE WHEN mon THEN 'N' ELSE '-' END || seps
         || coalesce((SELECT string_agg(DISTINCT c, '') FROM regexp_split_to_table(seps, '') c WHERE c <> left(seps, 1)), '');
END $f$;

-- a timestamp from the text fields of a fast layout; NULL unless every field is valid. The
-- numeric fields are digits (the caller checked), except next to a month name (named). Each
-- field is referenced as few times as possible: the function is inlined, and so are its arguments.
CREATE OR REPLACE FUNCTION {etl}.omop_fast_make(y text, m text, d text, h text, i text, s text, named boolean)
RETURNS timestamp LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN length(y) = 4 AND length(m) IN (1, 2) AND length(d) IN (1, 2)
             AND coalesce(length(h), 1) IN (1, 2) AND coalesce(length(i), 1) IN (1, 2) AND coalesce(length(s), 1) IN (1, 2)
             AND (NOT named OR translate(y || d || coalesce(h, '') || coalesce(i, '') || coalesce(s, ''),
                                         '0123456789', '') = '')
  THEN CASE WHEN y::int >= 1 AND m::int IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) AND coalesce(h::int, 0) <= 23
             AND coalesce(i::int, 0) <= 59 AND coalesce(s::int, 0) <= 59
             AND d::int BETWEEN 1 AND CASE m::int
                   WHEN 2 THEN CASE WHEN y::int % 4 = 0 AND (y::int % 100 <> 0 OR y::int % 400 = 0) THEN 29 ELSE 28 END
                   WHEN 4 THEN 30 WHEN 6 THEN 30 WHEN 9 THEN 30 WHEN 11 THEN 30 ELSE 31 END
    THEN make_timestamp(y::int, m::int, d::int, coalesce(h::int, 0), coalesce(i::int, 0), coalesce(s::int, 0)) END
END $f$;

-- the same from the fields of a value split once (several kinds of separator): a function call
-- costs less than splitting the value again for every reference to a field
CREATE OR REPLACE FUNCTION {etl}.omop_fast_fields(f text[], y int, m int, d int, h int, i int, s int, named boolean)
RETURNS timestamp LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $f$
BEGIN
  RETURN {etl}.omop_fast_make(f[y], CASE WHEN named THEN array_position(
           '{jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec}'::text[], lower(f[m]))::text ELSE f[m] END,
         f[d], f[h], f[i], f[s], named);
END $f$;

-- the complete, valid values of a fast layout other than ISO; NULL for anything else. With a
-- constant layout every substr() of it is folded when the statement is planned.
CREATE OR REPLACE FUNCTION {etl}.omop_fast_num(x text, lay text) RETURNS timestamp
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN lay IS NOT NULL AND lay <> 'ISO'
   AND CASE WHEN substr(lay, 8, 1) = 'N'
            THEN translate(x, '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', '')
            ELSE translate(x, '0123456789', '') END = substr(lay, 9, substr(lay, 1, 1)::int - 1)
  THEN CASE WHEN length(lay) > 7 + substr(lay, 1, 1)::int
    THEN {etl}.omop_fast_fields(
           string_to_array(translate(x, substr(lay, 8 + substr(lay, 1, 1)::int),
                                     repeat(substr(lay, 9, 1), length(lay) - 7 - substr(lay, 1, 1)::int)),
                           substr(lay, 9, 1)),
           substr(lay, 2, 1)::int, substr(lay, 3, 1)::int, substr(lay, 4, 1)::int, substr(lay, 5, 1)::int,
           substr(lay, 6, 1)::int, substr(lay, 7, 1)::int, substr(lay, 8, 1) = 'N')
    ELSE {etl}.omop_fast_make(
           split_part(x, substr(lay, 9, 1), substr(lay, 2, 1)::int),
           CASE WHEN substr(lay, 8, 1) = 'N'
                THEN array_position('{jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec}'::text[],
                                    lower(split_part(x, substr(lay, 9, 1), substr(lay, 3, 1)::int)))::text
                ELSE split_part(x, substr(lay, 9, 1), substr(lay, 3, 1)::int) END,
           split_part(x, substr(lay, 9, 1), substr(lay, 4, 1)::int),
           CASE WHEN substr(lay, 5, 1) <> '0' THEN split_part(x, substr(lay, 9, 1), substr(lay, 5, 1)::int) END,
           CASE WHEN substr(lay, 6, 1) <> '0' THEN split_part(x, substr(lay, 9, 1), substr(lay, 6, 1)::int) END,
           CASE WHEN substr(lay, 7, 1) <> '0' THEN split_part(x, substr(lay, 9, 1), substr(lay, 7, 1)::int) END,
           substr(lay, 8, 1) = 'N') END END $f$;

-- the complete, valid ISO dates and timestamps without an offset: YYYY-M-D, YYYY-MM-DD[ T]HH:MI[:SS]
-- (exactly 4 or 5 non-digits, each where the layout has it: every field cast below is of digits)
CREATE OR REPLACE FUNCTION {etl}.omop_fast_ts(x text) RETURNS timestamp
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN length(x) BETWEEN 8 AND 10 THEN {etl}.omop_fast_num(x, '3123000---')
  WHEN length(x) IN (16, 19) AND substr(x, 11, 1) IN (' ', 'T')
   AND translate(x, '0123456789', '') = '--' || substr(x, 11, 1) || CASE WHEN length(x) = 16 THEN ':' ELSE '::' END
   AND substr(x, 5, 1) = '-' AND substr(x, 8, 1) = '-' AND substr(x, 14, 1) = ':'
   AND (length(x) = 16 OR substr(x, 17, 1) = ':')
  THEN CASE WHEN substr(x, 6, 2) COLLATE "C" BETWEEN '01' AND '12' AND left(x, 4) COLLATE "C" >= '0001'
             AND substr(x, 9, 2)::int BETWEEN 1 AND CASE substr(x, 6, 2)::int WHEN 2 THEN
                   CASE WHEN left(x, 4)::int % 4 = 0 AND (left(x, 4)::int % 100 <> 0 OR left(x, 4)::int % 400 = 0)
                        THEN 29 ELSE 28 END
                 WHEN 4 THEN 30 WHEN 6 THEN 30 WHEN 9 THEN 30 WHEN 11 THEN 30 ELSE 31 END
             AND substr(x, 12, 2) COLLATE "C" <= '23' AND substr(x, 15, 2) COLLATE "C" <= '59'
             AND coalesce(nullif(substr(x, 18, 2), ''), '00') COLLATE "C" <= '59'
    THEN make_timestamp(left(x, 4)::int, substr(x, 6, 2)::int, substr(x, 9, 2)::int, substr(x, 12, 2)::int,
                        substr(x, 15, 2)::int, coalesce(nullif(substr(x, 18, 2), ''), '0')::float8) END END $f$;

-- the fast path of a layout (omop_fast_layout); NULL sends the value to the slow path
CREATE OR REPLACE FUNCTION {etl}.omop_fast(x text, lay text) RETURNS timestamp
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
  SELECT CASE WHEN lay = 'ISO' THEN {etl}.omop_fast_ts(x) ELSE {etl}.omop_fast_num(x, lay) END $f$;

-- the other complete, valid ISO timestamps: with fractional seconds or an offset ('Z', '+03',
-- '+0300', '+03:00'; |hh| <= 14), as ODK and most APIs write them. True: PostgreSQL's own input
-- functions read the value without error (x::timestamp drops the offset, x::timestamptz honours
-- it); NULL for any other value.
CREATE OR REPLACE FUNCTION {etl}.omop_iso_full(x text) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN length(x) BETWEEN 17 AND 40 AND left(x, 4) <> '0000'
   AND x ~ '^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])[T ]([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9](\.[0-9]{1,9})?)?([Zz]|[+-](0[0-9]|1[0-4])(:?[0-5][0-9])?)?$'
  THEN CASE WHEN substr(x, 9, 2) COLLATE "C" <= '28' OR substr(x, 9, 2)::int <= CASE substr(x, 6, 2)::int WHEN 2 THEN
              CASE WHEN left(x, 4)::int % 4 = 0 AND (left(x, 4)::int % 100 <> 0 OR left(x, 4)::int % 400 = 0)
                   THEN 29 ELSE 28 END
            WHEN 4 THEN 30 WHEN 6 THEN 30 WHEN 9 THEN 30 WHEN 11 THEN 30 ELSE 31 END THEN true END END $f$;

-- the site time zone (etl.timezone), else the session's; and whether etl.timezone is set
CREATE OR REPLACE FUNCTION {etl}.omop_zone() RETURNS text
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  SELECT coalesce(nullif(current_setting('omop_agent.timezone', true), ''), current_setting('TimeZone')) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_zone_set() RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  SELECT coalesce(current_setting('omop_agent.timezone', true), '') <> '' $f$;

-- a value counted as <what> at site, returned unchanged
CREATE OR REPLACE FUNCTION {etl}.omop_counted(site text, what text, val anyelement) RETURNS anyelement
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  PERFORM {etl}.omop_count(site, what);
  RETURN val;
END $f$;

-- the wall clock of an omop_iso_full value: with an offset, in etl.timezone when it is set (counted
-- as offset_converted), else as written (offset_ignored); an offset that is the zone's own (or,
-- unset, the session's) changes nothing and is not counted. NULL for any other value.
CREATE OR REPLACE FUNCTION {etl}.omop_iso_local_at(site text, x text) RETURNS timestamp
LANGUAGE sql VOLATILE AS $f$
SELECT CASE WHEN {etl}.omop_iso_full(x) THEN
  CASE WHEN translate(substr(x, 17), '0123456789:.', '') = '' THEN x::timestamp  -- no offset
       WHEN x::timestamptz = x::timestamp AT TIME ZONE {etl}.omop_zone() THEN x::timestamp
       WHEN {etl}.omop_zone_set()
         THEN {etl}.omop_counted(site, 'offset_converted', x::timestamptz AT TIME ZONE {etl}.omop_zone())
       ELSE {etl}.omop_counted(site, 'offset_ignored', x::timestamp) END END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_iso_local(x text) RETURNS timestamp
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN {etl}.omop_iso_full(x) THEN
  CASE WHEN {etl}.omop_zone_set() AND translate(substr(x, 17), '0123456789:.', '') <> ''
       THEN x::timestamptz AT TIME ZONE {etl}.omop_zone() ELSE x::timestamp END END $f$;

-- the slow path: a value read with the template (rx: omop_fmt_rx), else without its day (a partial
-- date), else as ISO 8601 (in any template, also the partial YYYY-MM and YYYY), else, under 'ISO'
-- only, as one of the other forms PostgreSQL's ::date reads unambiguously under DateStyle 'ISO,
-- YMD': year first (YYYYMMDD, YYYY/MM/DD) or with an English month name and a four-digit year
-- ('5 Mar 2019', 'March 5, 2019'). Words such as 'today' or 'epoch' are invalid. The shape is checked with one regular
-- expression and the fields are then taken from the runs of digits and letters, in order
-- (regexp_match with several groups is too slow for a per-value parser).
-- val: the date and time as written; off: its offset in minutes (NULL: none); st: ok | imputed |
-- partial_null | invalid (NULL for a blank value)
CREATE OR REPLACE FUNCTION {etl}.omop_walk(x text, rx text[], pol text, OUT val timestamp, OUT st text, OUT off int)
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$
DECLARE
  s text := btrim(x, E' \t\r\n'); tz text; ntoks text[]; atoks text[]; nums text[]; alphas text[]; mon text;
  runs text[]; parts text[]; r text; i int; j int; ws text[];
  -- numeric, not integer: a field is never cast to a type it could overflow before its range is checked
  y numeric; m numeric; d numeric; hh numeric; mi numeric; sec numeric; ap text; sgn int; um boolean; ud boolean;
  dim int; lo date; v date;
  tm constant text := '(?:(?:[Tt]| +)[0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?)?$';
BEGIN
  IF s IS NULL OR s = '' OR rx IS NULL THEN RETURN; END IF;
  st := 'invalid';
  IF length(s) > 64 THEN RETURN; END IF;
  -- an offset after the time: Z, +hh, +hhmm or +hh:mm (|hh| <= 14)
  IF strpos(s, ':') > 0 AND s ~ ':[0-9]{2}(?:[.,][0-9]+)? *(?:[Zz]|[+-][0-9]{2}(?::?[0-9]{2})?)$' THEN
    tz := substring(s FROM '(?:[Zz]|[+-][0-9]{2}(?::?[0-9]{2})?)$');
    s := rtrim(left(s, length(s) - length(tz)));
    IF upper(tz) = 'Z' THEN
      off := 0;
    ELSE  -- '+03', '+0300', '+03:00' -> '0300'
      sgn := CASE WHEN left(tz, 1) = '-' THEN -1 ELSE 1 END;
      tz := rpad(replace(substr(tz, 2), ':', ''), 4, '0');
      IF left(tz, 2)::int > 14 OR right(tz, 2)::int > 59 THEN RETURN; END IF;
      off := sgn * (left(tz, 2)::int * 60 + right(tz, 2)::int);
    END IF;
  END IF;
  IF rx[1] <> 'ISO' AND s ~ rx[1] THEN
    ntoks := rx[2]::text[];
    atoks := rx[3]::text[];
    runs := rx[4]::text[];
  ELSIF rx[6] IS NOT NULL AND s ~ rx[6] THEN  -- the template without its day: a partial date
    ntoks := rx[7]::text[];
    atoks := rx[8]::text[];
    runs := rx[9]::text[];
  ELSIF s ~ ('^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}' || tm) THEN
    ntoks := '{Y4,M,D,H,MI,S,F}';
  ELSIF s ~ '^[0-9]{4}(?:-[0-9]{1,2})?$' THEN
    ntoks := '{Y4,M}';
  ELSIF rx[1] = 'ISO' THEN
    IF s ~ '^[0-9]{8}$' THEN
      s := left(s, 4) || '-' || substr(s, 5, 2) || '-' || right(s, 2);
      ntoks := '{Y4,M,D}';
    ELSIF s ~ ('^[0-9]{4}[/.][0-9]{1,2}[/.][0-9]{1,2}' || tm) THEN
      ntoks := '{Y4,M,D,H,MI,S,F}';
    ELSIF s ~ ('^[0-9]{1,2}[ -]?[[:alpha:]]+\.?[ -]?,? *[0-9]{4}' || tm)
       OR s ~ ('^[[:alpha:]]+\.? +[0-9]{1,2},? +[0-9]{4}' || tm) THEN
      ntoks := '{D,Y4,H,MI,S,F}';
      atoks := '{MON}';
    ELSIF s ~ ('^[0-9]{4}[ -][[:alpha:]]+\.?[ -][0-9]{1,2}' || tm) THEN
      ntoks := '{Y4,D,H,MI,S,F}';
      atoks := '{MON}';
    END IF;
  END IF;
  IF ntoks IS NULL THEN off := NULL; RETURN; END IF;
  -- the shape matched: the runs of digits are the numeric fields in order (a missing time gives NULLs)
  nums := array_remove(regexp_split_to_array(s, '[^0-9]+'), '');
  IF strpos(runs::text, '.') > 0 THEN  -- numbers that touch (DDMMYYYY): split by width
    parts := '{}';
    FOR i IN 1 .. coalesce(array_length(nums, 1), 0) LOOP
      r := coalesce(runs[i], '0');
      IF r = '0' THEN
        parts := parts || nums[i];
      ELSE
        ws := string_to_array(r, '.');
        j := 1;
        FOR k IN 1 .. array_length(ws, 1) LOOP
          parts := parts || nullif(substr(nums[i], j, ws[k]::int), '');  -- a missing time: NULL
          j := j + ws[k]::int;
        END LOOP;
      END IF;
    END LOOP;
    nums := parts;
  END IF;
  y := nums[array_position(ntoks, 'Y4')]::numeric;
  IF y IS NULL AND 'Y2' = ANY (ntoks) THEN  -- two-digit years: not after the current year
    y := nums[array_position(ntoks, 'Y2')]::numeric;
    y := y + CASE WHEN y <= extract(year FROM current_date)::int % 100 THEN 2000 ELSE 1900 END;
  END IF;
  m := nums[array_position(ntoks, 'M')]::numeric;
  d := nums[array_position(ntoks, 'D')]::numeric;
  hh := coalesce(nums[array_position(ntoks, 'H')], nums[array_position(ntoks, 'H12')], '0')::numeric;
  mi := coalesce(nums[array_position(ntoks, 'MI')], '0')::numeric;
  sec := coalesce(nums[array_position(ntoks, 'S')], '0')::numeric
         + coalesce(('0.' || left(nums[array_position(ntoks, 'F')], 6))::numeric, 0);
  IF atoks <> '{}' THEN  -- and the runs of letters are the month names, AM / PM and quoted letters
    alphas := array_remove(regexp_split_to_array(s, '[^[:alpha:]]+'), '');
    mon := lower(alphas[array_position(atoks, 'MON')]);
    ap := upper(left(alphas[array_position(atoks, 'AP')], 1));
    IF mon IS NOT NULL THEN
      m := array_position('{jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec}'::text[], left(mon, 3));
      IF m IS NULL OR NOT (length(mon) = 3 OR mon = 'sept' OR mon = (
          '{january,february,march,april,may,june,july,august,september,october,november,december}'::text[])[m]) THEN
        RETURN;
      END IF;
    END IF;
  END IF;
  IF y IS NULL OR y NOT BETWEEN 1 AND 9999 OR m > 99 OR d > 99 THEN RETURN; END IF;
  IF ap IS NOT NULL THEN
    IF hh NOT BETWEEN 1 AND 12 THEN RETURN; END IF;
    hh := hh % 12 + CASE WHEN ap = 'P' THEN 12 ELSE 0 END;
  END IF;
  IF hh > 23 OR mi > 59 OR sec >= 60 THEN RETURN; END IF;
  m := coalesce(m, 0);  -- a template without the month or the day gives a partial date
  d := coalesce(d, 0);
  um := m = 0 OR m BETWEEN 97 AND 99;
  ud := um OR d = 0 OR d BETWEEN 97 AND 99;
  IF NOT um AND m > 12 THEN RETURN; END IF;
  IF ud AND y = 9999 THEN RETURN; END IF;  -- 99/99/9999: nothing is known
  IF NOT um THEN
    dim := CASE WHEN m = 2 THEN CASE WHEN y % 4 = 0 AND (y % 100 <> 0 OR y % 400 = 0) THEN 29 ELSE 28 END
                WHEN m IN (4, 6, 9, 11) THEN 30 ELSE 31 END;
    IF NOT ud AND d > dim THEN RETURN; END IF;
  END IF;
  st := 'ok';
  IF NOT ud THEN
    val := make_timestamp(y::int, m::int, d::int, hh::int, mi::int, sec::float8);
    RETURN;
  END IF;
  lo := CASE WHEN um THEN make_date(y::int, 1, 1) ELSE make_date(y::int, m::int, 1) END;
  CASE
    WHEN pol = 'first' THEN v := lo; st := 'imputed';
    WHEN pol = 'last' THEN v := CASE WHEN um THEN make_date(y::int, 12, 31) ELSE make_date(y::int, m::int, dim) END;
                           st := 'imputed';
    WHEN pol = 'mid' THEN v := CASE WHEN um THEN make_date(y::int, 6, 15) ELSE make_date(y::int, m::int, 15) END;
                          st := 'imputed';
    WHEN pol = 'birth' AND NOT um THEN v := lo; st := 'imputed';
    WHEN pol = 'month' AND NOT um THEN v := lo;
    WHEN pol = 'year' THEN v := lo;
    ELSE st := 'partial_null'; RETURN;
  END CASE;
  IF st = 'imputed' THEN  -- not after the extract date, unless the known part of the date is
    v := GREATEST(lo, LEAST(v, coalesce(nullif(current_setting('omop_agent.imputation_cap', true), '')::date,
                                        current_date)));
  END IF;
  val := v + make_interval(hours => hh::int, mins => mi::int, secs => sec::float8);
END $f$;

-- the wall clock of a parsed value: an offset is converted to etl.timezone when it is set, else ignored
CREATE OR REPLACE FUNCTION {etl}.omop_local(val timestamp, off int) RETURNS timestamp
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  SELECT CASE WHEN off IS NULL OR NOT {etl}.omop_zone_set() THEN val
    ELSE ((val - make_interval(mins => off)) AT TIME ZONE 'UTC') AT TIME ZONE {etl}.omop_zone()
  END $f$;

-- the instant of a parsed value: at its offset, else in the session time zone
CREATE OR REPLACE FUNCTION {etl}.omop_instant(val timestamp, off int) RETURNS timestamptz
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  SELECT CASE WHEN off IS NULL THEN val::timestamptz ELSE (val - make_interval(mins => off)) AT TIME ZONE 'UTC' END $f$;

-- whether a template honours an offset in the value as to_timestamp does: 'ISO' (a ::timestamptz
-- cast) and templates with TZH / OF (omop_fmt_rx 'tz'). With any other template an offset is
-- handled as for a timestamp (omop_local), and the wall clock is then read in the session zone.
CREATE OR REPLACE FUNCTION {etl}.omop_honours_offset(rx text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT rx[1] = 'ISO' OR rx[5] = 'tz' $f$;

-- the slow paths, uncounted
CREATE OR REPLACE FUNCTION {etl}.omop_walk_ts(x text, rx text[], pol text) RETURNS timestamp
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$
DECLARE r record;
BEGIN
  r := {etl}.omop_walk(x, rx, pol);
  RETURN {etl}.omop_local(r.val, r.off);
END $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_walk_tstz(x text, rx text[], pol text) RETURNS timestamptz
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$
DECLARE r record;
BEGIN
  r := {etl}.omop_walk(x, rx, pol);
  IF {etl}.omop_honours_offset(rx) THEN
    RETURN {etl}.omop_instant(r.val, r.off);
  END IF;
  RETURN {etl}.omop_local(r.val, r.off)::timestamptz;
END $f$;

-- the slow paths, counted: what a value needed (invalid, imputed, partial_null), and an offset
-- that is not the zone's own (etl.timezone, else the session's): offset_converted when etl.timezone
-- is set, offset_ignored when it is not (omop_iso_local_at); not when the offset is honoured
CREATE OR REPLACE FUNCTION {etl}.omop_walk_ts_at(site text, x text, rx text[], pol text) RETURNS timestamp
LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r record;
BEGIN
  r := {etl}.omop_walk(x, rx, pol);
  IF r.st <> 'ok' THEN
    PERFORM {etl}.omop_count(site, r.st);
  END IF;
  IF r.off IS NOT NULL AND r.val IS NOT NULL
     AND {etl}.omop_instant(r.val, r.off) <> r.val AT TIME ZONE {etl}.omop_zone() THEN
    PERFORM {etl}.omop_count(site, CASE WHEN {etl}.omop_zone_set() THEN 'offset_converted' ELSE 'offset_ignored' END);
  END IF;
  RETURN {etl}.omop_local(r.val, r.off);
END $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_walk_tstz_at(site text, x text, rx text[], pol text) RETURNS timestamptz
LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r record;
BEGIN
  IF NOT {etl}.omop_honours_offset(rx) THEN
    RETURN {etl}.omop_walk_ts_at(site, x, rx, pol)::timestamptz;
  END IF;
  r := {etl}.omop_walk(x, rx, pol);
  IF r.st <> 'ok' THEN
    PERFORM {etl}.omop_count(site, r.st);
  END IF;
  RETURN {etl}.omop_instant(r.val, r.off);
END $f$;

-- an invalid value: counted, and NULL of the type of val
CREATE OR REPLACE FUNCTION {etl}.omop_invalid(site text, val anyelement) RETURNS anyelement
LANGUAGE plpgsql VOLATILE AS $f$
BEGIN
  PERFORM {etl}.omop_count(site, 'invalid');
  RETURN val;
END $f$;

-- the readers of a text value with a template given as its fast layout (omop_fast_layout) and its
-- regular expression (omop_fmt_rx), both folded when the statement is planned: the fast path,
-- then an ISO value with fractional seconds or an offset, then the slow path. They reference the
-- value several times, so PostgreSQL inlines them only into a call whose value is cheap and not
-- volatile (a column); the *_once forms below are for any other value.
CREATE OR REPLACE FUNCTION {etl}.omop_read_ts(x text, lay text, rx text[], policy text) RETURNS timestamp
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  ELSE coalesce({etl}.omop_fast(x, lay), {etl}.omop_iso_local(x), {etl}.omop_walk_ts(x, rx, policy)) END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_tstz(x text, lay text, rx text[], policy text) RETURNS timestamptz
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  ELSE coalesce({etl}.omop_fast(x, lay)::timestamptz,
                CASE WHEN {etl}.omop_honours_offset(rx) THEN CASE WHEN {etl}.omop_iso_full(x) THEN x::timestamptz END
                     ELSE {etl}.omop_iso_local(x)::timestamptz END,
                {etl}.omop_walk_tstz(x, rx, policy)) END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_ts_at(site text, policy text, x text, lay text, rx text[]) RETURNS timestamp
LANGUAGE sql VOLATILE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  ELSE coalesce({etl}.omop_fast(x, lay), {etl}.omop_iso_local_at(site, x), {etl}.omop_walk_ts_at(site, x, rx, policy)) END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_tstz_at(site text, policy text, x text, lay text, rx text[])
RETURNS timestamptz LANGUAGE sql VOLATILE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  ELSE coalesce({etl}.omop_fast(x, lay)::timestamptz,
                CASE WHEN {etl}.omop_honours_offset(rx) THEN CASE WHEN {etl}.omop_iso_full(x) THEN x::timestamptz END
                     ELSE {etl}.omop_iso_local_at(site, x)::timestamptz END,
                {etl}.omop_walk_tstz_at(site, x, rx, policy)) END $f$;
-- the same in plpgsql, where the value is a parameter: the reader is inlined whatever the value
-- was (the layout is then read per value, which costs a few microseconds)
CREATE OR REPLACE FUNCTION {etl}.omop_read_ts_once(x text, lay text, rx text[], policy text) RETURNS timestamp
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$ BEGIN RETURN {etl}.omop_read_ts(x, lay, rx, policy); END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_tstz_once(x text, lay text, rx text[], policy text) RETURNS timestamptz
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$ BEGIN RETURN {etl}.omop_read_tstz(x, lay, rx, policy); END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_ts_once_at(site text, policy text, x text, lay text, rx text[])
RETURNS timestamp LANGUAGE plpgsql VOLATILE AS $f$
BEGIN RETURN {etl}.omop_read_ts_at(site, policy, x, lay, rx); END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_read_tstz_once_at(site text, policy text, x text, lay text, rx text[])
RETURNS timestamptz LANGUAGE plpgsql VOLATILE AS $f$
BEGIN RETURN {etl}.omop_read_tstz_at(site, policy, x, lay, rx); END $f$;

-- uncounted conversions --------------------------------------------------------------------------
-- dates and times: text is read with the template (default ISO) and the partial-date policy
-- (default reject); a date or timestamp converts as a cast would. Each references its value once,
-- so it is always inlined and a constant template is read once, when the statement is planned.
CREATE OR REPLACE FUNCTION {etl}.try_timestamp(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_ts(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy) $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamp(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamp(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamp(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamp $f$;

CREATE OR REPLACE FUNCTION {etl}.try_date(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_ts(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy)::date $f$;
CREATE OR REPLACE FUNCTION {etl}.try_date(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.try_date(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.try_date(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::date $f$;

CREATE OR REPLACE FUNCTION {etl}.try_timestamptz(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_tstz(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy) $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamptz(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamptz(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.try_timestamptz(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;

-- the same for a value that is an expression rather than a column (etl.tolerant writes them): its
-- text is read in plpgsql (omop_read_*_once), so that it is computed once per row
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamp_once(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_ts_once(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamp_once(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamp_once(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamp_once(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamp LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_date_once(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_ts_once(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy)::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_date_once(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_date_once(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_date_once(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamptz_once(x text, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$
SELECT {etl}.omop_read_tstz_once(x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt), policy) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamptz_once(x date, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamptz_once(x timestamp, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql STABLE PARALLEL SAFE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_try_timestamptz_once(x timestamptz, fmt text DEFAULT 'ISO', policy text DEFAULT 'reject')
RETURNS timestamptz LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;

-- numbers: text in plain decimal notation (an integer may end in .0); NaN and infinities are
-- invalid, and so is a value out of the type's range. A boolean is 1 or 0.
CREATE OR REPLACE FUNCTION {etl}.try_number(x text) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x IS NULL OR length(x) > 1000 THEN NULL
  WHEN translate(x, '0123456789', '') IN ('', '.') AND x NOT IN ('', '.') THEN x::numeric
  WHEN x ~ '^[[:space:]]*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]{1,3})?[[:space:]]*$'
    THEN btrim(x, E' \t\r\n')::numeric END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_number(x numeric) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x::text NOT IN ('NaN', 'Infinity', '-Infinity') THEN x END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_number(x double precision) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x::text NOT IN ('NaN', 'Infinity', '-Infinity') THEN x::numeric END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_number(x integer) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::numeric $f$;
CREATE OR REPLACE FUNCTION {etl}.try_number(x bigint) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::numeric $f$;
CREATE OR REPLACE FUNCTION {etl}.try_number(x boolean) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::integer::numeric $f$;

CREATE OR REPLACE FUNCTION {etl}.try_integer(x text) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  WHEN translate(x, '0123456789', '') = '' AND length(x) BETWEEN 1 AND 9 THEN x::integer
  WHEN x ~ '^[[:space:]]*[+-]?[0-9]{1,30}(\.0*)?[[:space:]]*$' THEN
    CASE WHEN btrim(x, E' \t\r\n')::numeric BETWEEN -2147483648 AND 2147483647
         THEN btrim(x, E' \t\r\n')::numeric::integer END END $f$;
-- round(): numeric rounds to the nearest integer as the cast does; NaN and infinities are out of range
CREATE OR REPLACE FUNCTION {etl}.try_integer(x numeric) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN round(x) BETWEEN -2147483648 AND 2147483647 THEN x::integer END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_integer(x double precision) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT {etl}.try_integer(x::numeric) $f$;
CREATE OR REPLACE FUNCTION {etl}.try_integer(x bigint) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x BETWEEN -2147483648 AND 2147483647 THEN x::integer END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_integer(x integer) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.try_integer(x boolean) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::integer $f$;

CREATE OR REPLACE FUNCTION {etl}.try_bigint(x text) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN x IS NULL THEN NULL
  WHEN translate(x, '0123456789', '') = '' AND length(x) BETWEEN 1 AND 18 THEN x::bigint
  WHEN x ~ '^[[:space:]]*[+-]?[0-9]{1,30}(\.0*)?[[:space:]]*$' THEN
    CASE WHEN btrim(x, E' \t\r\n')::numeric BETWEEN -9223372036854775808 AND 9223372036854775807
         THEN btrim(x, E' \t\r\n')::numeric::bigint END END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_bigint(x numeric) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
SELECT CASE WHEN round(x) BETWEEN -9223372036854775808 AND 9223372036854775807 THEN x::bigint END $f$;
CREATE OR REPLACE FUNCTION {etl}.try_bigint(x double precision) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT {etl}.try_bigint(x::numeric) $f$;
CREATE OR REPLACE FUNCTION {etl}.try_bigint(x bigint) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.try_bigint(x integer) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::bigint $f$;
CREATE OR REPLACE FUNCTION {etl}.try_bigint(x boolean) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$ SELECT x::integer::bigint $f$;

-- counted front ends: the compiler writes them for the casts of column and expression fields (and
-- the mapping's WHERE, site 'where'). Text is read as by try_*; other types convert, and only a
-- value that cannot be converted calls a counter. Spec SQL may not call them (sqlguard). The
-- date and time front ends and the identity overloads are SQL that references the value once, so
-- they are always inlined into the statement and a constant template is read when it is planned;
-- for a column the reader (omop_read_*_at) is inlined too, for an expression the compiler writes
-- the *_once_at forms. The checked numeric overloads are plpgsql: their argument is usually an
-- expression with a counted call in it, which PostgreSQL does not inline into a body that
-- references it twice, and a plpgsql call costs less than a SQL function that is not inlined.
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS date LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_ts_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt))::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_at(site text, policy text, x date)
RETURNS date LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_at(site text, policy text, x timestamp)
RETURNS date LANGUAGE sql IMMUTABLE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_at(site text, policy text, x timestamptz)
RETURNS date LANGUAGE sql STABLE AS $f$ SELECT x::date $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS timestamp LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_ts_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt)) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_at(site text, policy text, x date)
RETURNS timestamp LANGUAGE sql IMMUTABLE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_at(site text, policy text, x timestamp)
RETURNS timestamp LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_at(site text, policy text, x timestamptz)
RETURNS timestamp LANGUAGE sql STABLE AS $f$ SELECT x::timestamp $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS timestamptz LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_tstz_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt)) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_at(site text, policy text, x date)
RETURNS timestamptz LANGUAGE sql STABLE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_at(site text, policy text, x timestamp)
RETURNS timestamptz LANGUAGE sql STABLE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_at(site text, policy text, x timestamptz)
RETURNS timestamptz LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;

-- the same for a value that is an expression rather than a column (see omop_read_*_once)
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_once_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS date LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_ts_once_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt))::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_once_at(site text, policy text, x date)
RETURNS date LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_once_at(site text, policy text, x timestamp)
RETURNS date LANGUAGE sql IMMUTABLE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_date_once_at(site text, policy text, x timestamptz)
RETURNS date LANGUAGE sql STABLE AS $f$ SELECT x::date $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_once_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS timestamp LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_ts_once_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt)) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_once_at(site text, policy text, x date)
RETURNS timestamp LANGUAGE sql IMMUTABLE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_once_at(site text, policy text, x timestamp)
RETURNS timestamp LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamp_once_at(site text, policy text, x timestamptz)
RETURNS timestamp LANGUAGE sql STABLE AS $f$ SELECT x::timestamp $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_once_at(site text, policy text, x text, fmt text DEFAULT 'ISO')
RETURNS timestamptz LANGUAGE sql VOLATILE AS $f$
SELECT {etl}.omop_read_tstz_once_at(site, policy, x, {etl}.omop_fast_layout(fmt), {etl}.omop_fmt_rx(fmt)) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_once_at(site text, policy text, x date)
RETURNS timestamptz LANGUAGE sql STABLE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_once_at(site text, policy text, x timestamp)
RETURNS timestamptz LANGUAGE sql STABLE AS $f$ SELECT x::timestamptz $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_timestamptz_once_at(site text, policy text, x timestamptz)
RETURNS timestamptz LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x text)
RETURNS numeric LANGUAGE sql VOLATILE AS $f$
SELECT coalesce({etl}.try_number(x), CASE WHEN btrim(x) <> '' THEN {etl}.omop_invalid(site, NULL::numeric) END) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x numeric)
RETURNS numeric LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r numeric;
BEGIN
  r := {etl}.try_number(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x double precision)
RETURNS numeric LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r numeric;
BEGIN
  r := {etl}.try_number(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x integer)
RETURNS numeric LANGUAGE sql IMMUTABLE AS $f$ SELECT x::numeric $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x bigint)
RETURNS numeric LANGUAGE sql IMMUTABLE AS $f$ SELECT x::numeric $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_number_at(site text, policy text, x boolean)
RETURNS numeric LANGUAGE sql IMMUTABLE AS $f$ SELECT x::integer::numeric $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x text)
RETURNS integer LANGUAGE sql VOLATILE AS $f$
SELECT coalesce({etl}.try_integer(x), CASE WHEN btrim(x) <> '' THEN {etl}.omop_invalid(site, NULL::integer) END) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x numeric)
RETURNS integer LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r integer;
BEGIN
  r := {etl}.try_integer(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x double precision)
RETURNS integer LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r integer;
BEGIN
  r := {etl}.try_integer(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x bigint)
RETURNS integer LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r integer;
BEGIN
  r := {etl}.try_integer(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x integer)
RETURNS integer LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_integer_at(site text, policy text, x boolean)
RETURNS integer LANGUAGE sql IMMUTABLE AS $f$ SELECT x::integer $f$;

CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x text)
RETURNS bigint LANGUAGE sql VOLATILE AS $f$
SELECT coalesce({etl}.try_bigint(x), CASE WHEN btrim(x) <> '' THEN {etl}.omop_invalid(site, NULL::bigint) END) $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x numeric)
RETURNS bigint LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r bigint;
BEGIN
  r := {etl}.try_bigint(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x double precision)
RETURNS bigint LANGUAGE plpgsql VOLATILE AS $f$
DECLARE r bigint;
BEGIN
  r := {etl}.try_bigint(x);
  IF r IS NULL AND x IS NOT NULL THEN
    PERFORM {etl}.omop_count(site, 'invalid');
  END IF;
  RETURN r;
END $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x bigint)
RETURNS bigint LANGUAGE sql IMMUTABLE AS $f$ SELECT x $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x integer)
RETURNS bigint LANGUAGE sql IMMUTABLE AS $f$ SELECT x::bigint $f$;
CREATE OR REPLACE FUNCTION {etl}.omop_as_bigint_at(site text, policy text, x boolean)
RETURNS bigint LANGUAGE sql IMMUTABLE AS $f$ SELECT x::integer::bigint $f$;

-- installed last: a schema that answers with this version has every function above
CREATE OR REPLACE FUNCTION {etl}.omop_agent_functions_version() RETURNS text
LANGUAGE sql IMMUTABLE AS $f$ SELECT '{functions_version}' $f$;
