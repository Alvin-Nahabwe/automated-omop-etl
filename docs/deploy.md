# Installing omop-agent on a server (Docker)

This sets up omop-agent on the server that holds the source data. Patient data stays on that
server. When a project uses Claude (`llm.provider: anthropic`), only aggregates leave it (see
"Privacy and safety" in the README).

You get two containers, started together by `deploy/compose.yaml`:

| Container | What it is |
|---|---|
| `db` | PostgreSQL 16 for the staged source, the vocabulary and the OMOP CDM. It is not reachable from outside the server. The app uses its own `omop` role, which is not a superuser. |
| `app` | The omop-agent command line and the review web app on port 8000. It runs as an ordinary user. |

It also uses four folders next to `compose.yaml`:

| Folder | Holds |
|---|---|
| `projects/` | One sub-folder per project: `omop_agent.yaml` and its `work/` (specs, mappings, logs, reports). |
| `data/` | The web app's accounts, its session key and the sign-in log. |
| `athena/` | The unpacked Athena vocabulary download. |
| `source/` | Source CSV exports. The app can read them but not change them. |

## Two images

* `Dockerfile` (the default): the command line and the web app. Quality checks use the
  built-in engine.
* `Dockerfile.r`: the same plus R, Java, Achilles and the official OHDSI DataQualityDashboard,
  about 1 GB larger. Projects created in it run Achilles and the full DQD on every final CDM
  (`quality.use_r_dqd`), and a run fails rather than silently skipping them if R cannot run
  (`quality.require_r_dqd`). Achilles' results in the results schema are what ATLAS needs for
  its Data Sources pages. To use it, set in `.env`:

  ```bash
  OMOP_AGENT_DOCKERFILE=Dockerfile.r
  OMOP_AGENT_IMAGE_TAG=local-r
  ```

## Requirements

* Linux with Docker Engine and Docker Compose v2.
* About 4 CPU cores and 16 GB RAM. Disk: about 10 GB for a full vocabulary, plus your data.
* Outbound HTTPS to `api.anthropic.com` only for projects that use Claude. Outbound HTTPS to
  `athena.ohdsi.org` only if you download the vocabulary from the server; otherwise copy the
  unpacked bundle into `athena/`.

## Install

```bash
git clone https://github.com/Alvin-Nahabwe/automated-omop-etl.git
cd automated-omop-etl/deploy
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

* `POSTGRES_PASSWORD` and `OMOP_DB_PASSWORD`: two different passwords made of letters and
  digits, for example from `openssl rand -hex 24`.
* `ANTHROPIC_API_KEY`: your Claude API key, if projects will use Claude.
* `ATHENA_DOWNLOAD_URL`: the link Athena e-mails you, if you download the vocabulary here.
  Treat it like a password.
* `OMOP_AGENT_WEB_SECRET`: optional. A key of at least 32 characters for signing sessions,
  for example from `openssl rand -hex 32`. If it is empty, one is generated in `data/secret_key`.
* `OMOP_AGENT_UID` and `OMOP_AGENT_GID`: the output of `id -u` and `id -g`. The app then writes
  files as you, and you can edit them.

Then start it:

```bash
mkdir -p projects data athena source
docker compose up -d --build
docker compose ps                                   # both should be "healthy"
docker compose run --rm app users add YOUR-NAME --role operator --full-name "Your Name"
```

The app now listens on `http://127.0.0.1:8000` on the server. For other people to use it, put
it behind HTTPS (next section).

To try it first, create the demo project (synthetic data, recorded answers, no API key). Then
sign in and press **Continue the pipeline** on its page:

```bash
docker compose run --rm app demo /srv/projects/demo --no-run
```

## HTTPS

Keep the app on `127.0.0.1` (`OMOP_AGENT_BIND`) and let your usual web server terminate HTTPS
and forward to it. Example for nginx on the same server:

```nginx
server {
    listen 443 ssl;
    server_name omop.example.org;
    ssl_certificate     /etc/ssl/certs/omop.crt;
    ssl_certificate_key /etc/ssl/private/omop.key;
    client_max_body_size 2m;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 120s;
    }
}
```

Then set `OMOP_AGENT_SECURE_COOKIES=true` in `.env`, so the session cookie is only sent over
HTTPS, and run `docker compose up -d`.

The app slows down password guessing per user and per client address. Inside the container,
every connection appears to come from Docker's gateway, so the app uses the address your proxy
reports in `X-Forwarded-For`. `FORWARDED_ALLOW_IPS=*` (the default) trusts that header. This is
safe while the port is bound to 127.0.0.1, because only programs on the server can reach it.
If you bind the app to a network address instead, set `FORWARDED_ALLOW_IPS` to the gateway
address your proxy's requests come from.

The app has no single sign-on or two-factor login yet. Make it reachable only from your
internal network or VPN.

## A first real project

1. Put the source exports in `source/<name>/`, one CSV per table.
2. Get the vocabulary, in one of two ways:
   * Copy the unpacked Athena bundle into `athena/`.
   * Download it from the server:
     `docker compose run --rm app vocab-download --dest /srv/athena -c /srv/projects/<name>/omop_agent.yaml`.
     Run this after step 3, because it needs the project's config file.
3. Create the project:
   ```bash
   docker compose run --rm app init /srv/projects/<name> --name "<name>" \
       --source /srv/source/<name> --athena-dir /srv/athena
   ```
   This writes `projects/<name>/omop_agent.yaml`. Its database URL is
   `${OMOP_AGENT_DB_URL}`, so no password is stored in it. Its schemas are named after the
   project (`<name>_cdm`, `<name>_src`, ...), so projects in the same database don't overwrite
   each other; the app refuses to run a project whose schemas another project uses. Review the
   privacy settings in that file, and set two values the build cannot know:
   * `cdm_source.source_release_date`: the date of the data extract. Residency episodes that are
     still open (no exit date) end on it. Unset, the build derives a date from the exit dates and
     writes it to `cdm_source`; the `etlSourceReleaseDate` check fails while open episodes end on
     it (and is fatal when that date is after the day of the build).
   * `etl.timezone`: the site's time zone, e.g. `Africa/Nairobi`, when times in the source carry an
     offset or `Z` (ODK submission times do). The build converts them to it. Unset, each time is
     kept as written, and a design that reads times with `Z` or with several different offsets
     fails validation.
4. In the browser, open the project and press **Continue the pipeline**. The first run
   profiles the source and then stops for the **privacy review**, before anything is sent to
   Claude. A reviewer opens the **Privacy** tab, checks the withheld columns and the profile
   exactly as Claude will see it (look for names, phone numbers, local ID numbers, GPS or free
   text among the listed values), fixes the settings if needed and approves it. A column added
   to `privacy.pii_columns` is withheld at once. One released with `privacy.not_pii_columns`
   only shows values after the source is profiled again (**Start over**).
5. **Continue the pipeline** again: Claude interprets the profile, the vocabulary is loaded (a
   few minutes for a full bundle) and the ETL design is drafted. The run then stops until a
   reviewer approves the design.
6. A reviewer reads **ETL design** and approves it, or asks for changes. Changes are made by
   editing `projects/<name>/work/etl_spec.yaml` on the server and approving again. The page
   shows what changed.
7. **Continue the pipeline** again: the codes are mapped and the run stops at the mappings.
   Reviewers work through **Code mappings** (the "Needs review" filter first) and approve them.
8. **Continue the pipeline**: the CDM is built, checked and repaired, and the report is written.

A run pauses at a review step whenever what was approved has changed. That happens after a
hand edit, after the repair agent changes the design, or when new source codes appear. The
spec page shows those changes as a diff.

## Data protection

Before the first real project, agree the following with APHRC's data protection officer or
ethics committee. Sending aggregates to the Claude API is a transfer of data derived from
health records to a service outside Kenya, which the Data Protection Act expects to be covered.

* **What leaves the server**, and only for projects with `llm.provider: anthropic`: column
  names and statistics, values seen at least `privacy.min_cell_count` times (5 by default),
  the source codes being mapped with their counts, aggregate counts from the repair agent, and
  database error messages with their values removed. Never patient rows, rare values or the
  values of withheld columns. Every prompt and answer is kept in the project's
  `work/llm_calls/`. `privacy.send_values_to_llm: false` sends statistics only;
  `llm.provider: none` sends nothing (you then write the ETL design yourself).
* **The privacy review** is the control point: a named reviewer approves what will be sent
  before the first call, and again whenever the source or the settings change.
* **What stays on the server**: the staged copy of the source (including identifier columns)
  in the `<project>_src` schema, the CDM, the work folders and the backups. Keep the server,
  the backups and `.env` under the same controls as the source data.
* **Ending a project**: drop its schemas (`DROP SCHEMA <project>_src, <project>_cdm, ... CASCADE`
  as the `postgres` user), delete `projects/<project>/` (which holds `llm_calls/`) and remove it
  from the backups, as your retention policy requires.

## Roles

| Role | Can |
|---|---|
| viewer | read everything: designs, mappings, run logs, reports, the audit trail |
| reviewer | also edit mappings and approve the design and the mappings |
| operator | also start and stop pipeline runs |

Manage accounts on the server:

```bash
docker compose run --rm app users list
docker compose run --rm app users add NAME --role reviewer
docker compose run --rm app users passwd NAME
docker compose run --rm app users set-role NAME operator
docker compose run --rm app users disable NAME
```

Changing an account signs it out everywhere, and so does signing out. Approvals and edits are recorded with the user name
in each project's `work/audit.jsonl` (the **Audit trail** tab). Sign-ins are logged in
`data/access.jsonl`.

## Backups

Back up three things: the database, the `projects/` and `data/` folders, and `.env`.

```bash
umask 077   # the backups contain patient data and secrets
docker compose exec -T db pg_dump -U postgres -Fc omop > omop-$(date +%F).dump
tar czf omop-files-$(date +%F).tgz projects data .env
```

`data/` holds the session key and the account password hashes. `.env` holds the database
passwords and API keys. Store both like other secrets.

## Updating

```bash
git pull
docker compose up -d --build
```

Accounts, projects, approvals and the database are kept.

An update can change the CDM that an unchanged design builds. The update that added one row per
key, the observation-period merge, routing by domain and the eras does: the first build of each
project after it says so in its log and in the **Audit trail** (an `upgrade_notice` event), and
the README's "Upgrading" section lists the changes. Approved designs and mappings stay
approved. Things to check after an update:

* The first build installs a new version of the helper functions in the project's ETL schema. The
  app's `omop` role owns that schema, so this needs nothing from you. If a project sets
  `target.etl_role`, the build grants the ETL tables it creates to that role; the role needs the
  privileges listed under "Privacy and safety" in the README.
* CSV sources: staging now stores `NA`, `NULL`, `\N` and `.` as empty values. This takes effect
  when the source is staged again:
  `docker compose run --rm app profile -c /srv/projects/<name>/omop_agent.yaml`, then **Continue
  the pipeline**. A project that sends data to Claude then needs a new privacy review. Don't use
  **Start over** for this: it also redrafts the ETL design and maps every code again. Set
  `source.na_values: []` in the project's config to keep these values as data.
* Set `cdm_source.source_release_date` and, where needed, `etl.timezone` (see "A first real
  project", step 3), and read the new failed checks in the report: the build now reports what it
  could not read, what it merged and what it moved.

## Troubleshooting

* **`429 Too Many Requests` from Docker Hub.** Set
  `PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim` and
  `POSTGRES_IMAGE=mirror.gcr.io/library/postgres:16` in `.env`.
* **Your network inspects TLS** and `pip` fails during the build. Build the image once, giving
  it your proxy's CA certificate:
  `docker build --secret id=extra_ca,src=/path/to/ca.crt --build-arg HTTPS_PROXY=$HTTPS_PROXY -t omop-agent:local ..`.
  Then run `docker compose up -d` without `--build`.
* **"Permission denied" under `projects/` or `data/`.** Set `OMOP_AGENT_UID` and
  `OMOP_AGENT_GID` in `.env` to the folders' owner, or `chown` the folders to 10001:10001.
* **"The vocabulary database cannot be reached"** on the mappings page. Check
  `docker compose ps` and `docker compose logs db`.
* **`init` scripts did not run** (no `omop` role). They only run when the database volume is
  first created. To start the database from scratch, run `docker compose down -v`; this
  deletes the database.
