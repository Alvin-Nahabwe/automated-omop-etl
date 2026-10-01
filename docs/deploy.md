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

Then set these in `.env` and run `docker compose up -d`:

* `OMOP_AGENT_SECURE_COOKIES=true`, so the session cookie is only sent over HTTPS.
* `FORWARDED_ALLOW_IPS=*`. This makes the app trust the client address the proxy reports, which
  it needs to lock out repeated failed sign-ins. It is safe while the port is bound to
  127.0.0.1, because only programs on the server can reach it.

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
   `${OMOP_AGENT_DB_URL}`, so no password is stored in it. Review the privacy settings and
   `cdm_source` in that file.
4. In the browser, open the project and press **Continue the pipeline**. The first run
   profiles the source, loads the vocabulary (a few minutes for a full bundle) and drafts the
   ETL design. It then stops until a reviewer approves the design.
5. A reviewer reads **ETL design** and approves it, or asks for changes. Changes are made by
   editing `projects/<name>/work/etl_spec.yaml` on the server and approving again. The page
   shows what changed.
6. **Continue the pipeline** again: the codes are mapped and the run stops at the mappings.
   Reviewers work through **Code mappings** (the "Needs review" filter first) and approve them.
7. **Continue the pipeline**: the CDM is built, checked and repaired, and the report is written.

A run pauses at a review step whenever what was approved has changed. That happens after a
hand edit, after the repair agent changes the design, or when new source codes appear. The
spec page shows those changes as a diff.

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

Changing an account signs it out everywhere. Approvals and edits are recorded with the user name
in each project's `work/audit.jsonl` (the **Audit trail** tab). Sign-ins are logged in
`data/access.jsonl`.

## Backups

Back up three things: the database, the `projects/` and `data/` folders, and `.env`.

```bash
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
