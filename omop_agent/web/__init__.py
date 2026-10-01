"""The review web app: browser screens for the pipeline's review gates, runs and reports.

It is meant to be installed on the same server as the data (see ``deploy/``) and reached over
the institution's network. ``create_app`` lives in ``omop_agent.web.app``, which needs the
``web`` extras (``pip install 'omop-agent[web]'``); ``users`` and ``worker`` do not.
"""
