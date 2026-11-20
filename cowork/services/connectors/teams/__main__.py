"""python -m cowork.services.connectors.teams check: asks Microsoft for the bot's token."""

import sys

from cowork.services.connectors.teams.bot import check

if sys.argv[1:] != ["check"]:
    print("usage: python -m cowork.services.connectors.teams check")
    sys.exit(2)
sys.exit(check())
