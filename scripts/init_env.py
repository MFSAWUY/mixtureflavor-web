"""Create a private Compose configuration. Never replace existing credentials."""
import argparse
from pathlib import Path
import re
import secrets

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', help='An owned domain with its DNS pointed to this server. Omit for local Docker.')
    args = parser.parse_args()
    if args.domain and not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', args.domain):
        parser.error('Use a hostname without protocol, port or path.')
    site = args.domain or ':80'
    bind, http, https = ('0.0.0.0', 80, 443) if args.domain else ('127.0.0.1', 8080, 8443)
    path = ROOT / '.env'
    with path.open('x', encoding='utf-8') as file:
        file.write(f'SITE_ADDRESS={site}\nBIND_ADDRESS={bind}\nHTTP_PORT={http}\nHTTPS_PORT={https}\nFLAVOR_ADMIN_USER=admin\nFLAVOR_ADMIN_PASSWORD={secrets.token_urlsafe(32)}\nFLAVOR_DATABASE=/data/submissions.sqlite3\n')
    path.chmod(0o600)
    print(f'Configuration written to {path}. Credentials are in that private file, not in this output.')


if __name__ == '__main__':
    main()
