#!/usr/bin/env bash
# authorize-forward-key.sh <ssh-host-alias> <allowed-port> [<allowed-port> ...]
#
# Install the dedicated tailnet-forward key on a model host, restricted to
# port-forwarding of the given remote ports. Idempotent: any earlier entry with
# this key's comment is removed first, and the new entry is appended after a
# guaranteed newline, so it can never merge with the previous line.
#
# The key is read from disk by this script; it is never typed on a command line
# and never printed. Run it yourself: the agent's policy forbids it from
# touching ~/.ssh, which is also why the key material must not be pasted around.
set -euo pipefail

host=${1:?usage: authorize-forward-key.sh <host> <port> [<port> ...]}
shift
[ "$#" -gt 0 ] || { echo "give at least one port to permit" >&2; exit 2; }

key_dir=$HOME/.local/state/smil-forward
key=$key_dir/id_ed25519
pub=$key.pub
comment='forsker-pc tailnet model forwards'

[ -s "$pub" ] || { echo "missing $pub (create it with ssh-keygen -t ed25519 -N '' -f $key)" >&2; exit 1; }

opts='restrict,port-forwarding'
for p in "$@"; do
    opts="$opts,permitopen=\"127.0.0.1:$p\""
done
line="$opts $(cat "$pub")"
payload=$(printf '\n%s\n' "$line" | base64 -w0)

# base64 the payload so no quoting or whitespace can damage it in transit.
ssh -o BatchMode=yes "$host" "umask 077; f=\$HOME/.ssh/authorized_keys; touch \"\$f\"; sed -i '/${comment//\//\\/}/d' \"\$f\"; printf '%s' '$payload' | base64 -d >> \"\$f\"; printf 'installed: '; grep -c '${comment//\//\\/}' \"\$f\""

echo "entry on $host: $(ssh -o BatchMode=yes "$host" "grep '${comment//\//\\/}' \$HOME/.ssh/authorized_keys | cut -c1-$((${#opts} + 12))")..."

echo "auth test (key alone, no agent, no multiplexing):"
ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ControlMaster=no -o ControlPath=none -o PreferredAuthentications=publickey \
    "$host" 'echo "  $HOSTNAME auth-ok"'
