 #!/bin/bash
set -e
peru-hatch "$@"
mkdir -p build/public
rsync -a --exclude '.*' --exclude 'build' . build/public/
