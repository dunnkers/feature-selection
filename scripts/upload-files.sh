#!/bin/sh
#
# Copy over directory files after every commit to an SSH server via rsync.
#
# To enable a git hook for this script, create a symlink to .git/hooks/

echo "upload-files started"
rsync -aP --exclude-from='.gitignore' --exclude='.git' --exclude='.vscode' --exclude='.gitignore' . $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/
echo "upload-files finished"