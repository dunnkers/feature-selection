#!/bin/sh
#
# A git hook to copy over directory files after every commit to an SSH server via rsync.
#
# To enable this hook, create a symlink to .git/hooks/

echo "upload-files started"
rsync -aP --exclude-from='.gitignore' --exclude='.git' --exclude='.vscode' --exclude='.gitignore' . $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/
echo "upload-files finished"