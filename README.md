# feature-selection
By Jeroen Overschie.

## Setup
To run on the Peregrine HPC environment (University of Groningen), the following setup can be used.

1. Create an environment variable `PEREGRINE_USERNAME` with your peregrine username login credentials.

```shell
export PEREGRINE_USERNAME=<your username>
```

Preferably, add the export to `~/.bash_profile` to let the environment variable stick upon instantiating new terminal instances and restarts. (be aware that when using zsh the env variable might not show up in prompt- do put it in `.bash_profile` though; the git hooks code will use the default shell.)

You can confirm git catches your environment variable by adding:

```shell
[alias]
    test = !bash -c '"echo \"Value: $PEREGRINE_USERNAME\";"'
```

to your `~/.gitconfig` file. Do restart your terminal after setting the env variable, or in case of Visual Studio code, restart the editor if using the integrated git environment (executes by default from `/usr/local/bin/git`).

2. Upload your public ssh key to Peregrine to avoid password prompts. e.g. see [this link](https://superuser.com/questions/8077/how-do-i-set-up-ssh-so-i-dont-have-to-type-my-password) for a guide.

(optional) 3. Symlink the provided git hook, to automatically transfer this directory upon any git commits. Run:
```shell
cd .git/hooks
ln -s -f ../../scripts/upload-files ./post-commit
```
We first cd into hooks directory, because [symlinks are relative](https://stackoverflow.com/questions/4592838/symbolic-link-to-a-hook-in-git).

Now, your files will be automatically copied over to Peregrine on every git commit. 💎

4. Done ✨

## Running tests
1. Transfer files
```shell
sh scripts/post-commit
```

2. Run job
```shell
sh scripts/submit-job
```

## About
MSc Research Internship 2020.