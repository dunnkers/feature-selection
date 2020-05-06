# feature-selection
By Jeroen Overschie.

### Architectual testing setup
To run on the Peregrine HPC environment (University of Groningen), the following setup can be used.

1. Create an environment variable `PEREGRINE_USERNAME` with your peregrine username login credentials.

```shell
export PEREGRINE_USERNAME=<your username>
```

Preferably, add the export to `~/.bash_profile` to let the environment variable stick upon instantiating new terminal instances and restarts. (be aware that when using zsh the env variable might not show up in prompt- do put it in `.bash_profile` though; the git hooks code will use the default shell.)

2. Upload your public ssh key to Peregrine to avoid password prompts. e.g. see [this link](https://superuser.com/questions/8077/how-do-i-set-up-ssh-so-i-dont-have-to-type-my-password) for a guide.

3. Symlink the provided git hook, to automatically transfer this directory upon any git commits. Run:
```shell
ln -s ./git-hooks/* .git/hooks
```

4. Done ✨

Now, your files will be automatically copied over to Peregrine on every git commit. 💎