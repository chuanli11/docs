## Unable to install CLI due to certificate errors


### Solution

An alternative method for downloading the CLI. Use a Linux shell to run:

``` bash
kubectl port-forward -n runai svc/researcher-service 4180:4180
```

And in another shell run
```
wget --content-disposition http://localhost:4180/cli/linux
```

## When running the CLI you get an error an invalid configuration error

When running any CLI command you get:

    FATA[0000] invalid configuration: no configuration has been provided

### Solution

Your machine is not connected to the Kubernetes cluster. Make sure that you have a `~/.kube` directory that contains a configuration file pointing to the Kubernetes cluster.

## When running the CLI you get an error: open .../.kube/config.lock: permission denied

When running any CLI command you get a permission denied error.

### Solution

The user running the CLI does not have read permissions to the `.kube` directory.

## When running 'runai logs', the logs are delayed

By default, Python buffers stdout and stderr, which is not flushed in real-time. This may cause logs to appear sometimes minutes after being buffered.

### Solution

Set the env var PYTHONUNBUFFERED to any non-empty string or pass -u to Python. e.g. `python -u main.py`.

## Runai list jobs command works but runai submit does not

### Solution 

(Version 2.4 or earlier) Helm utility is not installed. See Run:ai CLI Installation documentation. 
