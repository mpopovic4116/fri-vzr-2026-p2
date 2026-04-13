# Running

The makefile should work by default without any parameters, although you can customize them by passing `k=v` values (see the top of the `Makefile` for details).
The final executable is emitted to a location that depends on the compilation flags used.
There's a wrapper `scripts/make.sh` which runs make and prints this path to stdout (and other messages to stderr).

The same `k=v` pairs should be passed to the executable itself.
The script `scripts/run.sh` will run `scripts/make.sh` and then run the final executable with the same args.
Prefer this for interactive use.

To generate `compile_commands.json`, run `bear -- make clean all` (possibly passing additional flags to select alternate implementations and whatnot).

# Implementations

There are three separate implementations:

- `seq` (default), which is a single threaded CPU implementation (`src_c/impl_cpu.c`)
- `omp`, which is `seq` with `#pragma omp`s
- `cuda`, which uses a single GPU (`src_c/impl_cuda.cu`)

You can select the implementation by passing `impl=<choice>`, like `impl=cuda`.
There's an additional compile flag `cuda_arch=<native|v100s>`.
It's set to native by default so it works outside the cluster, it works on the cluster as well, but set it to `v100s` for the final benchmark to use the same settings as the prof's code.

# Precision

The `precision=<64|32>` setting controls the kind of float used throughout the program.
There's also `precision=16`, but it's broken at the moment.

# Gif generation

Compiling with `gif=<non_empty>` will add extra code to generate a gif. Pass `gif=<output_filename>` to the executable to emit a gif.
The makefile just checks whether or not `gif` is empty, so passing the same flags to the compiler that you pass to the executable works just fine.
For example, `scripts/run.sh gif=out.gif`.

# Running on the cluster

There are several scripts that are meant to be chained together to run something on the cluster:

- `scripts/up.sh`: Rsyncs the codebase to the cluster
- `scripts/on_cluster.sh`: Runs a single command on the login node (automatically runs `scripts/up.sh` first)
- `scripts/with_modules.sh`: Loads the numactl (newer gcc) and CUDA modules
- `scripts/srun.sh`: Runs `srun` with some preset parameters

Make sure to configure `.envrc.local` first (see `.envrc.local.example`), and set up ssh multiplexing to avoid being nagged by 2FA.
Once you have things set up, you can run something on the cluster like this (the order of the script chain is important):

```bash
scripts/on_cluster.sh scripts/srun.sh --gpu scripts/with_modules.sh scripts/run.sh impl=cuda cuda_arch=v100s
```
