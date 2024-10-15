# How to run the script
Note) the following commands are for running the script on the local machine, with `soc-twitter-follows-mun` dataset.

## Check help message for options
```bash
cargo run --release -- --help
```

## Graph initialization
```bash
cargo run --release -- --graph soc-twitter-follows-mun
```

## Preload posts
- Given number is the number of posts to preload per user (default is 10, which means every user will get 1~20 posts)
```bash
cargo run --release -- --graph soc-twitter-follows-mun --compose-only --num-requests=10 --print-every=1000
```

## Run read-only requests
- For reliability, we set the number of requests to 32.
- By setting number of requests to -1, the script will run indefinitely.
```bash
cargo run --release -- --graph soc-twitter-follows-mun --timeline-only --num-requests=-1 --limit=32
```
