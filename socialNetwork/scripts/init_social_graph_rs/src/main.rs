use clap::Parser;
use futures::future::join_all;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Zipf};
use reqwest::Client;
// use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};
use std::time::Instant;

const DEFAULT_DELAY_MS: u64 = 100;

/// DeathStarBench social graph initializer in Rust.
#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    /// Graph name. (`socfb-Reed98`, `ego-twitter`, or `soc-twitter-follows-mun`)
    #[clap(long, default_value = "socfb-Reed98")]
    graph: String,
    /// IP address of socialNetwork NGINX web server.
    #[clap(long, default_value = "127.0.0.1")]
    ip: String,
    /// IP port of socialNetwork NGINX web server.
    #[clap(long, default_value = "8080")]
    port: u16,
    /// Initialize with up to 20 posts per user.
    #[clap(long, action)]
    compose: bool,
    /// Run in compose-only mode, assuming the graph is already initialized.
    #[clap(long, action)]
    compose_only: bool,
    /// Run in timeline-only mode, performing user timeline reads.
    #[clap(long, action)]
    timeline_only: bool,
    /// Theta parameter for Zipfian distribution (default: 0.99).
    #[clap(long, default_value = "0.99")]
    theta: f64,
    /// Total number of simultaneous connections.
    #[clap(long, default_value = "200")]
    limit: usize,
    /// Total number of requests to perform (only applicable in compose-only and timeline-only modes).
    #[clap(short, long, default_value = "200")]
    num_requests: isize, // Changed to isize to allow -1
    /// Print progress every N requests (default: 10000).
    #[clap(long, default_value = "10000")]
    print_every: usize,
    /// Number of posts to compose per user (default: 20).
    #[clap(long, default_value = "20")]
    num_compose: usize,
}

#[tokio::main]
async fn main() {
    // Parse command line arguments
    let args = Args::parse();

    if args.timeline_only && args.compose_only {
        eprintln!("Cannot specify both --timeline-only and --compose-only");
        std::process::exit(1);
    }

    let addr = format!("http://{}:{}", args.ip, args.port);
    let limit = args.limit;

    // Create an HTTP client
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();

    // Read nodes (needed for all modes)
    let nodes = get_num_nodes(&format!(
        "../../datasets/social-graph/{}/{}.nodes",
        args.graph, args.graph
    ));

    if args.timeline_only {
        // Timeline-only mode: perform user timeline reads
        timeline(
            &client,
            &addr,
            nodes,
            limit,
            args.theta,
            args.num_requests,
            args.print_every,
        )
        .await;
    } else if args.compose_only {
        // Compose-only mode: only add posts
        compose(
            &client,
            &addr,
            nodes,
            limit,
            args.num_compose, // Cast to usize
            args.print_every,
        )
        .await;
    } else {
        // Read edges (needed only if not in compose-only or timeline-only mode)
        let edges = get_edges(&format!(
            "../../datasets/social-graph/{}/{}.edges",
            args.graph, args.graph
        ));
        println!("Nodes: {}, Edges: {}", nodes, edges.len());

        // Run tasks
        register(&client, &addr, nodes, limit, args.print_every).await;
        follow(&client, &addr, &edges, limit, args.print_every).await;
        if args.compose {
            compose(
                &client,
                &addr,
                nodes,
                limit,
                args.num_compose,
                args.print_every,
            )
            .await;
        }
    }
}

fn get_num_nodes(file_path: &str) -> usize {
    let file = File::open(file_path).expect("Unable to open nodes file");
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("Unable to read line from nodes file");
    line.trim()
        .parse::<usize>()
        .expect("Unable to parse number of nodes")
}

fn get_edges(file_path: &str) -> Vec<(String, String)> {
    let file = File::open(file_path).expect("Unable to open edges file");
    let reader = BufReader::new(file);
    let mut edges = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Unable to read line from edges file");
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() >= 2 {
            edges.push((tokens[0].to_string(), tokens[1].to_string()));
        }
    }
    edges
}

async fn upload_register(
    client: &Client,
    addr: &str,
    user: &str,
) -> Result<String, reqwest::Error> {
    let url = format!("{}/wrk2-api/user/register", addr);
    let params = [
        ("first_name", format!("first_name_{}", user)),
        ("last_name", format!("last_name_{}", user)),
        ("username", format!("username_{}", user)),
        ("password", format!("password_{}", user)),
        ("user_id", user.to_string()),
    ];
    let res = client.post(&url).form(&params).send().await?;
    res.text().await
}

async fn upload_follow(
    client: &Client,
    addr: &str,
    user_0: &str,
    user_1: &str,
) -> Result<String, reqwest::Error> {
    let url = format!("{}/wrk2-api/user/follow", addr);
    let params = [
        ("user_name", format!("username_{}", user_0)),
        ("followee_name", format!("username_{}", user_1)),
    ];
    let res = client.post(&url).form(&params).send().await?;
    res.text().await
}

async fn upload_compose(
    client: &Client,
    addr: &str,
    user_id: usize,
    num_users: usize,
    rng: &mut StdRng,
) -> Result<String, reqwest::Error> {
    let mut text = (0..256)
        .map(|_| {
            let idx = rng.gen_range(0..(26 + 26 + 10));
            let c = if idx < 26 {
                (b'A' + idx as u8) as char
            } else if idx < 52 {
                (b'a' + (idx - 26) as u8) as char
            } else {
                (b'0' + (idx - 52) as u8) as char
            };
            c
        })
        .collect::<String>();
    // User mentions
    let mentions = rng.gen_range(0..=5);
    for _ in 0..mentions {
        let user = rng.gen_range(0..num_users);
        text += &format!(" @username_{}", user);
    }
    // URLs
    let urls = rng.gen_range(0..=5);
    for _ in 0..urls {
        let random_string: String = (0..64)
            .map(|_| {
                let idx = rng.gen_range(0..(26 + 10));
                let c = if idx < 26 {
                    (b'a' + idx as u8) as char
                } else {
                    (b'0' + (idx - 26) as u8) as char
                };
                c
            })
            .collect();
        text += &format!(" http://{}", random_string);
    }
    // Media
    let media_count = rng.gen_range(0..=5);
    let mut media_ids = Vec::new();
    let mut media_types = Vec::new();
    for _ in 0..media_count {
        let media_id: String = (0..18)
            .map(|_| (b'0' + rng.gen_range(0..10) as u8) as char)
            .collect();
        media_ids.push(format!("\"{}\"", media_id));
        media_types.push("\"png\"".to_string());
    }
    let payload = [
        ("username", format!("username_{}", user_id)),
        ("user_id", user_id.to_string()),
        ("text", text),
        ("media_ids", format!("[{}]", media_ids.join(","))),
        ("media_types", format!("[{}]", media_types.join(","))),
        ("post_type", "0".to_string()),
    ];
    let url = format!("{}/wrk2-api/post/compose", addr);
    let res = client.post(&url).form(&payload).send().await?;
    res.text().await
}

fn print_results(results: &[String]) {
    let mut error_count = 0;
    let mut last_error_message = String::new();

    for result in results {
        if !result.is_empty() && !result.starts_with("Success") {
            error_count += 1;
            last_error_message = result.clone();
        }
    }

    if error_count > 0 {
        println!("Total Errors: {}", error_count);
        // println!("Last Error Message: {}", last_error_message.trim());
    } else {
        println!("All operations succeeded.");
    }
}

async fn register(
    client: &Client,
    addr: &str,
    nodes: usize,
    limit: usize,
    print_every: usize,
) {
    println!("Registering Users...");
    let sem = Arc::new(Semaphore::new(limit));
    let mut handles = Vec::new();
    let mut results = Vec::new();
    for i in 0..nodes {
        let sem_clone = sem.clone();
        let permit = sem_clone.acquire_owned().await.unwrap();
        let client = client.clone();
        let addr = addr.to_string();
        let user = i.to_string();
        let handle: JoinHandle<String> = tokio::spawn(async move {
            let _permit = permit;
            upload_register(&client, &addr, &user)
                .await
                .unwrap_or_else(|e| e.to_string())
        });
        handles.push(handle);
        if handles.len() >= limit {
            // Wait for all handles in parallel
            let parallel_results = join_all(handles.drain(..)).await;
            for res in parallel_results {
                results.push(res.unwrap());
            }
            if i % print_every == 0 && i != 0 {
                println!("Registered {} users", i);
            }
        }
    }
    // Await any remaining handles
    if !handles.is_empty() {
        let parallel_results = join_all(handles.drain(..)).await;
        for res in parallel_results {
            results.push(res.unwrap());
        }
    }
    print_results(&results);
}

async fn follow(
    client: &Client,
    addr: &str,
    edges: &[(String, String)],
    limit: usize,
    print_every: usize,
) {
    println!("Adding follows...");
    let sem = Arc::new(Semaphore::new(limit));
    let mut handles = Vec::new();
    let mut results = Vec::new();
    let mut idx = 0;
    for edge in edges {
        // Create an owned vector of user pairs
        let user_pairs = vec![
            (edge.0.clone(), edge.1.clone()),
            (edge.1.clone(), edge.0.clone()),
        ];
        for (user_0, user_1) in user_pairs {
            let sem_clone = sem.clone();
            let permit = sem_clone.acquire_owned().await.unwrap();
            let client = client.clone();
            let addr = addr.to_string();
            let handle: JoinHandle<String> = tokio::spawn(async move {
                let _permit = permit;
                upload_follow(&client, &addr, &user_0, &user_1)
                    .await
                    .unwrap_or_else(|e| e.to_string())
            });
            handles.push(handle);
            idx += 1;
            if handles.len() >= limit {
                // Wait for all handles in parallel
                let parallel_results = join_all(handles.drain(..)).await;
                for res in parallel_results {
                    results.push(res.unwrap());
                }
                if idx % print_every == 0 {
                    println!("Processed {} follows", idx);
                }
            }
        }
    }
    // Await any remaining handles
    if !handles.is_empty() {
        let parallel_results = join_all(handles.drain(..)).await;
        for res in parallel_results {
            results.push(res.unwrap());
        }
    }
    print_results(&results);
}

async fn compose(
    client: &Client,
    addr: &str,
    nodes: usize,
    limit: usize,
    num_compose: usize,
    print_every: usize,
) {
    println!("Composing posts...");
    let sem = Arc::new(Semaphore::new(limit));
    let mut handles = Vec::new();
    let mut results = Vec::new();
    let mut idx = 0;
    let mut rng = StdRng::seed_from_u64(1); // Initialize rng here

    for user_id in 0..nodes {
        let num_requests = rng.gen_range(1..=2 * num_compose);
        for _ in 0..num_requests {
            let sem_clone = sem.clone();
            let permit = sem_clone.acquire_owned().await.unwrap();
            let client = client.clone();
            let addr = addr.to_string();
            let num_users = nodes;

            // Clone rng for the async task
            let mut rng_task = rng.clone();

            let handle: JoinHandle<String> = tokio::spawn(async move {
                let _permit = permit;
                upload_compose(&client, &addr, user_id, num_users, &mut rng_task)
                    .await
                    .unwrap_or_else(|e| e.to_string())
            });
            handles.push(handle);
            idx += 1;

            if handles.len() >= limit {
                // Await all handles in parallel
                let parallel_results = join_all(handles.drain(..)).await;
                for res in parallel_results {
                    results.push(res.unwrap());
                }
                if idx % print_every == 0 {
                    print_results(&results);
                    results.clear(); // Clear results to free up memory
                    println!("Performed {} compose", idx);
                }
            }
        }
    }

    // Await any remaining handles
    if !handles.is_empty() {
        let parallel_results = join_all(handles.drain(..)).await;
        for res in parallel_results {
            results.push(res.unwrap());
        }
    }
    print_results(&results);
}

async fn timeline(
    client: &Client,
    addr: &str,
    nodes: usize,
    limit: usize,
    theta: f64,
    num_requests: isize,
    print_every: usize,
) {
    println!("Performing user timeline reads...");
    let sem = Arc::new(Semaphore::new(limit));
    let mut handles = Vec::new();
    let mut results = Vec::new();
    let mut idx: usize = 0;
    let mut rng = StdRng::seed_from_u64(1); // Initialize rng here
    let zipf = Zipf::new(nodes as u64, theta).unwrap();

    let max_retries = 5;  // Maximum number of retry attempts

    loop {
        if num_requests != -1 && idx >= num_requests as usize {
            break;
        }

        let sem_clone = sem.clone();
        let permit = sem_clone.acquire_owned().await.unwrap();
        let client = client.clone();
        let addr = addr.to_string();

        // Sample user_id and other parameters
        let user_id = zipf.sample(&mut rng) as usize - 1; // Adjust to 0-based index
        let start = rng.gen_range(0..100);
        let stop = start + 10;

        let handle: JoinHandle<String> = tokio::spawn(async move {
            let _permit = permit;
            let mut retries = 0;
            let mut backoff_delay = Duration::from_millis(DEFAULT_DELAY_MS);  // Initial backoff delay (100ms)

            loop {
                let params = [
                    ("user_id", user_id.to_string()),
                    ("start", start.to_string()),
                    ("stop", stop.to_string()),
                ];
                let url = format!("{}/wrk2-api/user-timeline/read", addr);
                
                
                // Send the GET request
                let res = client
                    .get(&url)
                    .query(&params)
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .send()
                    .await;

                match res {
                    Ok(response) => {
                        if response.status().is_success() {
                            return "Success".to_string();
                        } else {
                            return format!("Failed with status code: {}", response.status());
                        }
                    }
                    Err(e) if retries < max_retries => {
                        sleep(backoff_delay).await;
                        backoff_delay *= 2; // Exponential backoff
                        retries += 1;
                    }
                    Err(e) => {
                        // If we've reached max retries, return the error
                        return format!("Error after {} retries: {}", max_retries, e);
                    }
                }
            }
        });
        handles.push(handle);
        idx += 1;

        if handles.len() >= limit {
            // Await all handles in parallel
            let parallel_results = join_all(handles.drain(..)).await;
            for res in parallel_results {
                results.push(res.unwrap());
            }
            if idx % print_every == 0 {
                print_results(&results);
                results.clear(); // Clear results to free up memory
                println!("Performed {} timeline reads", idx);
            }
        }
    }

    // Await any remaining handles
    if !handles.is_empty() {
        let parallel_results = join_all(handles.drain(..)).await;
        for res in parallel_results {
            results.push(res.unwrap());
        }
    }

    // Print and clear remaining results
    print_results(&results);
    results.clear(); // Clear final batch of results
}
