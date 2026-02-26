use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use miles_control_api::{JobRuntime, JobState, LoopPhase, RolloutCursor};
use miles_control_orchestrator::server::{build_router, ApiState, SharedStateStore};
use miles_control_orchestrator::{
    command_context, read_spec, should_run_periodic_action, transition,
};
use miles_control_scheduler::build_placement_plan;
use miles_control_state_store::{FileStateStore, InMemoryStateStore, StateStore};
use miles_control_worker_client::{MockWorkerClient, WorkerClient};
use std::sync::Arc;
use tracing::{info, Level};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "miles-control-orchestrator")]
#[command(about = "Phase-1 Rust control-plane orchestrator scaffold")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a JobSpec JSON file and print a placement plan.
    Validate {
        #[arg(long)]
        spec: String,
        #[arg(long, default_value_t = false)]
        use_critic: bool,
    },
    /// Run a no-op lifecycle for a JobSpec JSON file.
    Run {
        #[arg(long)]
        spec: String,
    },
    /// Serve HTTP control-plane APIs backed by persistent snapshots.
    Serve {
        #[arg(long, default_value = "127.0.0.1:18080")]
        bind: String,
        #[arg(long, default_value = "./snapshots")]
        snapshot_dir: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let cli = Cli::parse();

    match cli.command {
        Command::Validate { spec, use_critic } => validate_cmd(&spec, use_critic).await,
        Command::Run { spec } => run_cmd(&spec).await,
        Command::Serve { bind, snapshot_dir } => serve_cmd(&bind, &snapshot_dir).await,
    }
}

async fn validate_cmd(spec_path: &str, use_critic: bool) -> Result<()> {
    let spec = read_spec(spec_path).await?;
    spec.validate().context("jobspec validation failed")?;
    let placement = build_placement_plan(&spec.resource_plan, use_critic)?;
    info!(
        total_gpus = placement.total_gpus,
        actor = ?placement.actor_gpu_indices,
        critic = ?placement.critic_gpu_indices,
        rollout = ?placement.rollout_gpu_indices,
        "validated job spec and built placement plan"
    );
    Ok(())
}

async fn run_cmd(spec_path: &str) -> Result<()> {
    let spec = read_spec(spec_path).await?;
    spec.validate().context("jobspec validation failed")?;

    let store = InMemoryStateStore::default();
    let client = MockWorkerClient;
    let job_id = Uuid::new_v4();

    let mut runtime = JobRuntime {
        job_id,
        state: JobState::Created,
        phase: LoopPhase::Init,
        cursor: RolloutCursor {
            current_rollout_id: 0,
            total_rollouts: spec.train_config.num_rollout,
        },
        fence_token: 0,
        updated_at: Utc::now(),
    };
    store.put_job(spec.clone(), runtime.clone()).await?;

    transition(
        &store,
        &mut runtime,
        JobState::Starting,
        LoopPhase::Init,
        "job starting",
    )
    .await?;
    client
        .init(command_context(job_id, runtime.fence_token))
        .await
        .context("worker init failed")?;

    transition(
        &store,
        &mut runtime,
        JobState::Running,
        LoopPhase::UpdateWeights,
        "initial weight update",
    )
    .await?;
    client
        .update_weights(command_context(job_id, runtime.fence_token))
        .await
        .context("initial update_weights failed")?;

    for rollout_id in 0..spec.train_config.num_rollout {
        runtime.cursor.current_rollout_id = rollout_id;

        transition(
            &store,
            &mut runtime,
            JobState::Running,
            LoopPhase::Generate,
            "generate",
        )
        .await?;
        client
            .generate(command_context(job_id, runtime.fence_token))
            .await
            .context("generate failed")?;

        transition(
            &store,
            &mut runtime,
            JobState::Running,
            LoopPhase::Train,
            "train",
        )
        .await?;
        client
            .train_step(command_context(job_id, runtime.fence_token))
            .await
            .context("train step failed")?;

        transition(
            &store,
            &mut runtime,
            JobState::Running,
            LoopPhase::UpdateWeights,
            "update weights",
        )
        .await?;
        client
            .update_weights(command_context(job_id, runtime.fence_token))
            .await
            .context("update_weights failed")?;

        if should_run_periodic_action(rollout_id, spec.eval_config.eval_interval) {
            transition(
                &store,
                &mut runtime,
                JobState::Running,
                LoopPhase::Eval,
                "eval",
            )
            .await?;
            client
                .eval(command_context(job_id, runtime.fence_token))
                .await
                .context("eval failed")?;
        }

        if should_run_periodic_action(rollout_id, spec.save_config.save_interval) {
            transition(
                &store,
                &mut runtime,
                JobState::Running,
                LoopPhase::Save,
                "save",
            )
            .await?;
            client
                .save_checkpoint(command_context(job_id, runtime.fence_token))
                .await
                .context("save checkpoint failed")?;
        }
    }

    transition(
        &store,
        &mut runtime,
        JobState::Stopping,
        LoopPhase::Cleanup,
        "cleanup",
    )
    .await?;
    transition(
        &store,
        &mut runtime,
        JobState::Stopped,
        LoopPhase::Cleanup,
        "job stopped",
    )
    .await?;

    info!(job_id = %job_id, "no-op job lifecycle completed");
    Ok(())
}

async fn serve_cmd(bind: &str, snapshot_dir: &str) -> Result<()> {
    let store = Arc::new(
        FileStateStore::new(snapshot_dir)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
    ) as SharedStateStore;
    let state = ApiState::new(store)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    info!(
        bind = bind,
        snapshot_dir = snapshot_dir,
        "starting control-plane API server"
    );
    axum::serve(listener, app)
        .await
        .with_context(|| "control-plane API server failed")?;
    Ok(())
}
