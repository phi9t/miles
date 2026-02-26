use anyhow::{Context, Result};
use chrono::Utc;
use miles_control_api::{JobRuntime, JobSpec, JobState, LoopPhase};
use miles_control_state_store::StateStore;
use miles_control_worker_client::CommandContext;
use tracing::info;
use uuid::Uuid;

pub mod lifecycle;
pub mod server;

pub fn command_context(job_id: Uuid, fence_token: u64) -> CommandContext {
    CommandContext {
        job_id,
        request_id: Uuid::new_v4(),
        attempt_id: 1,
        deadline_ms: 30_000 + fence_token,
    }
}

pub async fn transition<S: StateStore + ?Sized>(
    store: &S,
    runtime: &mut JobRuntime,
    state: JobState,
    phase: LoopPhase,
    message: &str,
) -> Result<()> {
    runtime.state = state;
    runtime.phase = phase;
    runtime.fence_token += 1;
    runtime.updated_at = Utc::now();
    store.update_runtime(runtime.clone()).await?;

    info!(
        job_id = %runtime.job_id,
        ?state,
        ?phase,
        rollout_id = runtime.cursor.current_rollout_id,
        fence = runtime.fence_token,
        "{}",
        message
    );
    Ok(())
}

pub async fn read_spec(spec_path: &str) -> Result<JobSpec> {
    let data = tokio::fs::read_to_string(spec_path)
        .await
        .with_context(|| format!("failed to read spec file at {}", spec_path))?;
    let spec =
        serde_json::from_str::<JobSpec>(&data).with_context(|| "failed to parse JobSpec JSON")?;
    Ok(spec)
}

pub fn should_run_periodic_action(rollout_id: u32, interval: Option<u32>) -> bool {
    match interval {
        Some(i) if i > 0 => (rollout_id + 1) % i == 0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miles_control_api::{
        DatasetConfig, EvalConfig, FaultToleranceConfig, JobMode, ResourcePlan, RolloutConfig,
        RolloutCursor, SaveConfig, TrainBackend, TrainConfig,
    };
    use miles_control_state_store::InMemoryStateStore;
    use std::fs;

    fn sample_spec() -> JobSpec {
        JobSpec {
            job_name: "test-job".to_string(),
            mode: JobMode::Sync,
            backend: TrainBackend::Fsdp,
            resource_plan: ResourcePlan {
                actor_num_nodes: 1,
                actor_num_gpus_per_node: 1,
                critic_num_nodes: None,
                critic_num_gpus_per_node: None,
                rollout_num_gpus: 1,
                rollout_num_gpus_per_engine: 1,
                num_gpus_per_node: 8,
                colocate: false,
            },
            dataset_config: DatasetConfig {
                prompt_data: "/tmp/p.jsonl".to_string(),
                input_key: "prompt".to_string(),
                label_key: "label".to_string(),
            },
            rollout_config: RolloutConfig {
                batch_size: 2,
                n_samples_per_prompt: 2,
                max_response_len: Some(256),
            },
            train_config: TrainConfig {
                global_batch_size: 4,
                num_rollout: 3,
                update_weights_interval: 1,
            },
            eval_config: EvalConfig {
                eval_interval: Some(2),
            },
            save_config: SaveConfig {
                save_interval: Some(2),
                output_dir: Some("/tmp/out".to_string()),
            },
            fault_tolerance_config: FaultToleranceConfig {
                enabled: true,
                max_retries: 2,
                retry_backoff_ms: 100,
            },
        }
    }

    #[test]
    fn periodic_action_boundary_matrix() {
        assert!(!should_run_periodic_action(0, None));
        assert!(!should_run_periodic_action(0, Some(0)));
        assert!(should_run_periodic_action(0, Some(1)));
        assert!(!should_run_periodic_action(0, Some(3)));
        assert!(should_run_periodic_action(2, Some(3)));
    }

    #[test]
    fn command_context_has_defaults_and_non_nil_ids() {
        let job_id = Uuid::new_v4();
        let ctx = command_context(job_id, 42);
        assert_eq!(ctx.job_id, job_id);
        assert_eq!(ctx.attempt_id, 1);
        assert_eq!(ctx.deadline_ms, 30_042);
        assert_ne!(ctx.request_id, Uuid::nil());
    }

    #[tokio::test]
    async fn read_spec_success_and_roundtrip() {
        let spec = sample_spec();
        let path = format!("/tmp/miles-test-spec-{}.json", Uuid::new_v4());
        let raw = serde_json::to_string(&spec).expect("serialize spec");
        fs::write(&path, raw).expect("write temp spec");

        let parsed = read_spec(&path).await.expect("read spec");
        fs::remove_file(path).expect("cleanup temp spec");
        assert_eq!(parsed, spec);
    }

    #[tokio::test]
    async fn read_spec_missing_file_has_contextual_error() {
        let missing = format!("/tmp/miles-missing-{}.json", Uuid::new_v4());
        let err = read_spec(&missing)
            .await
            .expect_err("missing file should fail");
        let msg = err.to_string();
        assert!(msg.contains("failed to read spec file"));
    }

    #[tokio::test]
    async fn read_spec_invalid_json_has_contextual_error() {
        let path = format!("/tmp/miles-invalid-{}.json", Uuid::new_v4());
        fs::write(&path, "{not json").expect("write invalid json");
        let err = read_spec(&path)
            .await
            .expect_err("invalid json should fail");
        fs::remove_file(path).expect("cleanup temp invalid json");
        let msg = err.to_string();
        assert!(msg.contains("failed to parse JobSpec JSON"));
    }

    #[tokio::test]
    async fn transition_updates_runtime_and_persists_state() {
        let store = InMemoryStateStore::default();
        let spec = sample_spec();
        let mut runtime = JobRuntime {
            job_id: Uuid::new_v4(),
            state: JobState::Created,
            phase: LoopPhase::Init,
            cursor: RolloutCursor {
                current_rollout_id: 0,
                total_rollouts: 3,
            },
            fence_token: 10,
            updated_at: Utc::now(),
        };
        let old_ts = runtime.updated_at;
        store
            .put_job(spec, runtime.clone())
            .await
            .expect("insert runtime");

        transition(
            &store,
            &mut runtime,
            JobState::Running,
            LoopPhase::Generate,
            "go",
        )
        .await
        .expect("transition");

        assert_eq!(runtime.state, JobState::Running);
        assert_eq!(runtime.phase, LoopPhase::Generate);
        assert_eq!(runtime.fence_token, 11);
        assert!(runtime.updated_at >= old_ts);

        let stored = store.get_job(runtime.job_id).await.expect("stored runtime");
        assert_eq!(stored.runtime.state, JobState::Running);
        assert_eq!(stored.runtime.phase, LoopPhase::Generate);
        assert_eq!(stored.runtime.fence_token, 11);
    }
}
