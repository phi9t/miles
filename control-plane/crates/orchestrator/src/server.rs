use crate::lifecycle::{plan_action, JobAction, LifecycleDecision};
use crate::transition;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use miles_control_api::{JobEvent, JobRuntime, JobSpec, JobState, LoopPhase, RolloutCursor};
use miles_control_state_store::{StateStore, StoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub type SharedStateStore = Arc<dyn StateStore>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct JobMetrics {
    pub action_invocations_total: u64,
    pub transitions_total: u64,
    pub noop_actions_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    Healthy,
    Draining,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerRecord {
    pub worker_id: String,
    pub status: WorkerStatus,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct ApiState {
    store: SharedStateStore,
    events: Arc<RwLock<HashMap<Uuid, Vec<JobEvent>>>>,
    metrics: Arc<RwLock<HashMap<Uuid, JobMetrics>>>,
    workers: Arc<RwLock<HashMap<String, WorkerRecord>>>,
}

impl ApiState {
    pub async fn new(store: SharedStateStore) -> Result<Self, StoreError> {
        let jobs = store.list_jobs().await?;
        let mut events = HashMap::new();
        let mut metrics = HashMap::new();

        for job in jobs {
            events.insert(
                job.runtime.job_id,
                vec![JobEvent {
                    job_id: job.runtime.job_id,
                    phase: job.runtime.phase,
                    state: job.runtime.state,
                    message: "job restored from snapshot".to_string(),
                    timestamp: Utc::now(),
                }],
            );
            metrics.insert(job.runtime.job_id, JobMetrics::default());
        }

        Ok(Self {
            store,
            events: Arc::new(RwLock::new(events)),
            metrics: Arc::new(RwLock::new(metrics)),
            workers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    async fn add_event(&self, runtime: &JobRuntime, message: String) {
        let mut guard = self.events.write().await;
        let events = guard.entry(runtime.job_id).or_default();
        events.push(JobEvent {
            job_id: runtime.job_id,
            phase: runtime.phase,
            state: runtime.state,
            message,
            timestamp: Utc::now(),
        });
    }

    async fn with_metrics<F>(&self, job_id: Uuid, f: F)
    where
        F: FnOnce(&mut JobMetrics),
    {
        let mut guard = self.metrics.write().await;
        let metrics = guard.entry(job_id).or_default();
        f(metrics);
    }
}

#[derive(Debug, Serialize)]
struct CreateJobResponse {
    job_id: Uuid,
    runtime: JobRuntime,
}

#[derive(Debug, Serialize)]
struct ActionResponse {
    job_id: Uuid,
    state: JobState,
    phase: LoopPhase,
    status: &'static str,
    message: String,
}

#[derive(Debug, Serialize)]
struct StateResponse {
    job_id: Uuid,
    runtime: JobRuntime,
}

#[derive(Debug, Serialize)]
struct EventsResponse {
    job_id: Uuid,
    events: Vec<JobEvent>,
}

#[derive(Debug, Serialize)]
struct MetricsResponse {
    job_id: Uuid,
    metrics: JobMetrics,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error_code: String,
    message: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    error_code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_code: "bad_request",
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            error_code: "invalid_transition",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_code: "internal_error",
            message: message.into(),
        }
    }

    fn from_store(err: StoreError) -> Self {
        match err {
            StoreError::NotFound(_) => Self {
                status: StatusCode::NOT_FOUND,
                error_code: "not_found",
                message: err.to_string(),
            },
            _ => Self::internal(err.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error_code: self.error_code.to_string(),
                message: self.message,
            }),
        )
            .into_response()
    }
}

#[derive(Debug, Deserialize)]
struct WorkerRegisterRequest {
    worker_id: Option<String>,
}

pub fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/:job_id/start", post(start_job))
        .route("/v1/jobs/:job_id/pause", post(pause_job))
        .route("/v1/jobs/:job_id/resume", post(resume_job))
        .route("/v1/jobs/:job_id/stop", post(stop_job))
        .route("/v1/jobs/:job_id/state", get(get_job_state))
        .route("/v1/jobs/:job_id/events", get(get_job_events))
        .route("/v1/jobs/:job_id/metrics", get(get_job_metrics))
        .route("/v1/workers/register", post(register_worker))
        .route("/v1/workers/:worker_id/heartbeat", post(worker_heartbeat))
        .route("/v1/workers/:worker_id/drain", post(worker_drain))
        .route("/v1/workers", get(list_workers))
        .with_state(state)
}

async fn create_job(
    State(state): State<ApiState>,
    Json(spec): Json<JobSpec>,
) -> Result<(StatusCode, Json<CreateJobResponse>), ApiError> {
    spec.validate()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let job_id = Uuid::new_v4();
    let runtime = JobRuntime {
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

    state
        .store
        .put_job(spec, runtime.clone())
        .await
        .map_err(ApiError::from_store)?;
    state
        .store
        .snapshot_job(job_id)
        .await
        .map_err(ApiError::from_store)?;

    state.add_event(&runtime, "job created".to_string()).await;
    state.with_metrics(job_id, |_| {}).await;

    Ok((
        StatusCode::CREATED,
        Json(CreateJobResponse { job_id, runtime }),
    ))
}

async fn apply_lifecycle_action(
    state: &ApiState,
    job_id: Uuid,
    action: JobAction,
) -> Result<ActionResponse, ApiError> {
    let stored = state
        .store
        .get_job(job_id)
        .await
        .map_err(ApiError::from_store)?;
    let mut runtime = stored.runtime;

    state
        .with_metrics(job_id, |m| {
            m.action_invocations_total += 1;
        })
        .await;

    match plan_action(runtime.state, action).map_err(|e| ApiError::conflict(e.to_string()))? {
        LifecycleDecision::Noop(message) => {
            state
                .with_metrics(job_id, |m| {
                    m.noop_actions_total += 1;
                })
                .await;
            Ok(ActionResponse {
                job_id,
                state: runtime.state,
                phase: runtime.phase,
                status: "noop",
                message: message.to_string(),
            })
        }
        LifecycleDecision::Apply(steps) => {
            for step in steps {
                transition(
                    state.store.as_ref(),
                    &mut runtime,
                    step.state,
                    step.phase,
                    step.message,
                )
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
                state
                    .with_metrics(job_id, |m| {
                        m.transitions_total += 1;
                    })
                    .await;
                state.add_event(&runtime, step.message.to_string()).await;
            }
            state
                .store
                .snapshot_job(job_id)
                .await
                .map_err(ApiError::from_store)?;
            Ok(ActionResponse {
                job_id,
                state: runtime.state,
                phase: runtime.phase,
                status: "updated",
                message: format!("{} applied", action.as_str()),
            })
        }
    }
}

async fn start_job(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<ActionResponse>, ApiError> {
    let resp = apply_lifecycle_action(&state, job_id, JobAction::Start).await?;
    Ok(Json(resp))
}

async fn pause_job(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<ActionResponse>, ApiError> {
    let resp = apply_lifecycle_action(&state, job_id, JobAction::Pause).await?;
    Ok(Json(resp))
}

async fn resume_job(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<ActionResponse>, ApiError> {
    let resp = apply_lifecycle_action(&state, job_id, JobAction::Resume).await?;
    Ok(Json(resp))
}

async fn stop_job(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<ActionResponse>, ApiError> {
    let resp = apply_lifecycle_action(&state, job_id, JobAction::Stop).await?;
    Ok(Json(resp))
}

async fn get_job_state(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<StateResponse>, ApiError> {
    let stored = state
        .store
        .get_job(job_id)
        .await
        .map_err(ApiError::from_store)?;
    Ok(Json(StateResponse {
        job_id,
        runtime: stored.runtime,
    }))
}

async fn get_job_events(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<EventsResponse>, ApiError> {
    state
        .store
        .get_job(job_id)
        .await
        .map_err(ApiError::from_store)?;

    let events = state
        .events
        .read()
        .await
        .get(&job_id)
        .cloned()
        .unwrap_or_default();
    Ok(Json(EventsResponse { job_id, events }))
}

async fn get_job_metrics(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<MetricsResponse>, ApiError> {
    state
        .store
        .get_job(job_id)
        .await
        .map_err(ApiError::from_store)?;

    let metrics = state
        .metrics
        .read()
        .await
        .get(&job_id)
        .cloned()
        .unwrap_or_default();
    Ok(Json(MetricsResponse { job_id, metrics }))
}

async fn register_worker(
    State(state): State<ApiState>,
    Json(req): Json<WorkerRegisterRequest>,
) -> Json<WorkerRecord> {
    let worker_id = req
        .worker_id
        .unwrap_or_else(|| format!("worker-{}", Uuid::new_v4()));
    let record = WorkerRecord {
        worker_id: worker_id.clone(),
        status: WorkerStatus::Healthy,
        updated_at: Utc::now(),
    };

    state
        .workers
        .write()
        .await
        .insert(worker_id, record.clone());
    Json(record)
}

async fn worker_heartbeat(
    State(state): State<ApiState>,
    Path(worker_id): Path<String>,
) -> Result<Json<WorkerRecord>, ApiError> {
    let mut guard = state.workers.write().await;
    let Some(record) = guard.get_mut(&worker_id) else {
        return Err(ApiError {
            status: StatusCode::NOT_FOUND,
            error_code: "not_found",
            message: format!("worker {worker_id} not found"),
        });
    };

    record.status = WorkerStatus::Healthy;
    record.updated_at = Utc::now();
    Ok(Json(record.clone()))
}

async fn worker_drain(
    State(state): State<ApiState>,
    Path(worker_id): Path<String>,
) -> Result<Json<WorkerRecord>, ApiError> {
    let mut guard = state.workers.write().await;
    let Some(record) = guard.get_mut(&worker_id) else {
        return Err(ApiError {
            status: StatusCode::NOT_FOUND,
            error_code: "not_found",
            message: format!("worker {worker_id} not found"),
        });
    };

    record.status = WorkerStatus::Draining;
    record.updated_at = Utc::now();
    Ok(Json(record.clone()))
}

async fn list_workers(State(state): State<ApiState>) -> Json<Vec<WorkerRecord>> {
    let workers = state
        .workers
        .read()
        .await
        .values()
        .cloned()
        .collect::<Vec<_>>();
    Json(workers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use miles_control_api::{
        DatasetConfig, EvalConfig, FaultToleranceConfig, JobMode, ResourcePlan, RolloutConfig,
        SaveConfig, TrainBackend, TrainConfig,
    };
    use miles_control_state_store::FileStateStore;

    fn sample_spec() -> JobSpec {
        JobSpec {
            job_name: "api-job".to_string(),
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
                prompt_data: "/tmp/prompts.jsonl".to_string(),
                input_key: "prompt".to_string(),
                label_key: "label".to_string(),
            },
            rollout_config: RolloutConfig {
                batch_size: 4,
                n_samples_per_prompt: 2,
                max_response_len: Some(128),
            },
            train_config: TrainConfig {
                global_batch_size: 8,
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

    fn temp_snapshot_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("miles-control-api-{}", Uuid::new_v4()))
    }

    async fn spawn_server(snapshot_dir: &std::path::Path) -> (String, tokio::task::JoinHandle<()>) {
        let store = Arc::new(FileStateStore::new(snapshot_dir).await.expect("file store"))
            as SharedStateStore;
        let state = ApiState::new(store).await.expect("state");
        let app = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server run");
        });

        (format!("http://{}", addr), handle)
    }

    async fn wait_for_server(base: &str) {
        let client = reqwest::Client::new();
        for _ in 0..40 {
            let res = client.get(format!("{base}/v1/workers")).send().await;
            if res.is_ok() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("server did not become ready");
    }

    #[tokio::test]
    async fn create_start_stop_and_read_state() {
        let dir = temp_snapshot_dir();
        let (base, handle) = spawn_server(&dir).await;
        wait_for_server(&base).await;
        let client = reqwest::Client::new();

        let create_resp = client
            .post(format!("{base}/v1/jobs"))
            .json(&sample_spec())
            .send()
            .await
            .expect("create request");
        assert_eq!(create_resp.status(), StatusCode::CREATED);
        let created: serde_json::Value = create_resp.json().await.expect("create body");
        let job_id = created["job_id"]
            .as_str()
            .expect("job_id string")
            .to_string();

        let start_resp = client
            .post(format!("{base}/v1/jobs/{job_id}/start"))
            .send()
            .await
            .expect("start request");
        assert_eq!(start_resp.status(), StatusCode::OK);

        let stop_resp = client
            .post(format!("{base}/v1/jobs/{job_id}/stop"))
            .send()
            .await
            .expect("stop request");
        assert_eq!(stop_resp.status(), StatusCode::OK);

        let state_resp = client
            .get(format!("{base}/v1/jobs/{job_id}/state"))
            .send()
            .await
            .expect("state request");
        assert_eq!(state_resp.status(), StatusCode::OK);
        let state_json: serde_json::Value = state_resp.json().await.expect("state body");
        assert_eq!(state_json["runtime"]["state"], "stopped");

        let events_resp = client
            .get(format!("{base}/v1/jobs/{job_id}/events"))
            .send()
            .await
            .expect("events request");
        assert_eq!(events_resp.status(), StatusCode::OK);
        let events_json: serde_json::Value = events_resp.json().await.expect("events body");
        let events_len = events_json["events"]
            .as_array()
            .expect("events array")
            .len();
        assert!(events_len >= 3);

        let metrics_resp = client
            .get(format!("{base}/v1/jobs/{job_id}/metrics"))
            .send()
            .await
            .expect("metrics request");
        assert_eq!(metrics_resp.status(), StatusCode::OK);
        let metrics_json: serde_json::Value = metrics_resp.json().await.expect("metrics body");
        assert_eq!(metrics_json["metrics"]["action_invocations_total"], 2);

        handle.abort();
        let _ = handle.await;
        tokio::fs::remove_dir_all(&dir).await.expect("cleanup dir");
    }

    #[tokio::test]
    async fn start_and_stop_are_idempotent() {
        let dir = temp_snapshot_dir();
        let (base, handle) = spawn_server(&dir).await;
        wait_for_server(&base).await;
        let client = reqwest::Client::new();

        let created: serde_json::Value = client
            .post(format!("{base}/v1/jobs"))
            .json(&sample_spec())
            .send()
            .await
            .expect("create")
            .json()
            .await
            .expect("create body");
        let job_id = created["job_id"].as_str().expect("job_id");

        let start_one: serde_json::Value = client
            .post(format!("{base}/v1/jobs/{job_id}/start"))
            .send()
            .await
            .expect("start one")
            .json()
            .await
            .expect("start one body");
        assert_eq!(start_one["status"], "updated");

        let start_two: serde_json::Value = client
            .post(format!("{base}/v1/jobs/{job_id}/start"))
            .send()
            .await
            .expect("start two")
            .json()
            .await
            .expect("start two body");
        assert_eq!(start_two["status"], "noop");

        let stop_one: serde_json::Value = client
            .post(format!("{base}/v1/jobs/{job_id}/stop"))
            .send()
            .await
            .expect("stop one")
            .json()
            .await
            .expect("stop one body");
        assert_eq!(stop_one["status"], "updated");

        let stop_two: serde_json::Value = client
            .post(format!("{base}/v1/jobs/{job_id}/stop"))
            .send()
            .await
            .expect("stop two")
            .json()
            .await
            .expect("stop two body");
        assert_eq!(stop_two["status"], "noop");

        handle.abort();
        let _ = handle.await;
        tokio::fs::remove_dir_all(&dir).await.expect("cleanup dir");
    }

    #[tokio::test]
    async fn state_recovers_after_server_restart_from_snapshots() {
        let dir = temp_snapshot_dir();

        let (base_one, handle_one) = spawn_server(&dir).await;
        wait_for_server(&base_one).await;
        let client = reqwest::Client::new();

        let created: serde_json::Value = client
            .post(format!("{base_one}/v1/jobs"))
            .json(&sample_spec())
            .send()
            .await
            .expect("create")
            .json()
            .await
            .expect("create body");
        let job_id = created["job_id"].as_str().expect("job_id").to_string();

        client
            .post(format!("{base_one}/v1/jobs/{job_id}/start"))
            .send()
            .await
            .expect("start");
        client
            .post(format!("{base_one}/v1/jobs/{job_id}/stop"))
            .send()
            .await
            .expect("stop");

        handle_one.abort();
        let _ = handle_one.await;

        let (base_two, handle_two) = spawn_server(&dir).await;
        wait_for_server(&base_two).await;

        let state_resp = client
            .get(format!("{base_two}/v1/jobs/{job_id}/state"))
            .send()
            .await
            .expect("state after restart");
        assert_eq!(state_resp.status(), StatusCode::OK);
        let state_json: serde_json::Value = state_resp.json().await.expect("state body");
        assert_eq!(state_json["runtime"]["state"], "stopped");

        handle_two.abort();
        let _ = handle_two.await;
        tokio::fs::remove_dir_all(&dir).await.expect("cleanup dir");
    }
}
