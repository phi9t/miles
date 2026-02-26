use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommandContext {
    pub job_id: Uuid,
    pub request_id: Uuid,
    pub attempt_id: u32,
    pub deadline_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandStatus {
    Ok,
    RetryableError,
    FatalError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandResult {
    pub status: CommandStatus,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[async_trait]
pub trait WorkerClient: Send + Sync {
    async fn init(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn generate(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn train_step(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn update_weights(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn eval(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn save_checkpoint(&self, ctx: CommandContext) -> Result<CommandResult>;
    async fn health(&self) -> Result<CommandResult>;
}

#[derive(Debug, Default, Clone)]
pub struct MockWorkerClient;

impl MockWorkerClient {
    fn ok(msg: &str) -> CommandResult {
        CommandResult {
            status: CommandStatus::Ok,
            message: msg.to_string(),
            timestamp: Utc::now(),
        }
    }
}

#[async_trait]
impl WorkerClient for MockWorkerClient {
    async fn init(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("init"))
    }

    async fn generate(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("generate"))
    }

    async fn train_step(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("train_step"))
    }

    async fn update_weights(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("update_weights"))
    }

    async fn eval(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("eval"))
    }

    async fn save_checkpoint(&self, _ctx: CommandContext) -> Result<CommandResult> {
        Ok(Self::ok("save_checkpoint"))
    }

    async fn health(&self) -> Result<CommandResult> {
        Ok(Self::ok("health"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    fn sample_ctx() -> CommandContext {
        CommandContext {
            job_id: Uuid::new_v4(),
            request_id: Uuid::new_v4(),
            attempt_id: 2,
            deadline_ms: 15_000,
        }
    }

    #[tokio::test]
    async fn mock_worker_methods_return_ok_with_expected_messages() {
        let client = MockWorkerClient;
        let ctx = sample_ctx();

        let init = client.init(ctx.clone()).await.expect("init");
        let generate = client.generate(ctx.clone()).await.expect("generate");
        let train = client.train_step(ctx.clone()).await.expect("train");
        let update = client.update_weights(ctx.clone()).await.expect("update");
        let eval = client.eval(ctx.clone()).await.expect("eval");
        let save = client.save_checkpoint(ctx).await.expect("save");
        let health = client.health().await.expect("health");

        for (result, msg) in [
            (init, "init"),
            (generate, "generate"),
            (train, "train_step"),
            (update, "update_weights"),
            (eval, "eval"),
            (save, "save_checkpoint"),
            (health, "health"),
        ] {
            assert_eq!(result.status, CommandStatus::Ok);
            assert_eq!(result.message, msg);
            assert!(result.timestamp <= Utc::now() + Duration::seconds(1));
        }
    }

    #[test]
    fn command_context_roundtrip_json() {
        let ctx = sample_ctx();
        let val = serde_json::to_value(&ctx).expect("serialize context");
        let back: CommandContext = serde_json::from_value(val).expect("deserialize context");
        assert_eq!(back, ctx);
    }

    #[test]
    fn command_status_uses_snake_case() {
        assert_eq!(
            serde_json::to_value(CommandStatus::RetryableError).expect("serialize"),
            json!("retryable_error")
        );
        assert_eq!(
            serde_json::to_value(CommandStatus::FatalError).expect("serialize"),
            json!("fatal_error")
        );
    }
}
