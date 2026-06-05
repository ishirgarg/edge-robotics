# network/ — (stub, future work)

Placeholder for simulating network conditions between a robot client and a remote policy
server (latency, jitter, bandwidth caps, packet loss), so we can study edge deployments
where inference is offloaded.

openpi already ships a websocket policy server (`openpi.serving.websocket_policy_server`)
and a client (`packages/openpi-client`). The plan is to add a thin shim here that wraps the
client transport with a configurable delay/bandwidth model and records client-vs-server-vs-
policy timing (the server already returns `server_timing`/`policy_timing`).

Nothing here yet — the current profiler runs the model locally with dummy inputs.
