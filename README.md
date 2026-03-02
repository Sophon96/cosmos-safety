# NVIDIA Cosmos Cookoff Hackathon Submission

## Motivation

Our goal is to create robotic systems that can pour liquids in a secure way. 
- Robotic labs and lab-in-the-loop AI will be the future of biology, and integrating models that reason about potential failures before they happen will ensure productivity gains don't sacrifice safety. 
- We recognize that specialized fine tuning scenarios can be limited in resources, and we aim to demonstrate how Nvidia's Cosmos models can be used to fill this gap. 

## Implementation

- We use the LeRobot So101 with an 80M parameter ACT model to be the brain of the robot.  
- We integrate Cosmos Reason to reason about trajectories on short time frames and determines whether the pouring trajectory will be successful. An unviable trajectory will be paused before the pouring commences.

### Cosmos Safety Monitor

When running `lerobot-record` with a policy, enable the Cosmos safety monitor with `--cosmos_safety.enabled=True`. This will:

1. **Binary check** (1 token, ~1 sec interval): Detect if the robot is about to pour water
2. **Pause**: When detected, the robot holds its current position
3. **Full reasoning**: Cosmos Reason analyzes the trajectory to determine if it is on track
4. **Resume/Abort**: If the trajectory is viable, the robot resumes; otherwise it remains paused

Example:
```bash
lerobot-record --robot.type=so100_follower ... --policy.path=your/policy --cosmos_safety.enabled=True
```

Ensure you run from the cosmos project root and have the modified lerobot installed (`pip install -e ./lerobot`).

### Remote Cosmos VLM (cloud inference)

To run Cosmos on a cloud server while LeRobot runs locally:

**On cloud:**
```bash
cd cosmos && python -m uvicorn reason_server:app --host 0.0.0.0 --port 8000
```

**On local** (with SSH tunnel):
```bash
ssh -L 8000:localhost:8000 user@cloud-ip   # separate terminal
export COSMOS_REMOTE_URL=http://127.0.0.1:8000
# then run lerobot-record with --cosmos_safety.enabled=True
```

Without tunnel (cloud has public IP): `export COSMOS_REMOTE_URL=http://<cloud-ip>:8000`

## Evals

We hope to demonstrate the differences in accuracy between the Cosmos-integrated model and base ACT on 20 instances of liquid pouring. 
- One labeled cup will contain liquid to be poured into the second cup.
- The two cups will be positioned at random per iteration.
- The two policies will be evaluated on accuracy of pouring (Pass/Fail)


