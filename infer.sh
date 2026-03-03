rm -rf ~/.cache/huggingface/lerobot/Sophon96/eval_pour-milk

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM3 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=COM4 \
    --teleop.id=leader \
    --display_data=false \
    --dataset.repo_id=Sophon96/eval_pour-milk \
    --dataset.single_task="pour milk" \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=2 \
    --policy.path=Sophon96/pour-milk-policy \
    --cosmos_safety.enabled=True
